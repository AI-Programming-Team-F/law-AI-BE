import threading

import numpy
import pandas
from konlpy.tag import Okt

import log
import prep
import w2v

# 수집·전처리한 법률안 데이터를 메모리에 보관하는 싱글톤
# - 어디서 Data()를 호출해도 같은 인스턴스를 돌려준다.
# - 내부 상태(_df 등)는 직접 바꾸지 못하게 숨기고, 읽기 전용 속성과 메서드로만 접근한다.
class Data:
    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls):
        with cls._instance_lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._df = None
                instance._word2id = {}
                instance._id2word = {}
                instance._okt = None
                instance._stopwords = set()
                instance._wv = None            # 학습된 단어 벡터 (w2v.WordVectors)
                instance._doc_vectors = None   # 법률안 벡터 (법안 수, dim), _df와 행 순서 같음
                instance._status = 'idle'      # idle | running | done | failed
                instance._error = None
                instance._fetched = 0          # 요약본 수집 진행률 (%)
                instance._run_lock = threading.Lock()
                cls._instance = instance
        return cls._instance

    # ---------- 읽기 전용 속성 ----------

    # 검색에 쓸 데이터가 준비되었는지
    @property
    def is_ready(self) -> bool:
        return self._df is not None

    # 전처리 진행 상태
    @property
    def status(self) -> str:
        return self._status

    @property
    def error(self) -> str | None:
        return self._error

    # 요약본 수집 진행률 (0~100, 단위: %)
    @property
    def fetched(self) -> int:
        return self._fetched

    # DataFrame (복사본을 돌려주므로 밖에서 수정해도 내부 데이터는 바뀌지 않음)
    @property
    def df(self) -> pandas.DataFrame | None:
        return None if self._df is None else self._df.copy(deep=False)

    @property
    def word2id(self) -> dict[str, int]:
        return dict(self._word2id)

    @property
    def id2word(self) -> dict[int, str]:
        return dict(self._id2word)

    # ---------- 전처리 ----------

    # 백그라운드 스레드에서 전처리 시작
    # 이미 진행 중이면 새로 시작하지 않고 False 반환
    def start_preprocess(self, size: int = 2000) -> bool:
        if not self._run_lock.acquire(blocking=False):
            return False

        # 스레드가 시작되기 전에 상태를 바꿔 둬야 바로 반환되는 응답에도 반영됨
        self._status = 'running'
        self._error = None
        self._fetched = 0

        def run():
            try:
                self._preprocess(size)
            finally:
                self._run_lock.release()

        threading.Thread(target=run, daemon=True).start()
        return True

    # 수집 → 명사 추출 → 불용어 제거 → 단어 사전 생성
    # 끝까지 성공했을 때만 내부 데이터를 한 번에 교체한다 (진행 중에도 기존 데이터로 검색 가능)
    def _preprocess(self, size: int) -> None:
        try:
            log.step('형태소 분석기(Okt) 로딩')
            okt = self._okt or Okt()

            # 의안 번호 가져오기
            log.step(f'의안 목록 수집 시작 (목표 {size}건)')
            bills = prep.get_bills_many(22, size)
            if bills is None:
                raise RuntimeError('의안 목록 수집 실패')
            log.step(f'의안 목록 수집 완료 ({len(bills)}건)')

            # pandas DataFrame으로 변환 (필요한 열만 선택)
            df = pandas.DataFrame(bills)[['BILL_NAME', 'BILL_NO', 'PROC_RESULT']]

            # 의안 번호마다 요약본 추출
            total = len(df)
            count = 0

            def fetch_summary(bill_no: str) -> str | None:
                nonlocal count
                summary = prep.get_summary(bill_no)
                count += 1
                self._fetched = count * 100 // total
                return summary

            df['SUMMARY'] = df['BILL_NO'].apply(log.with_progress(fetch_summary, total, '요약본 수집'))

            # 요약본의 명사들을 추출하여 열 추가 (불용어 필터링)
            df['NOUNS'] = df['SUMMARY'].fillna('').apply(log.with_progress(okt.nouns, len(df), '명사 추출'))
            stopwords = list(df['NOUNS'].explode().value_counts().head(20).keys())
            for x in ['제안', '이유', '현행법', '최근', '법']: stopwords.append(x)
            df['NOUNS'] = df['NOUNS'].apply(lambda nouns: list(filter(lambda x: x not in stopwords and len(x) > 1, nouns)))

            # 없는 행 삭제
            before = len(df)
            df = df[df['SUMMARY'].notna() & (df['NOUNS'].str.len() > 0)].reset_index(drop=True)
            log.step(f'요약본/명사 없는 행 삭제: {before}건 → {len(df)}건')

            # 단어 사전 생성
            df, word2id, id2word = w2v.build_vocab(df, min_count=3)
            log.step(f'단어 사전 생성 완료 ({len(word2id)}개)')

            # Word2Vec 학습 (TensorFlow)
            log.step(f'Word2Vec 학습 시작 (단어 {len(word2id)}개)')
            vectors = w2v.train(df['IDS'], vocab_size=len(word2id) + 1)
            wv = w2v.WordVectors(vectors, word2id, id2word)
            doc_vectors = wv.fit_documents(df['IDS'])
            log.step('Word2Vec 학습 완료')

            # 완성된 데이터로 한 번에 교체
            self._df, self._word2id, self._id2word = df, word2id, id2word
            self._okt, self._stopwords, self._wv, self._doc_vectors = okt, set(stopwords), wv, doc_vectors
            self._status = 'done'
        except Exception as e:
            self._status = 'failed'
            self._error = str(e)
            log.step(f'전처리 실패: {e}')

    # ---------- 검색 ----------

    # 의안 번호로 원문 요약본(제안이유 및 주요내용) 찾기
    # 메모리에 있으면 그대로 쓰고, 없으면 국회 API에서 바로 가져온다.
    def get_summary(self, bill_no: str) -> str | None:
        df = self._df
        if df is not None:
            row = df.loc[df['BILL_NO'] == bill_no, 'SUMMARY']
            if not row.empty:
                return row.iloc[0]
        return prep.get_summary(bill_no)

    # 최종 점수 = KEYWORD_WEIGHT × 키워드 점수 + (1 - KEYWORD_WEIGHT) × 의미 점수 (0~1)
    # - 검색어 단어: Okt로 뽑은 명사 (예: "월세가 너무 비싸" → ["월세"])
    # - 키워드 점수: 검색어 단어가 법률안 이름에 있으면 2, 요약본에 있으면 1을 더한 뒤 최대값(3 × 단어 수)으로 나눔
    # - 의미 점수: Word2Vec으로 만든 검색어 벡터와 법률안 벡터의 코사인 유사도 (음수는 0)
    # 키워드가 하나도 없는 법안은 의미 점수가 MIN_SIMILARITY 이상일 때만 결과에 넣는다.
    KEYWORD_WEIGHT = 0.5
    MIN_SIMILARITY = 0.3

    # 검색어에서 명사만 뽑기 (전처리와 같은 규칙: 두 글자 이상, 불용어 제외)
    # 남는 명사가 없으면(예: "비싸", "법률") 띄어쓰기 단위 단어를 그대로 쓴다.
    def _query_words(self, query: str) -> list[str]:
        nouns = self._okt.nouns(query) if self._okt is not None else []
        words = [w for w in nouns if len(w) > 1 and w not in self._stopwords]
        return list(dict.fromkeys(words or query.split()))  # 순서 유지하며 중복 제거

    def search(self, query: str, limit: int = 20) -> list[dict]:
        df, wv, doc_vectors = self._df, self._wv, self._doc_vectors
        if df is None or not query.strip():
            return []

        terms = self._query_words(query)

        # 키워드 점수 (0~1)
        keyword = pandas.Series(0.0, index=df.index)
        for term in terms:
            keyword += df['BILL_NAME'].str.contains(term, regex=False) * 2
            keyword += df['SUMMARY'].str.contains(term, regex=False) * 1
        keyword /= 3 * len(terms)

        # 의미 점수 (0~1)
        semantic = pandas.Series(0.0, index=df.index)
        query_vector = wv.query_vector(terms) if wv is not None else None
        if query_vector is not None:
            semantic = pandas.Series(numpy.clip(doc_vectors @ query_vector, 0, 1), index=df.index)

        score = self.KEYWORD_WEIGHT * keyword + (1 - self.KEYWORD_WEIGHT) * semantic
        matched = (keyword > 0) | (semantic >= self.MIN_SIMILARITY)

        result = df.assign(SCORE=score.round(4))[matched]
        result = result.sort_values('SCORE', ascending=False, kind='stable').head(limit)  # 같은 점수면 최신순 유지
        result = result[['BILL_NO', 'BILL_NAME', 'PROC_RESULT', 'SUMMARY', 'SCORE']]

        # pandas는 빈 값을 NaN으로 표시하지만 JSON에는 NaN이 없으므로 None(null)으로 바꾼다.
        # (str 열은 None을 담을 수 없어서 object 타입으로 바꾼 뒤 교체)
        result = result.astype(object).where(result.notna(), None)
        return result.to_dict(orient='records')


# 전체에서 사용하는 단일 인스턴스
data = Data()
