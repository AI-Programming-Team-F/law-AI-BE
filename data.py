import gc
import threading

import numpy
import pandas

import log
import nlp
import prep
from store import store
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
                instance._model = None         # 로컬에서 학습해 둔 모델 (w2v.Model), 없으면 키워드 검색만
                instance._stopwords = frozenset(nlp.BASE_STOPWORDS)
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
        return {} if self._model is None else self._model.word2id

    @property
    def id2word(self) -> dict[int, str]:
        return {} if self._model is None else self._model.id2word

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

    # 수집 → 학습된 모델 불러오기 → 명사 추출 → 법률안 벡터 계산
    # - Word2Vec 학습은 서버에서 하지 않는다 (로컬에서 train.py로 학습해 model/에 저장해 둔 것을 쓴다)
    # - 끝까지 성공했을 때만 내부 데이터를 한 번에 교체한다 (진행 중에도 기존 데이터로 검색 가능)
    def _preprocess(self, size: int) -> None:
        try:
            # 의안 번호 가져오기
            log.step(f'의안 목록 수집 시작 (목표 {size}건)')
            bills = prep.get_bills_many(22, size)
            if bills is None:
                raise RuntimeError('의안 목록 수집 실패')
            log.step(f'의안 목록 수집 완료 ({len(bills)}건)')

            # pandas DataFrame으로 변환
            # (바뀔 수 있는 정보인 COMMITTEE, PROC_RESULT는 저장소가 아니라 방금 받은 목록에서 가져온다)
            listed = pandas.DataFrame(bills).drop_duplicates('BILL_NO')  # 수집 중 새 법안이 끼어들면 페이지가 밀려 중복될 수 있음
            del bills
            df = listed[['BILL_NAME', 'BILL_NO', 'BILL_ID', 'COMMITTEE', 'PROC_RESULT']].copy()
            stable = listed.set_index('BILL_NO').to_dict('index')
            del listed

            # 의안 번호마다 요약본 추출: 저장소에 있으면 그대로 쓰고, 없으면 국회 API에서 받아 저장
            total = len(df)
            count = loaded = saved = 0

            def fetch_summary(bill_no: str) -> str | None:
                nonlocal count, loaded, saved
                record = store.load(bill_no)
                if record is not None:
                    summary = record['SUMMARY']
                    loaded += 1
                else:
                    summary = prep.get_summary(bill_no)
                    saved += store.save({**stable[bill_no], 'BILL_NO': bill_no, 'SUMMARY': summary})
                count += 1
                self._fetched = count * 100 // total
                return summary

            df['SUMMARY'] = df['BILL_NO'].apply(log.with_progress(fetch_summary, total, '요약본 수집'))
            del stable
            log.step(f'요약본: 저장소에서 {loaded}건, 새로 저장 {saved}건')

            # 요약본 없는 행 삭제
            before = len(df)
            df = df[df['SUMMARY'].notna()].reset_index(drop=True)
            log.step(f'요약본 없는 행 삭제: {before}건 → {len(df)}건')

            # 학습된 모델 불러오기 (없으면 의미 검색 없이 키워드 검색만)
            model = w2v.Model.load(w2v.MODEL_DIR)
            if model is None:
                log.step(f'학습된 모델이 없음 ({w2v.MODEL_DIR}) → 키워드 검색만 사용. 로컬에서 python train.py 실행 필요')
                wv, doc_vectors, stopwords = None, None, frozenset(nlp.BASE_STOPWORDS)
            else:
                stopwords = model.stopwords
                log.step(f'학습된 모델 불러오기 완료 (단어 {len(model.word2id)}개)')

                # 명사 추출 → 단어 ID (명사 리스트는 메모리를 많이 쓰므로 저장하지 않고 ID 배열만 남긴다)
                def to_ids(summary: str):
                    return model.to_ids(nlp.clean(nlp.nouns(summary), stopwords))

                ids = df['SUMMARY'].apply(log.with_progress(to_ids, len(df), '명사 추출'))
                wv = model.word_vectors()
                doc_vectors = wv.fit_documents(ids)
                del ids

            # 완성된 데이터로 한 번에 교체
            self._df, self._model, self._stopwords = df, model, stopwords
            self._wv, self._doc_vectors = wv, doc_vectors
            self._status = 'done'
            gc.collect()
            log.step(f'최대 메모리 사용량(프로세스 전체, JVM 포함): {log.peak_memory_mb():.1f} MB')
        except Exception as e:
            self._status = 'failed'
            self._error = str(e)
            log.step(f'전처리 실패: {e}')

    # ---------- 검색 ----------

    # 의안 번호로 원문 요약본(제안이유 및 주요내용) 찾기
    # 메모리 → 저장소 → 국회 API 순서로 찾는다.
    def get_summary(self, bill_no: str) -> str | None:
        df = self._df
        if df is not None:
            row = df.loc[df['BILL_NO'] == bill_no, 'SUMMARY']
            if not row.empty:
                return row.iloc[0]
        record = store.load(bill_no)
        return record['SUMMARY'] if record is not None else prep.get_summary(bill_no)

    # 의안 번호로 법률안 하나 찾기 (없으면 None)
    # 메모리에 있으면 그대로 쓰고, 없으면 국회 API에서 바로 가져온다. (요약본이 아직 없으면 SUMMARY는 None)
    # 요약본은 저장소에 있으면 저장소 것을 쓴다.
    BILL_COLUMNS = ['BILL_NO', 'BILL_ID', 'BILL_NAME', 'COMMITTEE', 'PROC_RESULT', 'SUMMARY']

    def get_bill(self, bill_no: str) -> dict | None:
        df = self._df
        if df is not None:
            row = df.loc[df['BILL_NO'] == bill_no, self.BILL_COLUMNS]
            if not row.empty:
                return _to_records(row)[0]

        bill = prep.get_bill(22, bill_no)
        if bill is None:
            return None
        record = store.load(bill_no)
        bill['SUMMARY'] = record['SUMMARY'] if record is not None else prep.get_summary(bill_no)
        return {column: bill.get(column) for column in self.BILL_COLUMNS}

    # 의안 번호로 법률안 상세 정보 (없으면 None)
    # 바뀔 수 있는 정보(처리 단계, 날짜 등)를 최신으로 보여 주기 위해 메모리가 아니라 항상 국회 API에서 새로 가져온다.
    # 요약본만 저장소에 있으면 저장소 것을 쓴다.
    INFO_COLUMNS = ['BILL_NO', 'BILL_ID', 'BILL_NAME', 'PROPOSER', 'RST_PROPOSER', 'PROPOSE_DT',
                    'COMMITTEE', 'COMMITTEE_DT', 'CMT_PRESENT_DT', 'CMT_PROC_DT', 'CMT_PROC_RESULT_CD',
                    'LAW_SUBMIT_DT', 'LAW_PRESENT_DT', 'LAW_PROC_DT', 'LAW_PROC_RESULT_CD',
                    'PROC_DT', 'PROC_RESULT', 'DETAIL_LINK', 'MEMBER_LIST']

    def get_info(self, bill_no: str) -> dict | None:
        bill = prep.get_bill(22, bill_no)
        if bill is None:
            return None
        info = {column: bill.get(column) for column in self.INFO_COLUMNS}
        info['PUBL_PROPOSER'] = [name.strip() for name in (bill.get('PUBL_PROPOSER') or '').split(',') if name.strip()]
        record = store.load(bill_no)
        info['SUMMARY'] = record['SUMMARY'] if record is not None else prep.get_summary(bill_no)
        return info

    # ---------- 검색 규칙 ----------
    # 1) 검색어에서 법률안 형식(일부개정법률안 등)을 먼저 떼어 낸다. 형식은 핵심어가 아니라 정렬 조건으로만 쓴다.
    # 2) 남은 검색어에서 핵심어를 뽑고(nlp.keywords), 일반어(GENERIC_WORDS)와
    #    전체 법안의 COMMON_RATIO 이상에 나오는 흔한 단어는 뺀다 (전부 빠지면 원래 단어 사용).
    # 3) 단어마다 IDF(드문 단어일수록 큼)로 가중치를 준다.
    #    - 핵심어 포함 비율 = 포함된 단어의 IDF 합 / 전체 IDF 합 (이름이나 요약본 어디든 있으면 포함)
    #    - 키워드 점수     = Σ IDF × (이름에 있으면 2 + 요약본에 있으면 1) / (3 × 전체 IDF 합)
    # 4) SCORE = KEYWORD_WEIGHT × 키워드 점수 + (1 - KEYWORD_WEIGHT) × 의미 점수(Word2Vec 코사인 유사도)  → 0~1
    # 5) 정렬: 핵심어 포함 비율 → (검색어에 형식이 있으면) 제목에 그 형식이 있는지 → SCORE → 최신순
    #    핵심어가 하나도 없는 법안은 의미 점수가 MIN_SIMILARITY 이상일 때만 맨 뒤에 채운다.
    KEYWORD_WEIGHT = 0.5
    MIN_SIMILARITY = 0.3
    COMMON_RATIO = 0.3

    # 법안이면 거의 다 붙는 일반어 — 비율과 관계없이 핵심어에서 뺀다 (드물게 나와도 주제와는 무관하므로)
    GENERIC_WORDS = frozenset(['법률안', '법안', '법률', '법령', '개정안', '개정', '조항', '규정', '의안'])

    # 법률안 형식: (형식 이름, 검색어에서 알아볼 표현들 — 긴 것부터)
    BILL_TYPES = [
        ('일부개정법률안', ['일부개정법률안', '일부개정']),
        ('전부개정법률안', ['전부개정법률안', '전부개정']),
        ('폐지법률안', ['폐지법률안']),
        ('제정법률안', ['제정법률안']),
    ]

    # 검색어에서 법률안 형식 떼어 내기 → (형식을 뺀 검색어, 형식 이름 또는 None)
    def _split_bill_type(self, query: str) -> tuple[str, str | None]:
        for bill_type, expressions in self.BILL_TYPES:
            for expression in expressions:
                if expression in query:
                    return query.replace(expression, ' '), bill_type
        return query, None

    # 제목이 그 형식인지 (제정법률안은 이름이 개정/폐지 없이 "~법률안", "~법안"으로 끝남)
    @staticmethod
    def _has_bill_type(names: pandas.Series, bill_type: str) -> pandas.Series:
        if bill_type == '제정법률안':
            return ~names.str.contains('개정법률안', regex=False) & ~names.str.contains('폐지법률안', regex=False)
        return names.str.contains(bill_type, regex=False)

    # 검색어에서 핵심어 뽑기 (nlp.keywords 규칙 + 두 글자 이상, 불용어 제외)
    # 남는 단어가 없으면(예: "비싸", "법률") 띄어쓰기 단위 단어를 그대로 쓴다.
    def _query_words(self, query: str) -> list[str]:
        words = nlp.clean(nlp.keywords(query), self._stopwords)
        return list(dict.fromkeys(words or query.split()))  # 순서 유지하며 중복 제거

    def search(self, query: str, limit: int = 20) -> list[dict]:
        df, wv, doc_vectors = self._df, self._wv, self._doc_vectors
        if df is None or not query.strip():
            return []

        # 1) 형식 떼어 내기 (형식만 입력했으면 원래 검색어로 찾음)
        rest, bill_type = self._split_bill_type(query)
        words = self._query_words(rest) if rest.strip() else []
        if not words:
            words, bill_type = self._query_words(query), None

        # 단어마다 이름/요약본에 들어 있는지 (영문은 대소문자 구분 없이: AI = ai)
        in_name = {w: df['BILL_NAME'].str.contains(w, case=False, regex=False) for w in words}
        in_summary = {w: df['SUMMARY'].str.contains(w, case=False, regex=False) for w in words}
        doc_freq = {w: int((in_name[w] | in_summary[w]).sum()) for w in words}

        # 2) 일반어와 흔한 단어 빼기 (전부 빠지면 원래 단어 사용)
        terms = [w for w in words
                 if w not in self.GENERIC_WORDS and doc_freq[w] < self.COMMON_RATIO * len(df)] or words

        # 3) IDF 가중치, 핵심어 포함 비율, 키워드 점수 (모두 0~1)
        idf = {t: numpy.log((len(df) + 1) / (doc_freq[t] + 1)) + 1 for t in terms}
        total_idf = sum(idf.values())
        coverage = sum(idf[t] * (in_name[t] | in_summary[t]) for t in terms) / total_idf
        keyword = sum(idf[t] * (in_name[t] * 2 + in_summary[t] * 1) for t in terms) / (3 * total_idf)

        # 의미 점수 (0~1)
        semantic = pandas.Series(0.0, index=df.index)
        # (단어 사전은 Okt 명사로 만들었으므로 "자영업자" 같은 붙인 단어 대신 명사 조각도 함께 넣는다)
        vector_words = terms + nlp.clean(nlp.nouns(rest if bill_type else query), self._stopwords)
        query_vector = wv.query_vector(vector_words) if wv is not None else None
        if query_vector is not None:
            semantic = pandas.Series(numpy.clip(doc_vectors @ query_vector, 0, 1), index=df.index)

        # 4) 최종 점수
        score = self.KEYWORD_WEIGHT * keyword + (1 - self.KEYWORD_WEIGHT) * semantic

        # 5) 정렬
        type_match = (self._has_bill_type(df['BILL_NAME'], bill_type) if bill_type
                      else pandas.Series(False, index=df.index))
        result = df.assign(SCORE=score.round(4), _COVERAGE=coverage.round(6), _TYPE=type_match)
        result = result[(coverage > 0) | (semantic >= self.MIN_SIMILARITY)]
        result = result.sort_values(['_COVERAGE', '_TYPE', 'SCORE'], ascending=False, kind='stable').head(limit)
        return _to_records(result[self.BILL_COLUMNS + ['SCORE']])


# DataFrame → dict 리스트
# pandas는 빈 값을 NaN으로 표시하지만 JSON에는 NaN이 없으므로 None(null)으로 바꾼다.
# (str 열은 None을 담을 수 없어서 object 타입으로 바꾼 뒤 교체)
def _to_records(df: pandas.DataFrame) -> list[dict]:
    return df.astype(object).where(df.notna(), None).to_dict(orient='records')


# 전체에서 사용하는 단일 인스턴스
data = Data()
