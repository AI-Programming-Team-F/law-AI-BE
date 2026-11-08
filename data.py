import threading

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
                instance._status = 'idle'      # idle | running | done | failed
                instance._error = None
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
        self._status = 'running'
        self._error = None
        try:
            log.step('형태소 분석기(Okt) 로딩')
            okt = Okt()

            # 의안 번호 가져오기
            log.step(f'의안 목록 수집 시작 (목표 {size}건)')
            bills = prep.get_bills_many(22, size)
            if bills is None:
                raise RuntimeError('의안 목록 수집 실패')
            log.step(f'의안 목록 수집 완료 ({len(bills)}건)')

            # pandas DataFrame으로 변환 (필요한 열만 선택)
            df = pandas.DataFrame(bills)[['BILL_NAME', 'BILL_NO']]

            # 의안 번호마다 요약본 추출
            df['SUMMARY'] = df['BILL_NO'].apply(log.with_progress(prep.get_summary, len(df), '요약본 수집'))

            # 요약본의 명사들을 추출하여 열 추가 (불용어 필터링)
            df['NOUNS'] = df['SUMMARY'].fillna('').apply(log.with_progress(okt.nouns, len(df), '명사 추출'))
            stopwords = list(df['NOUNS'].explode().value_counts().head(20).keys())
            for x in ['제안', '이유', '현행법', '최근']: stopwords.append(x)
            df['NOUNS'] = df['NOUNS'].apply(lambda nouns: list(filter(lambda x: x not in stopwords and len(x) > 1, nouns)))

            # 없는 행 삭제
            before = len(df)
            df = df[df['SUMMARY'].notna() & (df['NOUNS'].str.len() > 0)].reset_index(drop=True)
            log.step(f'요약본/명사 없는 행 삭제: {before}건 → {len(df)}건')

            # 단어 사전 생성
            df, word2id, id2word = w2v.build_vocab(df, min_count=3)
            log.step(f'단어 사전 생성 완료 ({len(word2id)}개)')

            # 완성된 데이터로 한 번에 교체
            self._df, self._word2id, self._id2word = df, word2id, id2word
            self._status = 'done'

            log.step(f'DataFrame 메모리: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB')
            log.step(f'최대 메모리 사용량(프로세스 전체, JVM 포함): {log.peak_memory_mb():.1f} MB')
        except Exception as e:
            self._status = 'failed'
            self._error = str(e)
            log.step(f'전처리 실패: {e}')

    # ---------- 검색 ----------

    # TODO: Word2Vec 학습 후 유사 키워드 확장 / 의미 기반 검색으로 교체
    # 지금은 임시로, 검색어가 법률안 이름이나 요약본에 들어 있는 법안을 찾는다.
    # 법률안 이름에 들어 있으면 더 위에 오도록 점수를 매긴다.
    def search(self, query: str, limit: int = 20) -> list[dict]:
        df = self._df
        if df is None or not query.strip():
            return []

        terms = query.split()
        score = pandas.Series(0, index=df.index)
        for term in terms:
            score += df['BILL_NAME'].str.contains(term, regex=False) * 2
            score += df['SUMMARY'].str.contains(term, regex=False) * 1

        result = df.assign(SCORE=score)
        result = result[result['SCORE'] > 0].sort_values('SCORE', ascending=False).head(limit)
        return result[['BILL_NO', 'BILL_NAME', 'SUMMARY', 'SCORE']].to_dict(orient='records')


# 전체에서 사용하는 단일 인스턴스
data = Data()
