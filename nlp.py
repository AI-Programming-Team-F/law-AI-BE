import os
import threading

import pandas

# 명사 추출과 불용어 규칙
# 학습(train.py)과 서버(data.py)가 반드시 같은 규칙을 써야 단어 사전이 맞으므로 한곳에 모아 둔다.

# 항상 빼는 불용어 (요약본 머리말 등)
BASE_STOPWORDS = ['제안', '이유', '현행법', '최근', '법']

# 학습 데이터에서 가장 많이 나온 명사 몇 개를 불용어로 추가할지
AUTO_STOPWORDS = 20

_okt = None
_okt_lock = threading.Lock()

# 형태소 분석기(Okt) — JVM을 띄우므로 처음 필요할 때 한 번만 만든다.
# JVM 최대 힙은 환경 변수 OKT_MAX_HEAP(MB, 기본 256)으로 제한한다. (기본값 1024MB는 서버 메모리에 비해 큼)
def _get_okt():
    global _okt
    with _okt_lock:
        if _okt is None:
            from konlpy.tag import Okt
            _okt = Okt(max_heap_size=int(os.getenv('OKT_MAX_HEAP', '256')))
        return _okt

# 문장에서 명사 뽑기 (불용어 처리 전)
def nouns(text: str | None) -> list[str]:
    return _get_okt().nouns(text or '')

# 검색어에서 핵심어 뽑기 (띄어쓰기 단위로 판단)
# - 명사만 이어진 단어는 붙인 채로 쓴다 (예: "자영업자" → "자영업자", Okt는 "자영"+"업자"로 쪼갬)
# - 뒤에 붙은 조사·어미는 뗀다 (예: "월세가" → "월세")
# - 단어 전체가 동사로 분류돼도 명사처럼 쓰이는 말은 그대로 쓴다 (예: "돌봄", "감면")
# - 부사·형용사 등은 버린다 (예: "너무", "비싸")
# - 그 밖의 경우(예: "보호하는")는 그 안의 명사만 쓴다
_TRAILING_TAGS = {'Josa', 'Eomi', 'Punctuation', 'Suffix'}

def keywords(text: str | None) -> list[str]:
    okt = _get_okt()
    words = []
    for token in (text or '').split():
        morphs = okt.pos(token)
        while morphs and morphs[-1][1] in _TRAILING_TAGS:
            morphs = morphs[:-1]
        if not morphs:
            continue
        tags = {tag for _, tag in morphs}
        if tags == {'Noun'}:
            words.append(''.join(word for word, _ in morphs))
        elif len(morphs) == 1 and morphs[0][1] == 'Verb' and morphs[0][0] == token:
            words.append(token)
        else:
            words += [word for word, tag in morphs if tag == 'Noun']
    return words

# 학습 데이터의 명사 리스트들로 불용어 목록 만들기 (가장 많이 나온 명사 + 기본 불용어)
def make_stopwords(noun_lists: pandas.Series) -> list[str]:
    frequent = noun_lists.explode().dropna().value_counts().head(AUTO_STOPWORDS).index.tolist()
    return list(dict.fromkeys(frequent + BASE_STOPWORDS))

# 불용어와 한 글자 단어 빼기
def clean(words: list[str], stopwords: set[str] | frozenset[str]) -> list[str]:
    return [w for w in words if len(w) > 1 and w not in stopwords]
