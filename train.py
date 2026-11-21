# 로컬 학습 스크립트: data/의 요약본으로 Word2Vec을 학습해 model/에 저장한다.
# 서버에서는 학습하지 않고 model/만 불러오므로, 단어 사전을 갱신하고 싶을 때 로컬에서 실행한 뒤 커밋한다.
#
#   pip install -r requirements-train.txt
#   python train.py
from dotenv import load_dotenv

load_dotenv()

import pandas

import log
import nlp
import w2v
from store import store


def main(min_count: int = 3, epochs: int = 10) -> None:
    # 저장된 요약본 불러오기
    bill_nos = store.bill_nos()
    log.step(f'저장된 법률안 {len(bill_nos)}건 불러오기')
    records = [r for r in map(store.load, bill_nos) if r is not None]
    if not records:
        raise SystemExit('data/에 저장된 법률안이 없습니다. 서버에서 /preprocess를 먼저 실행하세요.')
    df = pandas.DataFrame(records)[['BILL_NO', 'SUMMARY']]

    # 명사 추출 → 불용어 제거 (서버와 같은 규칙: nlp.py)
    df['NOUNS'] = df['SUMMARY'].apply(log.with_progress(nlp.nouns, len(df), '명사 추출'))
    stopwords = nlp.make_stopwords(df['NOUNS'])
    df['NOUNS'] = df['NOUNS'].apply(lambda words: nlp.clean(words, set(stopwords)))
    df = df[df['NOUNS'].str.len() > 0].reset_index(drop=True)
    log.step(f'불용어 {len(stopwords)}개: {stopwords}')

    # 단어 사전 → Word2Vec 학습 (TensorFlow)
    df, word2id, _ = w2v.build_vocab(df, min_count=min_count)
    df = df[df['IDS'].str.len() > 0].reset_index(drop=True)
    if not word2id:
        raise SystemExit('학습할 단어가 없습니다. 데이터가 너무 적거나 불용어에 모두 걸렸습니다.')
    log.step(f'Word2Vec 학습 시작 (법률안 {len(df)}건, 단어 {len(word2id)}개)')
    vectors = w2v.train(df['IDS'], vocab_size=len(word2id) + 1, epochs=epochs)

    # 저장
    model = w2v.Model(vectors, word2id, stopwords)
    model.save(w2v.MODEL_DIR)
    log.step(f'저장 완료: {w2v.MODEL_DIR}')

    # 학습이 잘 됐는지 눈으로 확인
    wv = model.word_vectors()
    for word in ['소방', '아동', '환경', '월세']:
        log.step(f'  유사 단어 {word}: {wv.most_similar(word)}')
    log.step(f'최대 메모리 사용량: {log.peak_memory_mb():.1f} MB')


if __name__ == '__main__':
    main()
