from dotenv import load_dotenv
import pandas
from konlpy.tag import Okt

import log
import prep
import w2v

load_dotenv()

def main(size: int = 2000):
    log.step('형태소 분석기(Okt) 로딩')
    okt = Okt()

    # 의안 번호 가져오기
    log.step(f'의안 목록 수집 시작 (목표 {size}건)')
    bills = prep.get_bills_many(22, size)
    if bills is None:
        print('뭔가 잘못됨')
        return
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
    df = df[df['SUMMARY'].notna() & (df['NOUNS'].str.len() > 0)]
    log.step(f'요약본/명사 없는 행 삭제: {before}건 → {len(df)}건')

    # 단어 사전 생성
    df, word2id, id2word = w2v.build_vocab(df, min_count=3)
    log.step('단어 사전 생성 완료')

# 실행부
if __name__ == '__main__':
    main()