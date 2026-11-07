from dotenv import load_dotenv
import pandas
from konlpy.tag import Okt

import prep

load_dotenv()

def main():
    okt = Okt()

    # 의안 번호 가져오기
    bills = prep.get_bills(22, 20, 1)
    if bills is None:
        print('뭔가 잘못됨')
        return

    # pandas DataFrame으로 변환 (필요한 열만 선택)
    df = pandas.DataFrame(bills)[['BILL_NAME', 'BILL_NO']]

    # 의안 번호마다 요약본 추출
    df['SUMMARY'] = df['BILL_NO'].apply(prep.get_summary)

    # 요약본의 명사들을 추출하여 열 추가 (불용어 필터링)
    stopwords = ['제안', '참고', '이유', '내용', '현행법', '개정', '법률']
    df['NOUNS'] = df['SUMMARY'].fillna('').apply(okt.nouns).apply(lambda nouns: list(filter(lambda x: x not in stopwords and len(x) > 1, nouns)))
    df['NOUNS'].explode().value_counts() # 빈도수 순으로 정렬

    # 없는 행 삭제
    df = df[df['SUMMARY'].notna() & (df['NOUNS'].str.len() > 0)]

    # 출력
    print(df['SUMMARY'])
    print(df['NOUNS'])

# 실행부
if __name__ == '__main__':
    main()