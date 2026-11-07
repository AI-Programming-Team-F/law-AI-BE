from dotenv import load_dotenv
import pandas

import prep

load_dotenv()

def main():
    # 데이터
    result = {
        '법률안명': [],
        '의안 번호': [],
        '요약': []
    }

    # 의안 번호 가져오기
    bills = prep.get_bills(22, 20, 1)
    if bills is None:
        print('뭔가 잘못됨')
        return

    for bill in bills:
        # 요약본 추출
        summary = prep.get_summary(bill['BILL_NO'])

        # result에 넣기
        result['법률안명'].append(bill['BILL_NAME'])
        result['의안 번호'].append(bill['BILL_NO'])
        result['요약'].append(summary)

    # pandas DataFrame으로 변환
    df = pandas.DataFrame(result)
    print(df)

# 실행부
if __name__ == '__main__':
    main()