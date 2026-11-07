import os

import req

# 의안 번호에 따라 요약 내용 가져오기
def get_summary(BILL_NO : str) -> str:
    status_code, response = req.request(req.BILLSUMMARY, {
        'KEY': os.getenv('KEY'),
        'Type': 'json',
        'pIndex': 1,
        'pSize': 20,
        'AGE': 22,
        'BILL_NO': BILL_NO
    })

    if status_code == 200:
        try:
            summary = response['BPMBILLSUMMARY'][1]['row'][0]['SUMMARY']
            return summary
        except ValueError as e:
            print("JSON 파싱 실패! 실제 반환된 응답 텍스트:")
            print(response)
    else:
        print(f"요청 실패 (상태 코드: {status_code})")

    return None

# 의안 번호 불러오기
def get_bills(age: int, page_size: int, page_index: int) -> dict | None:
    status_code, response = req.request(req.LEGISLATIVE_BILL, {
            'KEY': os.getenv('KEY'),
            'Type': 'json',
            'pIndex': page_index,
            'pSize': page_size,
            'AGE': age
        })
    
    result = {}
    if status_code == 200:
        print("요청 성공(HTTP 200)")
        try:
            result = response['nzmimeepazxkubdpn'][1]['row']
        except ValueError as e:
            print("JSON 파싱 실패! 실제 반환된 응답 텍스트:")
            print(response)
    else:
        print(f"요청 실패 (상태 코드: {status_code})")
        return None

    return result