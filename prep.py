import os

import log
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
        except (KeyError, IndexError):
            print(f"\n요약본 없음 (의안 번호: {BILL_NO}) 응답: {response}")
    else:
        print(f"\n요청 실패 (의안 번호: {BILL_NO}, 상태 코드: {status_code})")

    return None

# 의안 번호 불러오기
def get_bills(age: int, page_size: int, page_index: int) -> list | None:
    status_code, response = req.request(req.LEGISLATIVE_BILL, {
            'KEY': os.getenv('KEY'),
            'Type': 'json',
            'pIndex': page_index,
            'pSize': page_size,
            'AGE': age
        })
    
    result = []
    if status_code == 200:
        try:
            result = response['nzmimeepazxkubdpn'][1]['row']
        except (KeyError, IndexError):
            print(f"\n의안 목록 없음 (페이지: {page_index}) 응답: {response}")
    else:
        print(f"\n요청 실패 (상태 코드: {status_code})")
        return None

    return result

# 의안 번호 여러 페이지에 걸쳐 불러오기 (API는 한 번에 최대 1,000건까지만 허용)
def get_bills_many(age: int, total: int, page_size: int = 1000) -> list | None:
    page_size = min(page_size, total)
    pages = (total + page_size - 1) // page_size

    result = []
    for page_index in range(1, pages + 1):
        rows = get_bills(age, page_size, page_index)
        if rows is None:
            return None
        result.extend(rows)
        log.progress(page_index, pages, '의안 목록 수집(페이지)')
        if len(rows) < page_size:  # 마지막 페이지
            if page_index < pages:
                log.progress(pages, pages, '의안 목록 수집(페이지)')
            break

    return result[:total]