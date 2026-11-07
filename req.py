from typing import Any

import requests

# 법률안 제안이유 및 주요내용
# https://open.assembly.go.kr/portal/data/service/selectAPIServicePage.do/OS46YD0012559515463
BILLSUMMARY = '/BPMBILLSUMMARY'

# 국회의원 발의법률안
# https://open.assembly.go.kr/portal/data/service/selectAPIServicePage.do/OK7XM1000938DS17215
LEGISLATIVE_BILL= '/nzmimeepazxkubdpn'

# JSON으로 응답 온 데이터 반환
# index=0: status_code, index=1: JSON으로 변환된 데이터
def request(endpoint : str, params : dict, base_URL : str ='https://open.assembly.go.kr/portal/openapi') -> tuple[int, Any]:
    response = requests.get(base_URL + ('' if endpoint.startswith('/') else '/') + endpoint, params=params)
    return (response.status_code, response.json()) # json으로 반환