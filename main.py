from dotenv import load_dotenv

load_dotenv()

import os
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import ai
from data import data

app = FastAPI(
    title='법률안 검색 API',
    description='22대 국회 법률안을 수집해 메모리에 적재하고 검색한다. '
                '서버 시작 후 `/preprocess`를 먼저 호출해야 검색 결과가 나온다.',
    version='0.1.0',
)

# CORS: 쉼표로 구분한 출처 목록 (예: ALLOWED_ORIGINS=http://localhost:5173,https://law-ai.vercel.app)
# 설정하지 않으면 어떤 출처도 허용하지 않는다.
allowed_origins = [o.strip().rstrip('/') for o in os.getenv('ALLOWED_ORIGINS', '').split(',') if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=['GET', 'POST'],
    allow_headers=['*'],
)

# ---------- 응답 형식 ----------

class PreprocessResponse(BaseModel):
    started: bool = Field(description='이번 호출로 수집을 새로 시작했는지 (이미 진행 중이면 false)')
    status: Literal['idle', 'running', 'done', 'failed'] = Field(description='전처리 상태')
    fetched: int = Field(ge=0, le=100, description='요약본 수집 진행률 (단위: %, 0~100)')
    error: str | None = Field(description='실패했을 때의 오류 메시지')

class BillInfo(BaseModel):
    BILL_NO: str = Field(description='의안 번호')
    BILL_NAME: str = Field(description='법률안 이름')
    COMMITTEE: str | None = Field(description='소관위원회 (아직 회부되지 않았으면 null)')
    PROC_RESULT: str | None = Field(description='본회의심의결과')
    SUMMARY: str | None = Field(description='요약본 (제안이유 및 주요내용, 아직 등록되지 않았으면 null)')

class Bill(BillInfo):
    SCORE: float = Field(ge=0, le=1, description='관련도 (0~1, 높을수록 관련 있음)')

class SummaryResponse(BaseModel):
    bill_no: str = Field(description='의안 번호')
    summary: str = Field(description='AI가 쉽게 풀어 쓴 요약')

class ProfileCriterion(BaseModel):
    name: str = Field(description='기준 이름 (법률안마다 다름)', examples=['주거 형태'])
    multiple: bool = Field(description='여러 개를 고를 수 있는지 (false면 하나만 선택)')
    options: list[str] = Field(description='선택지 (2개 이상)', examples=[['월세', '전세', '자가']])

class ProfileResponse(BaseModel):
    bill_no: str = Field(description='의안 번호')
    targets: list[str] = Field(description='원문에 직접 나온 대상 (원문 표현 그대로, 없으면 빈 리스트)',
                               examples=[['무주택 세대주', '근로자']])
    criteria: list[ProfileCriterion] = Field(description='적용 여부나 영향이 달라지는 기준과 선택지 (법률안마다 다름)')

class SelectedCriterion(BaseModel):
    name: str = Field(description='기준 이름 (/profile의 criteria[].name)', examples=['주거 형태'])
    selected: list[str] = Field([], description='고른 선택지', examples=[['월세']])

class HappenRequest(BaseModel):
    bill_no: str = Field(description='의안 번호', examples=['2222564'])
    targets: list[str] = Field([], description='고른 대상 (/profile의 targets 중)', examples=[['무주택 세대주', '근로자']])
    criteria: list[SelectedCriterion] = Field([], description='기준별로 고른 선택지')

class HappenCheck(BaseModel):
    condition: str = Field(description='조건', examples=['소득 기준'])
    status: Literal['match', 'mismatch', 'unknown'] = Field(
        description='match: 맞음, mismatch: 맞지 않음, unknown: 고른 조건으로는 알 수 없어 직접 확인 필요')
    reason: str = Field(description='이유', examples=['고른 조건에 소득이 없어 직접 확인이 필요해요'])

class HappenResponse(BaseModel):
    bill_no: str = Field(description='의안 번호')
    applicability: Literal['high', 'medium', 'low'] = Field(
        description='해당될 가능성 (high: 높음, medium: 확인 필요, low: 낮음)')
    headline: str = Field(description='한 줄 설명', examples=['이 조건에 맞는 분들은 월세에 대한 세금을 더 많이 돌려받을 수 있어요.'])
    changes: list[tuple[str, str, str]] = Field(
        description='바뀌는 것. 각 원소는 [바뀌는 것, 지금, 법안이 적용되면] 형태이다. 숫자는 원문에 있을 때만 나온다.',
        examples=[[['월세 세액공제율', '10%', '15%'], ['공제받을 수 있는 소득 기준', '현재 기준', '더 넓어짐']]],
    )
    checks: list[HappenCheck] = Field(description='고른 조건과 법안의 적용 조건 대조')
    evidence: list[str] = Field(description='근거가 된 원문 인용 (원문에 실제로 있는 문장만)')
    stage: str = Field(description='국회 처리 단계 (AI가 아닌 국회 처리 정보). 처리 전이면 "심사 중"', examples=['심사 중'])

class CompareResponse(BaseModel):
    a_bill_no: str = Field(description='비교할 법률안 A의 의안 번호')
    b_bill_no: str = Field(description='비교할 법률안 B의 의안 번호')
    differences: list[tuple[str, str]] = Field(
        description='두 법률안의 차이점. 각 원소는 [A, B] 형태이며, 유사한 주제에 대한 '
                    '법률안 A의 내용과 법률안 B의 내용을 나란히 표시한다. (개괄식)',
        examples=[[['소방공무원 인건비에 대한 국가 재정부담 확대를 목표로 함',
                    '소방발전기금 신설을 통한 소방재정 확충을 목표로 함']]],
    )

class SearchResponse(BaseModel):
    preprocess: bool = Field(description='검색할 데이터가 준비되었는지 (false면 data는 항상 빈 리스트)')
    data: list[Bill] = Field(description='검색 결과 (관련도 높은 순)')

class BillNoResponse(BaseModel):
    preprocess: bool = Field(description='전처리한 데이터가 준비되었는지 (false여도 국회 API에서 바로 찾음)')
    data: BillInfo = Field(description='찾은 법률안')

# ---------- 엔드포인트 ----------

@app.get(
    '/preprocess',
    response_model=PreprocessResponse,
    summary='데이터 수집·전처리 시작',
    description='국회 API에서 법률안을 수집하고 전처리한다. 백그라운드에서 진행되므로 바로 응답한다.\n\n'
                '- 이미 진행 중이면 새로 시작하지 않는다 (`started: false`).\n'
                '- 완료된 뒤 다시 호출하면 처음부터 다시 수집한다. 수집 중에도 기존 데이터로 검색할 수 있다.\n'
                '- 20,000건 기준 약 10분 걸린다.',
)
def preprocess(size: int = Query(20000, ge=1, description='수집할 법률안 수 (최신순)')):
    started = data.start_preprocess(size)
    return {'started': started, 'status': data.status, 'fetched': data.fetched, 'error': data.error}

@app.get(
    '/search',
    response_model=SearchResponse,
    summary='법률안 검색',
    description='검색어와 관련된 법률안을 관련도 순으로 돌려준다.\n\n'
                '- 검색어가 직접 들어 있는 법률안과, 검색어와 의미가 비슷한 법률안(Word2Vec)을 함께 찾는다.\n'
                '- `/preprocess`가 끝나야 결과가 나온다.',
)
def search(
    query: str = Query(description='검색어', examples=['소방']),
    limit: int = Query(20, ge=1, le=100, description='최대 결과 수'),
):
    return {'preprocess': data.is_ready, 'data': data.search(query, limit)}

@app.get(
    '/search/billno',
    response_model=BillNoResponse,
    summary='의안 번호로 법률안 조회',
    description='의안 번호가 정확히 일치하는 법률안 하나를 돌려준다.\n\n'
                '- 전처리한 데이터에 있으면 그대로 쓰고, 없으면 국회 API에서 바로 가져온다.\n'
                '- `/preprocess` 없이도 사용할 수 있다.',
    responses={404: {'description': '해당 의안 번호의 법률안이 없음'}},
)
def search_billno(billno: str = Query(description='의안 번호', examples=['2222564'])):
    bill = data.get_bill(billno.strip())
    if bill is None:
        raise HTTPException(404, '해당 의안 번호의 법률안이 없습니다.')
    return {'preprocess': data.is_ready, 'data': bill}

@app.get(
    '/summary',
    response_model=SummaryResponse,
    summary='법률안 AI 요약',
    description='법률안의 제안이유 및 주요내용을 AI로 짧고 쉽게 요약한다.\n\n'
                '주어진 bill_no가 preprocess로 가져온 데이터 중에 있으면 이를 활용하고, 아니면 국회 API로 따로 호출 후 요약한다.\n\n'
                '같은 내용에는 같은 요약(100자 미만)을 돌려주도록 설계되어 있다. (temperature=0, 무조건 같지 않을 수도 있음)\n\n'
                '`/preprocess` 없이도 사용할 수 있다.',
    responses={404: {'description': '해당 법률안이 없거나 요약본이 아직 등록되지 않음'},
               502: {'description': 'AI 요청 실패'}},
)
def summary(bill_no: str = Query(description='의안 번호', examples=['2222564'])):
    text = data.get_summary(bill_no)
    if not text:
        raise HTTPException(404, '해당 법률안의 요약본이 없습니다.')
    try:
        return {'bill_no': bill_no, 'summary': ai.summarize(text)}
    except Exception as e:
        raise HTTPException(502, str(e))

@app.get(
    '/compare',
    response_model=CompareResponse,
    summary='두 법률안 AI 비교',
    description='두 법률안의 제안이유 및 주요내용을 AI로 비교해, 차이점을 리스트로 돌려준다.\n\n'
                '- `differences`의 각 원소는 `[A, B]` 형태로 표시한다.\n'
                '- 각 원소는 유사한 주제에 대한 두 법률안의 차이이다. '
                '`A`는 그 주제에서 법률안 A의 내용, `B`는 법률안 B의 내용이다.\n'
                '- 같은 요청에는 항상 같은 결과를 돌려준다.\n'
                '- `/preprocess` 없이도 사용할 수 있다.',
    responses={400: {'description': '같은 법률안끼리 비교함'},
               404: {'description': '해당 법률안이 없거나 요약본이 아직 등록되지 않음'},
               502: {'description': 'AI 요청 실패'}},
)
def compare(
    a_bill_no: str = Query(description='법률안 A의 의안 번호', examples=['2222564']),
    b_bill_no: str = Query(description='법률안 B의 의안 번호', examples=['2222562']),
):
    if a_bill_no == b_bill_no:
        raise HTTPException(400, '서로 다른 법률안을 입력해야 합니다.')
    texts = {}
    for bill_no in (a_bill_no, b_bill_no):
        texts[bill_no] = data.get_summary(bill_no)
        if not texts[bill_no]:
            raise HTTPException(404, f'법률안 {bill_no}의 요약본이 없습니다.')
    try:
        differences = ai.compare(texts[a_bill_no], texts[b_bill_no])
    except Exception as e:
        raise HTTPException(502, str(e))
    return {'a_bill_no': a_bill_no, 'b_bill_no': b_bill_no, 'differences': differences}

@app.get(
    '/profile',
    response_model=ProfileResponse,
    summary='프로필 선택지 추천',
    description='법률안을 AI로 분석해, 맞춤 요약에 쓸 프로필 선택지를 추천한다.\n\n'
                '- `targets`: 원문에 직접 나온 대상. 원문에 실제로 있는 표현만 돌려준다.\n'
                '- `criteria`: 이 법률안의 적용 여부나 영향이 달라지는 기준과 선택지. '
                '기준의 종류와 개수(1~4개)는 법률안마다 다르다. (예: 주거 형태 → 월세/전세/자가)\n'
                '- `multiple`이 false인 기준은 하나만, true인 기준은 여러 개를 고를 수 있다.\n'
                '- 같은 요청에는 항상 같은 결과를 돌려준다.\n'
                '- `/preprocess` 없이도 사용할 수 있다.',
    responses={404: {'description': '해당 법률안이 없거나 요약본이 아직 등록되지 않음'},
               502: {'description': 'AI 요청 실패'}},
)
def profile(bill_no: str = Query(description='의안 번호', examples=['2222564'])):
    text = data.get_summary(bill_no)
    if not text:
        raise HTTPException(404, '해당 법률안의 요약본이 없습니다.')
    try:
        result = ai.profile(text)
    except Exception as e:
        raise HTTPException(502, str(e))
    return {'bill_no': bill_no, **result}

@app.post(
    '/happen',
    response_model=HappenResponse,
    summary='맞춤 요약 (이 법안이 적용되면?)',
    description='법률안과 사용자가 고른 프로필(대상, 기준별 선택지)을 받아, 그 조건에 맞는 사람에게 생기는 변화를 AI로 설명한다.\n\n'
                '- 프로필은 `/profile` 응답에서 고른 값을 그대로 보내면 된다. 아무것도 고르지 않아도 된다.\n'
                '- `changes`의 각 원소는 `[바뀌는 것, 지금, 법안이 적용되면]` 형태이다.\n'
                '- `evidence`는 원문에 실제로 있는 문장만 돌려준다.\n'
                '- `stage`는 AI가 아닌 국회 처리 정보다.\n'
                '- 같은 법률안·같은 프로필에는 항상 같은 결과를 돌려준다.\n'
                '- `/preprocess` 없이도 사용할 수 있다.',
    responses={404: {'description': '해당 법률안이 없거나 요약본이 아직 등록되지 않음'},
               502: {'description': 'AI 요청 실패'}},
)
def happen(request: HappenRequest):
    bill = data.get_bill(request.bill_no.strip())
    if bill is None or not bill['SUMMARY']:
        raise HTTPException(404, '해당 법률안의 요약본이 없습니다.')
    profile = {
        'targets': [t.strip() for t in request.targets if t.strip()],
        'criteria': [{'name': c.name.strip(), 'selected': [o.strip() for o in c.selected if o.strip()]}
                     for c in request.criteria],
    }
    try:
        result = ai.happen(bill['SUMMARY'], profile)
    except Exception as e:
        raise HTTPException(502, str(e))
    return {'bill_no': bill['BILL_NO'], **result, 'stage': bill['PROC_RESULT'] or '심사 중'}

# 실행부 (python main.py)
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=int(os.getenv('PORT', 8000)))
