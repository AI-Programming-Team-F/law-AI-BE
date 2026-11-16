from dotenv import load_dotenv

load_dotenv()

from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

import ai
from data import data

app = FastAPI(
    title='법률안 검색 API',
    description='22대 국회 법률안을 수집해 메모리에 적재하고 검색한다. '
                '서버 시작 후 `/preprocess`를 먼저 호출해야 검색 결과가 나온다.',
    version='0.1.0',
)

# ---------- 응답 형식 ----------

class PreprocessResponse(BaseModel):
    started: bool = Field(description='이번 호출로 수집을 새로 시작했는지 (이미 진행 중이면 false)')
    status: Literal['idle', 'running', 'done', 'failed'] = Field(description='전처리 상태')
    fetched: int = Field(ge=0, le=100, description='요약본 수집 진행률 (단위: %, 0~100)')
    error: str | None = Field(description='실패했을 때의 오류 메시지')

class Bill(BaseModel):
    BILL_NO: str = Field(description='의안 번호')
    BILL_NAME: str = Field(description='법률안 이름')
    SUMMARY: str = Field(description='요약본 (제안이유 및 주요내용)')
    SCORE: int = Field(description='관련도 (높을수록 관련 있음)')

class SummaryResponse(BaseModel):
    bill_no: str = Field(description='의안 번호')
    summary: str = Field(description='AI가 쉽게 풀어 쓴 요약')

class SearchResponse(BaseModel):
    preprocess: bool = Field(description='검색할 데이터가 준비되었는지 (false면 data는 항상 빈 리스트)')
    data: list[Bill] = Field(description='검색 결과 (관련도 높은 순)')

# ---------- 엔드포인트 ----------

@app.get(
    '/preprocess',
    response_model=PreprocessResponse,
    summary='데이터 수집·전처리 시작',
    description='국회 API에서 법률안을 수집하고 전처리한다. 백그라운드에서 진행되므로 바로 응답한다.\n\n'
                '- 이미 진행 중이면 새로 시작하지 않는다 (`started: false`).\n'
                '- 완료된 뒤 다시 호출하면 처음부터 다시 수집한다. 수집 중에도 기존 데이터로 검색할 수 있다.\n'
                '- 2,000건 기준 약 10분 걸린다.',
)
def preprocess(size: int = Query(2000, ge=1, description='수집할 법률안 수 (최신순)')):
    started = data.start_preprocess(size)
    return {'started': started, 'status': data.status, 'fetched': data.fetched, 'error': data.error}

@app.get(
    '/search',
    response_model=SearchResponse,
    summary='법률안 검색',
    description='검색어와 관련된 법률안을 관련도 순으로 돌려준다.\n\n'
                '※ 임시 구현. AI(Word2Vec) 기반 검색으로 교체 예정.',
)
def search(
    query: str = Query(description='검색어', examples=['소방']),
    limit: int = Query(20, ge=1, le=100, description='최대 결과 수'),
):
    return {'preprocess': data.is_ready, 'data': data.search(query, limit)}

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

# 실행부 (python main.py)
if __name__ == '__main__':
    import os
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=int(os.getenv('PORT', 8000)))
