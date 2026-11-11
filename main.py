from dotenv import load_dotenv

load_dotenv()

from typing import Literal

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

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
    error: str | None = Field(description='실패했을 때의 오류 메시지')

class Bill(BaseModel):
    BILL_NO: str = Field(description='의안 번호')
    BILL_NAME: str = Field(description='법률안 이름')
    SUMMARY: str = Field(description='요약본 (제안이유 및 주요내용)')
    SCORE: int = Field(description='관련도 (높을수록 관련 있음)')

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
    return {'started': started, 'status': data.status, 'error': data.error}

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

# 실행부 (python main.py)
if __name__ == '__main__':
    import os
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=int(os.getenv('PORT', 8000)))
