from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI

from data import data

app = FastAPI()

# 데이터 수집·전처리 시작 (백그라운드에서 진행되며, 이미 진행 중이면 새로 시작하지 않음)
@app.get('/preprocess')
def preprocess(size: int = 2000):
    started = data.start_preprocess(size)
    return {'started': started, 'status': data.status, 'error': data.error}

# 검색 (전처리가 끝나기 전에는 data가 빈 리스트)
@app.get('/search')
def search(query: str, limit: int = 20):
    return {'preprocess': data.is_ready, 'data': data.search(query, limit)}

# 실행부 (python main.py)
if __name__ == '__main__':
    import os
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=int(os.getenv('PORT', 8000)))
