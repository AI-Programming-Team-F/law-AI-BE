# 법률안 검색 서비스 — 백엔드 / AI

열린국회정보 Open API로 22대 국회 법률안을 수집하고, pandas로 정제한 뒤 TensorFlow Word2Vec으로 유사 키워드 제안과 의미 검색을 제공한다.

## 기능

- **데이터 수집** — 법률안 목록과 요약본을 API에서 받아 pandas DataFrame으로 메모리에 적재
- **전처리** — 요약본에서 명사 추출(Okt), 불용어 제거
- **단어 사전** — 명사를 정수 ID로 변환 (Word2Vec 입력용)
- **로그** — 단계별 진행 상황과 최대 메모리 사용량 출력

## 파일

| 파일      | 역할                 |
| --------- | -------------------- |
| `main.py` | 전체 실행            |
| `req.py`  | API 요청             |
| `prep.py` | 데이터 수집          |
| `w2v.py`  | Word2Vec (단어 사전) |
| `log.py`  | 로그, 메모리 측정    |

## 실행

Java(JDK)와 `.env`의 `KEY`(열린국회정보 API 키)가 필요하다.

```bash
pip install requests python-dotenv pandas konlpy
python main.py
```

## 앞으로 할 일

- TensorFlow Word2Vec 학습 → 유사 키워드 제안
- 의미 기반 법률안 검색
- FastAPI로 프론트엔드에 API 제공
