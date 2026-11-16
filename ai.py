import hashlib
import os
import threading

import requests

# Mindlogic API Gateway (OpenAI 호환 Chat Completions)
# https://docs.mindlogic.ai/docs/general/api-gateway/reference/chat-completions
URL = 'https://factchat-cloud.mindlogic.ai/v1/gateway/chat/completions/'

# temperature=0을 지원하는 모델이어야 함 (gpt-5 계열 일부는 1만 허용)
MODEL = os.getenv('NOVA_AI_MODEL', 'claude-sonnet-5')

SYSTEM_PROMPT = (
    '너는 대한민국 국회 법률안을 일반 시민에게 설명하는 도우미다. '
    '주어진 법률안의 제안이유 및 주요내용을 읽고, 핵심만 1~2문장 및 100자 미만의 쉬운 한국어로 요약하라. '
    '원문에 없는 내용은 추가하지 말고, 요약문만 문어체("~이다.")으로 출력하라.'
)

# 같은 내용에는 같은 답을 돌려주기 위한 캐시 (원문 해시 → 요약)
# temperature=0이어도 LLM 응답이 완전히 같다는 보장은 없으므로 한 번 만든 요약을 재사용한다.
_cache: dict[str, str] = {}
_cache_lock = threading.Lock()

# 법률안 요약 생성
# 실패하면 RuntimeError
def summarize(text: str) -> str:
    key = hashlib.sha256(text.encode('utf-8')).hexdigest()
    with _cache_lock:
        if key in _cache:
            return _cache[key]

    response = requests.post(
        URL,
        headers={
            'Authorization': f'Bearer {os.getenv("NOVA_AI_KEY")}',
            'Content-Type': 'application/json',
        },
        json={
            'model': MODEL,
            'temperature': 0,
            'messages': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': text},
            ],
        },
        timeout=60,
    )
    if response.status_code != 200:
        raise RuntimeError(f'AI 요청 실패 (상태 코드: {response.status_code}) {response.text[:200]}')

    summary = response.json()['choices'][0]['message']['content'].strip()
    with _cache_lock:
        _cache[key] = summary
    return summary
