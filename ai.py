import hashlib
import json
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

COMPARE_PROMPT = (
    '너는 대한민국 국회 법률안을 일반 시민에게 설명하는 도우미다. '
    '두 법률안 A와 B의 제안이유 및 주요내용을 읽고, 두 법률안이 어느 점에서 다른지 비교하라. '
    '차이점 하나당 항목 하나로 3~7개 항목을 작성하되, 각 항목은 같은 관점에서 A의 내용(a)과 B의 내용(b)을 나란히 쓴다. '
    'a와 b는 "A는", "B는" 없이 개괄식(명사형 종결, 예: "소방발전기금 신설을 통한 소방재정 확충을 목표로 함")으로, '
    '각각 쉬운 한국어로 60자 미만으로 쓰고, 원문에 없는 내용은 추가하지 마라.'
)

# 비교 결과 구조화된 출력 형식: {"differences": [{"a": "...", "b": "..."}, ...]}
# (strict 스키마에서 안전하도록 객체로 받은 뒤 [a, b] 배열로 바꿔서 돌려준다)
COMPARE_FORMAT = {
    'type': 'json_schema',
    'json_schema': {
        'name': 'bill_comparison',
        'strict': True,
        'schema': {
            'type': 'object',
            'properties': {
                'differences': {
                    'type': 'array',
                    'description': '두 법률안의 차이점 (항목당 하나)',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'a': {'type': 'string', 'description': '이 관점에서 법률안 A의 내용 (개괄식)'},
                            'b': {'type': 'string', 'description': '이 관점에서 법률안 B의 내용 (개괄식)'},
                        },
                        'required': ['a', 'b'],
                        'additionalProperties': False,
                    },
                },
            },
            'required': ['differences'],
            'additionalProperties': False,
        },
    },
}

# 같은 내용에는 같은 답을 돌려주기 위한 캐시 (요청 내용 해시 → 결과)
# temperature=0이어도 LLM 응답이 완전히 같다는 보장은 없으므로 한 번 만든 결과를 재사용한다.
_cache: dict[str, object] = {}
_cache_lock = threading.Lock()

# AI 호출 (temperature=0), 응답 텍스트 반환
# 실패하면 RuntimeError
def _chat(system: str, user: str, response_format: dict | None = None) -> str:
    body = {
        'model': MODEL,
        'temperature': 0,
        'messages': [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user},
        ],
    }
    if response_format is not None:
        body['response_format'] = response_format

    response = requests.post(
        URL,
        headers={
            'Authorization': f'Bearer {os.getenv("NOVA_AI_KEY")}',
            'Content-Type': 'application/json',
        },
        json=body,
        timeout=60,
    )
    if response.status_code != 200:
        raise RuntimeError(f'AI 요청 실패 (상태 코드: {response.status_code}) {response.text[:200]}')
    return response.json()['choices'][0]['message']['content'].strip()

# 캐시에 있으면 재사용, 없으면 make()로 만들어 저장
def _cached(key_text: str, make):
    key = hashlib.sha256(key_text.encode('utf-8')).hexdigest()
    with _cache_lock:
        if key in _cache:
            return _cache[key]
    result = make()
    with _cache_lock:
        _cache[key] = result
    return result

# 법률안 요약 생성
# 실패하면 RuntimeError
def summarize(text: str) -> str:
    return _cached('summary\0' + text, lambda: _chat(SYSTEM_PROMPT, text))

# 두 법률안 비교 → 차이점 리스트, 각 원소는 [A 기준 차이, B 기준 차이] (개괄식)
# 실패하면 RuntimeError
def compare(a_text: str, b_text: str) -> list[list[str]]:
    def make() -> list[list[str]]:
        content = _chat(COMPARE_PROMPT, f'[법률안 A]\n{a_text}\n\n[법률안 B]\n{b_text}', COMPARE_FORMAT)
        try:
            return [[str(x['a']).strip(), str(x['b']).strip()] for x in json.loads(content)['differences']]
        except (json.JSONDecodeError, KeyError, TypeError):
            raise RuntimeError(f'AI 응답 형식 오류: {content[:200]}')

    return _cached('compare\0' + a_text + '\0' + b_text, make)
