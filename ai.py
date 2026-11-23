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
    '차이점 하나당 항목 하나로 3~7개 항목을 작성하되, 각 항목은 무엇을 비교하는지(topic)와 '
    '그 관점에서 A의 내용(a), B의 내용(b)을 나란히 쓴다. '
    'topic은 "유효기간", "연장 근거", "피해 규모"처럼 15자 미만의 짧은 명사구로 쓴다. '
    'a와 b는 "A는", "B는" 없이 개괄식(명사형 종결, 예: "소방발전기금 신설을 통한 소방재정 확충을 목표로 함")으로, '
    '각각 쉬운 한국어로 60자 미만으로 쓰고, 원문에 없는 내용은 추가하지 마라.'
)

# 비교 결과 구조화된 출력 형식: {"differences": [{"topic": "...", "a": "...", "b": "..."}, ...]}
# (strict 스키마에서 안전하도록 객체로 받은 뒤 [topic, a, b] 배열로 바꿔서 돌려준다)
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
                            'topic': {'type': 'string', 'description': '비교 항목 (짧은 명사구)'},
                            'a': {'type': 'string', 'description': '이 관점에서 법률안 A의 내용 (개괄식)'},
                            'b': {'type': 'string', 'description': '이 관점에서 법률안 B의 내용 (개괄식)'},
                        },
                        'required': ['topic', 'a', 'b'],
                        'additionalProperties': False,
                    },
                },
            },
            'required': ['differences'],
            'additionalProperties': False,
        },
    },
}

HAPPEN_PROMPT = (
    '너는 대한민국 국회 법률안을 일반 시민에게 설명하는 도우미다. '
    '법률안의 제안이유 및 주요내용과 사용자가 고른 조건(프로필)을 읽고, '
    '이 법률안이 적용되면 그 조건에 맞는 사람에게 생기는 변화를 쉬운 해요체로 설명하라.\n'
    '- applicability: 고른 조건으로 볼 때 이 법률안에 해당될 가능성. '
    'high(조건이 법안의 대상과 맞음), medium(일부 조건을 알 수 없어 확인 필요), low(조건이 법안의 대상과 맞지 않음). '
    '정보가 부족하면 단정하지 말고 medium으로 하라.\n'
    '- headline: "이 조건에 맞는 분들은 ~"으로 시작하는 한 문장 (60자 미만).\n'
    '- changes: 바뀌는 것(item), 지금(before), 법안이 적용되면(after)을 1~4개. '
    '숫자는 원문에 있을 때만 쓰고, 없으면 "현재 기준" → "더 넓어짐"처럼 말로 쓴다.\n'
    '- checks: 고른 조건과 원문의 적용 조건을 하나씩 대조한다. '
    'status는 match(맞음), mismatch(맞지 않음), unknown(고른 조건으로는 알 수 없어 직접 확인 필요). '
    '사용자가 고르지 않았지만 원문에 있는 적용 조건(예: 소득 기준)도 unknown으로 넣는다. reason은 40자 미만.\n'
    '- evidence: 판단의 근거가 된 원문 문장을 원문 그대로 1~3개 짧게 인용한다. 고치거나 요약하지 마라.\n'
    '원문에 없는 내용은 추가하지 마라.'
)

# 맞춤 요약 구조화된 출력 형식
HAPPEN_FORMAT = {
    'type': 'json_schema',
    'json_schema': {
        'name': 'bill_happen',
        'strict': True,
        'schema': {
            'type': 'object',
            'properties': {
                'applicability': {'type': 'string', 'enum': ['high', 'medium', 'low']},
                'headline': {'type': 'string'},
                'changes': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'item': {'type': 'string', 'description': '바뀌는 것'},
                            'before': {'type': 'string', 'description': '지금'},
                            'after': {'type': 'string', 'description': '법안이 적용되면'},
                        },
                        'required': ['item', 'before', 'after'],
                        'additionalProperties': False,
                    },
                },
                'checks': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'condition': {'type': 'string', 'description': '조건'},
                            'status': {'type': 'string', 'enum': ['match', 'mismatch', 'unknown']},
                            'reason': {'type': 'string', 'description': '이유'},
                        },
                        'required': ['condition', 'status', 'reason'],
                        'additionalProperties': False,
                    },
                },
                'evidence': {'type': 'array', 'items': {'type': 'string'}, 'description': '근거가 된 원문 인용'},
            },
            'required': ['applicability', 'headline', 'changes', 'checks', 'evidence'],
            'additionalProperties': False,
        },
    },
}

PROFILE_PROMPT = (
    '너는 대한민국 국회 법률안을 일반 시민에게 설명하는 도우미다. '
    '법률안의 제안이유 및 주요내용을 읽고, 사용자가 "이 법안이 나에게 적용되는지" 판단할 수 있도록 프로필 선택지를 만들어라.\n'
    '- targets: 원문에 직접 나온 대상(사람이나 집단)을 원문 표현 그대로 1~5개 뽑는다. '
    '조사는 빼고 "무주택 세대주", "근로자"처럼 짧은 명사구로 쓴다. 원문에 없는 표현은 만들지 마라.\n'
    '- criteria: 이 법률안의 적용 여부나 영향이 달라지는 기준을 1~4개 만든다. '
    '기준은 원문의 조건에서 찾는다 (예: 주거 형태, 소득 수준, 나이대, 사업 규모, 자녀 유무 등 법안마다 다름). '
    '각 기준(name)마다 사용자가 고를 선택지(options)를 2~6개, 짧은 명사구로 만든다. '
    '하나만 고를 수 있는 기준이면 multiple=false, 여러 개를 고를 수 있으면 multiple=true로 한다. '
    'targets와 같은 내용은 기준으로 다시 만들지 마라.'
)

# 프로필 추천 구조화된 출력 형식:
# {"targets": [...], "criteria": [{"name": "...", "multiple": false, "options": ["...", ...]}, ...]}
PROFILE_FORMAT = {
    'type': 'json_schema',
    'json_schema': {
        'name': 'bill_profile',
        'strict': True,
        'schema': {
            'type': 'object',
            'properties': {
                'targets': {'type': 'array', 'items': {'type': 'string'},
                            'description': '원문에 직접 나온 대상 (원문 표현 그대로)'},
                'criteria': {
                    'type': 'array',
                    'description': '적용 여부나 영향이 달라지는 기준',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'name': {'type': 'string', 'description': '기준 이름 (예: 주거 형태)'},
                            'multiple': {'type': 'boolean', 'description': '여러 개를 고를 수 있는지'},
                            'options': {'type': 'array', 'items': {'type': 'string'},
                                        'description': '선택지 (예: 월세, 전세, 자가)'},
                        },
                        'required': ['name', 'multiple', 'options'],
                        'additionalProperties': False,
                    },
                },
            },
            'required': ['targets', 'criteria'],
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

# 두 법률안 비교 → 차이점 리스트, 각 원소는 [비교 항목, A 기준 차이, B 기준 차이] (개괄식)
# 실패하면 RuntimeError
def compare(a_text: str, b_text: str) -> list[list[str]]:
    def make() -> list[list[str]]:
        content = _chat(COMPARE_PROMPT, f'[법률안 A]\n{a_text}\n\n[법률안 B]\n{b_text}', COMPARE_FORMAT)
        try:
            return [[str(x['topic']).strip(), str(x['a']).strip(), str(x['b']).strip()]
                    for x in json.loads(content)['differences']]
        except (json.JSONDecodeError, KeyError, TypeError):
            raise RuntimeError(f'AI 응답 형식 오류: {content[:200]}')

    return _cached('compare2\0' + a_text + '\0' + b_text, make)

# 프로필에 맞춘 요약
# profile: {"targets": [...], "criteria": [{"name": "...", "selected": [...]}, ...]}
# 반환: {"applicability", "headline", "changes": [[바뀌는 것, 지금, 적용되면]], "checks": [...], "evidence": [...]}
# - evidence는 원문에 실제로 들어 있는 인용만 남긴다
# 실패하면 RuntimeError
def happen(text: str, profile: dict) -> dict:
    lines = []
    if profile['targets']:
        lines.append(f'- 대상: {", ".join(profile["targets"])}')
    for c in profile['criteria']:
        if c['selected']:
            lines.append(f'- {c["name"]}: {", ".join(c["selected"])}')
    profile_text = '\n'.join(lines) or '(고른 조건 없음)'

    def make() -> dict:
        content = _chat(HAPPEN_PROMPT, f'[고른 조건]\n{profile_text}\n\n[법률안]\n{text}', HAPPEN_FORMAT)
        try:
            result = json.loads(content)
            return {
                'applicability': result['applicability'],
                'headline': result['headline'].strip(),
                'changes': [[c['item'].strip(), c['before'].strip(), c['after'].strip()] for c in result['changes']],
                'checks': [{'condition': c['condition'].strip(), 'status': c['status'], 'reason': c['reason'].strip()}
                           for c in result['checks']],
                'evidence': list(dict.fromkeys(e.strip() for e in result['evidence'] if e.strip() and e.strip() in text)),
            }
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
            raise RuntimeError(f'AI 응답 형식 오류: {content[:200]}')

    return _cached('happen\0' + profile_text + '\0' + text, make)

# 프로필 선택지 추천 → {"targets": [...], "criteria": [{"name", "multiple", "options"}, ...]}
# - targets는 원문에 실제로 들어 있는 표현만 남긴다 (화면의 "원문" 표시를 믿을 수 있게)
# - 기준과 선택지는 법률안마다 AI가 만든다 (빈 이름·중복·선택지 2개 미만인 기준은 버림)
# 실패하면 RuntimeError
def profile(text: str) -> dict:
    def make() -> dict:
        content = _chat(PROFILE_PROMPT, text, PROFILE_FORMAT)
        try:
            result = json.loads(content)
            targets = [t.strip() for t in result['targets'] if t.strip() and t.strip() in text]

            criteria, names = [], set()
            for c in result['criteria']:
                name = c['name'].strip()
                options = list(dict.fromkeys(o.strip() for o in c['options'] if o.strip()))
                if name and name not in names and len(options) >= 2:
                    names.add(name)
                    criteria.append({'name': name, 'multiple': bool(c['multiple']), 'options': options})

            return {'targets': list(dict.fromkeys(targets)), 'criteria': criteria}
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
            raise RuntimeError(f'AI 응답 형식 오류: {content[:200]}')

    return _cached('profile\0' + text, make)
