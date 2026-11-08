import resource
import sys
import time
from typing import Any, Callable

_start = time.time()

# 경과 시간과 함께 단계 로그 출력
def step(message: str) -> None:
    print(f'[{time.time() - _start:7.1f}s] {message}', flush=True)

# 진행 상황을 한 줄로 출력 (\r로 같은 줄을 덮어씀, 마지막에만 줄바꿈)
def progress(current: int, total: int, label: str) -> None:
    percent = current / total if total else 1
    end = '\n' if current >= total else ''
    print(f'\r[{time.time() - _start:7.1f}s] {label}: {current}/{total} ({percent:.0%})', end=end, flush=True)

# 함수를 감싸서 호출될 때마다 진행 상황을 출력 (Series.apply에 넘겨서 사용)
def with_progress(func: Callable[[Any], Any], total: int, label: str) -> Callable[[Any], Any]:
    count = 0

    def wrapped(x: Any) -> Any:
        nonlocal count
        result = func(x)
        count += 1
        progress(count, total, label)
        return result

    return wrapped

# 프로세스 최대 메모리 사용량(MB)
# Okt가 사용하는 JVM도 같은 프로세스 안에서 돌아가므로 함께 포함됨
# ru_maxrss 단위: Linux는 KB, macOS는 byte (Windows는 resource 모듈 미지원)
def peak_memory_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == 'darwin' else peak / 1024
