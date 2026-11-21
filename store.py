import json
import os
import re
import tempfile

# 법률안의 "변하지 않는 정보"를 의안 번호별 JSON 파일로 보관하는 저장소
# - 파일 위치, 파일 이름 규칙, JSON 형식은 이 모듈 안에서만 다룬다.
#   밖에서는 dict로만 주고받으므로, 저장 방식이 바뀌어도 다른 코드는 바뀌지 않는다.
# - 처리결과(PROC_RESULT), 소관위원회(COMMITTEE)처럼 바뀔 수 있는 정보는 저장하지 않는다.
#   (그런 값은 매번 국회 API에서 새로 받는다)
# - 요약본이 아직 등록되지 않은 법안은 나중에 등록될 수 있으므로 저장하지 않는다.
class BillStore:
    # 저장하는 필드 (한 번 정해지면 바뀌지 않는 값만)
    FIELDS = ('BILL_NO', 'BILL_ID', 'BILL_NAME', 'PROPOSE_DT', 'SUMMARY')

    def __init__(self, directory: str):
        self._directory = directory

    # 의안 번호 → 파일 경로 (숫자가 아니면 다른 경로에 접근할 수 있으므로 거부)
    def _path(self, bill_no: str) -> str:
        if not re.fullmatch(r'\d+', str(bill_no)):
            raise ValueError(f'잘못된 의안 번호: {bill_no!r}')
        return os.path.join(self._directory, f'{bill_no}.json')

    # 저장된 법률안이 있는지
    def has(self, bill_no: str) -> bool:
        return os.path.isfile(self._path(bill_no))

    # 저장된 법률안 불러오기 (없거나 파일이 깨졌으면 None)
    def load(self, bill_no: str) -> dict | None:
        try:
            with open(self._path(bill_no), encoding='utf-8') as f:
                record = json.load(f)
        except (OSError, ValueError):
            return None
        if not isinstance(record, dict) or not record.get('SUMMARY'):
            return None
        return {field: record.get(field) for field in self.FIELDS}

    # 법률안 저장. 새로 저장했으면 True
    # - 이미 파일이 있으면 절대 덮어쓰지 않는다 (파일이 깨져 있어도 그대로 둔다)
    # - 요약본이 없으면 저장하지 않는다
    def save(self, bill: dict) -> bool:
        path = self._path(bill['BILL_NO'])
        if not bill.get('SUMMARY') or os.path.exists(path):
            return False

        record = {field: bill.get(field) for field in self.FIELDS}
        os.makedirs(self._directory, exist_ok=True)

        # 임시 파일에 다 쓴 뒤 최종 이름으로 연결(link)한다.
        # - 쓰다가 멈춰도 반쯤 쓴 파일이 최종 이름으로 남지 않는다
        # - os.link는 대상 파일이 이미 있으면 실패하므로, 확인과 저장 사이에 파일이 생겨도 덮어쓰지 않는다
        #   (os.replace는 덮어쓰기 때문에 쓰지 않음)
        fd, temp_path = tempfile.mkstemp(dir=self._directory, suffix='.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            os.link(temp_path, path)
        except FileExistsError:
            return False
        finally:
            os.remove(temp_path)
        return True


# 전체에서 사용하는 저장소 (위치는 환경 변수 DATA_DIR로 바꿀 수 있음, 기본: 프로젝트의 data 폴더)
store = BillStore(os.getenv('DATA_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')))
