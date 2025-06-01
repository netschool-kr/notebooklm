import os
import shutil
import subprocess
import sys
import time

# 로컬에 data 디렉토리 생성
os.makedirs('data', exist_ok=True)

# GitHub 저장소 URL 및 클론 디렉토리 설정
github_repo = 'https://github.com/netschool-kr/notebooklm.git'
clone_dir = 'notebooklm_tmp'

# 기존 클론 디렉토리가 있으면 삭제하여 충돌 방지
if os.path.exists(clone_dir):
    try:
        shutil.rmtree(clone_dir)
        time.sleep(0.5)
    except Exception as e:
        print(f"기존 디렉토리 삭제 실패: {e}")
        sys.exit(1)

# GitHub 저장소를 클론
print(f"Cloning repository: {github_repo} to {clone_dir}")
try:
    subprocess.run(['git', 'clone', '--depth', '1', github_repo, clone_dir], check=True)
    print("클론 성공")
except subprocess.CalledProcessError as e:
    print(f"Git 클론 실패: {e}")
    sys.exit(1)
except FileNotFoundError:
    print("Git이 설치되어 있지 않거나 PATH에 없습니다.")
    sys.exit(1)

# 클론된 저장소의 data 디렉토리 경로
data_src = os.path.join(clone_dir, 'data')

# data 폴더가 존재하면 로컬 data에 복사
if os.path.isdir(data_src):
    try:
        shutil.copytree(data_src, 'data', dirs_exist_ok=True)
        print(f"'{data_src}'의 파일을 'data' 디렉토리에 복사 완료")
    except Exception as e:
        print(f"파일 복사 중 오류 발생: {e}")
        sys.exit(1)
else:
    print(f"클론된 저장소에 'data' 폴더가 없습니다: {data_src}")
    sys.exit(1)

# 임시 클론 디렉토리 정리
try:
    shutil.rmtree(clone_dir)
    print(f"임시 디렉토리 '{clone_dir}' 삭제 완료")
except Exception as e:
    print(f"임시 디렉토리 삭제 실패: {e}")
