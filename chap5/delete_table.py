import os
from dotenv import load_dotenv
from google.cloud import bigquery
from google.api_core import exceptions

def delete_bigquery_table():
    """
    BigQuery에서 지정된 테이블을 삭제합니다.
    실수로 인한 삭제를 방지하기 위해 사용자에게 확인을 요청합니다.
    """
    # --- 1. 환경 설정 ---
    load_dotenv()
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        print("[오류] GOOGLE_CLOUD_PROJECT 환경변수가 설정되지 않았습니다.")
        print(".env 파일을 확인해주세요.")
        return

    # --- 2. 삭제할 테이블 정보 ---
    dataset_id = "book_data"
    table_id = "embeddings"
    full_table_id = f"{project_id}.{dataset_id}.{table_id}"

    # --- 3. 사용자 확인 절차 (중요) ---
    print("=" * 50)
    print("⚠️ 경고: BigQuery 테이블을 영구적으로 삭제합니다.")
    print(f"  - 프로젝트: {project_id}")
    print(f"  - 데이터셋: {dataset_id}")
    print(f"  - 테이블:   {table_id}")
    print("=" * 50)
    
    # 사용자에게 직접 '삭제'를 입력하도록 요청하여 실수를 방지
    confirm = input("삭제를 원하시면 '삭제'라고 정확히 입력해주세요: ")

    if confirm.strip() != "삭제":
        print("\n입력이 일치하지 않아 작업을 취소했습니다.")
        return

    # --- 4. 테이블 삭제 실행 ---
    try:
        bq_client = bigquery.Client(project=project_id)
        print(f"\n'{full_table_id}' 테이블 삭제를 시도합니다...")
        bq_client.delete_table(full_table_id)
        print(f"✅ 성공: 테이블이 성공적으로 삭제되었습니다.")

    except exceptions.NotFound:
        print(f"ℹ️ 정보: 테이블 '{full_table_id}'이(가) 이미 존재하지 않습니다. 별도의 작업이 필요 없습니다.")
        
    except Exception as e:
        print(f"[오류] 테이블 삭제 중 예기치 않은 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    delete_bigquery_table()
