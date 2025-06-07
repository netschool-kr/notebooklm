from dotenv import load_dotenv
import os
import vertexai
from vertexai.preview.language_models import TextEmbeddingModel, TextEmbeddingInput, TextGenerationModel
from google.cloud import bigquery

# 환경변수 로드
load_dotenv()
project = os.getenv("GOOGLE_CLOUD_PROJECT")
location = os.getenv("GOOGLE_CLOUD_LOCATION")
vertexai.init(project=project, location=location)

# BigQuery 클라이언트 및 테이블 정보 설정
bq_client = bigquery.Client(project=project)
#DATASET_ID = f"{os.environ['GOOGLE_CLOUD_PROJECT']}.book_data"
DATASET_ID = "book_data"
TABLE_ID   = "embeddings"
TABLE_NAME = f"`{project}.{DATASET_ID}.{TABLE_ID}`"



# --- BigQuery 테이블 스키마 확인을 위한 테스트 코드 ---
def get_table_schema(client, project_id, dataset_id, table_id):
    """
    지정된 BigQuery 테이블의 스키마 정보를 가져옵니다.

    Args:
        client (google.cloud.bigquery.Client): BigQuery 클라이언트 인스턴스.
        project_id (str): Google Cloud 프로젝트 ID.
        dataset_id (str): BigQuery 데이터세트 ID.
        table_id (str): BigQuery 테이블 ID.

    Returns:
        list: 테이블의 스키마 필드 목록 (google.cloud.bigquery.SchemaField 객체).
    """
    full_table_id = f"{project_id}.{dataset_id}.{table_id}"
    try:
        table = client.get_table(full_table_id) # API 요청
        return table.schema
    except Exception as e:
        print(f"Error retrieving table schema for {full_table_id}: {e}")
        return None

if __name__ == "__main__":
    # 테이블 스키마를 가져와서 출력합니다.
    print("\n--- BigQuery Table Schema ---")
    schema = get_table_schema(bq_client, project, DATASET_ID, TABLE_ID)
    if schema:
        for field in schema:
            print(f"  Column Name: {field.name}, Type: {field.field_type}, Mode: {field.mode}")
    else:
        print(f"Failed to retrieve schema for table: {TABLE_NAME}")

