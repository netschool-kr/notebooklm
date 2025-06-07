import os
import fitz  # PyMuPDF
from dotenv import load_dotenv
from google.cloud import bigquery
from google.cloud.bigquery import Dataset, Table, SchemaField
import vertexai
from vertexai.preview.language_models import TextEmbeddingModel, TextEmbeddingInput

def main():
    """
    데이터 디렉토리의 파일들을 임베딩하여 BigQuery에 저장하는 메인 함수
    """
    # --- 1. 환경 설정 ---
    load_dotenv()
    project_id = os.environ["GOOGLE_CLOUD_PROJECT"]
    project_location = os.environ["GOOGLE_CLOUD_LOCATION"]
    print(f"Project ID: {project_id}, Location: {project_location}")

    # Vertex AI SDK 초기화
    vertexai.init(project=project_id, location=project_location)
    
    # BigQuery 클라이언트 초기화
    bq_client = bigquery.Client(project=project_id)
    
    # Vertex AI 임베딩 모델 로드
    embedding_model = TextEmbeddingModel.from_pretrained("text-multilingual-embedding-002")

    # --- 2. 텍스트 추출 및 임베딩 생성 ---
    data_dir = "data"
    processed_data = []  # id, content, embedding을 담을 리스트

    try:
        filenames = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    except FileNotFoundError:
        print(f"[오류] '{data_dir}' 디렉토리를 찾을 수 없습니다. 스크립트와 같은 위치에 만들어주세요.")
        return

    print(f"'{data_dir}' 디렉토리에서 파일 처리 시작...")
    for fname in filenames:
        file_path = os.path.join(data_dir, fname)
        text = ""

        # 파일 확장자에 따라 텍스트 추출
        if fname.lower().endswith(".pdf"):
            try:
                with fitz.open(file_path) as doc:
                    text = "".join(page.get_text() for page in doc)
            except Exception as e:
                print(f"[오류] PDF 처리 실패: {fname} - {e}")
                continue
        elif fname.lower().endswith(".txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception as e:
                print(f"[오류] TXT 읽기 실패: {fname} - {e}")
                continue
        else:
            print(f"[알림] 지원하지 않는 파일 형식, 건너뜀: {fname}")
            continue

        if not text.strip():
            print(f"[알림] 내용이 비어 있음, 건너뜀: {fname}")
            continue

        # 텍스트 임베딩 생성 (Vertex AI SDK 사용)
        try:
            # RETRIEVAL_DOCUMENT는 저장/인덱싱될 문서를 임베딩할 때 사용
            embedding_input = TextEmbeddingInput(task_type="RETRIEVAL_DOCUMENT", text=text)
            embedding_response = embedding_model.get_embeddings([embedding_input])
            vector = embedding_response[0].values
            
            processed_data.append({
                "id": fname,
                "content": text,
                "embedding": vector
            })
            print(f"[성공] 임베딩 생성 완료: {fname}")
        except Exception as e:
            print(f"[오류] 임베딩 생성 실패: {fname} - {e}")
            continue

    print(f"\n총 {len(processed_data)}개의 파일 처리가 완료되었습니다.")

    if not processed_data:
        print("처리할 데이터가 없어 프로그램을 종료합니다.")
        return

    # --- 3. BigQuery에 데이터 저장 ---
    dataset_id = "book_data"
    table_id = "embeddings"
    full_dataset_id = f"{project_id}.{dataset_id}"
    full_table_id = f"{full_dataset_id}.{table_id}"

    print(f"\nBigQuery에 데이터 저장 시작: {full_table_id}")

    dataset = Dataset(full_dataset_id)
    dataset.location = "US"
    bq_client.create_dataset(dataset, exists_ok=True)
    print(f"데이터셋 '{dataset_id}' 준비 완료.")

    schema = [
        SchemaField("id", "STRING", mode="REQUIRED"),
        SchemaField("content", "STRING", mode="REQUIRED"),
        SchemaField("embedding", "FLOAT", mode="REPEATED"), # FLOAT64 대신 FLOAT 사용
    ]
    table = Table(full_table_id, schema=schema)

    # 기존 테이블 삭제 후 새로 생성
    bq_client.delete_table(table, not_found_ok=True)
    print(f"기존 테이블 '{table_id}' 삭제 (스키마 업데이트를 위해).")
    bq_client.create_table(table)
    print(f"새로운 스키마로 테이블 '{table_id}' 생성 완료.")

    try:
        errors = bq_client.insert_rows_json(table, processed_data)
        if not errors:
            print(f"성공적으로 {len(processed_data)}개의 행을 BigQuery에 저장했습니다.")
        else:
            print("BigQuery 삽입 중 오류 발생:", errors)
    except Exception as e:
        print(f"BigQuery 데이터 삽입 중 예외 발생: {e}")

if __name__ == "__main__":
    main()
