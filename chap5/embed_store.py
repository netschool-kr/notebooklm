import os
from google import genai
from dotenv import load_dotenv
import fitz  # PyMuPDF: PDF 텍스트 추출용

# .env 파일에서 GCP 설정 로드
load_dotenv()
project_id = os.environ["GOOGLE_CLOUD_PROJECT"]
project_location = os.environ["GOOGLE_CLOUD_LOCATION"]
print("project_id=", project_id, " : project_location=", project_location)

# Vertex AI GenAI 사용 설정
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"

# genai 클라이언트 초기화
client = genai.Client()

# 데이터 디렉토리에서 파일 목록 읽기
data_dir = "data"
filenames = [
    f for f in os.listdir(data_dir)
    if os.path.isfile(os.path.join(data_dir, f))
]
embeddings = []  # (파일명, 벡터) 튜플을 저장할 리스트

# 각 파일에 대해 파일 확장자에 맞게 텍스트 추출 후 임베딩 생성
for fname in filenames:
    file_path = os.path.join(data_dir, fname)
    text = ""

    # PDF 파일 처리 (.pdf 확장자)
    if fname.lower().endswith(".pdf"):
        try:
            pdf = fitz.open(file_path)
            for page in pdf:
                text += page.get_text()
            pdf.close()
        except Exception as e:
            print(f"[오류] PDF 처리 실패: {fname} - {e}")
            continue

    # 텍스트 파일 처리 (.txt 확장자)
    elif fname.lower().endswith(".txt"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            print(f"[오류] TXT 읽기 실패: {fname} - {e}")
            continue

    # PDF, TXT 이외의 파일은 건너뜀
    else:
        print(f"[알림] 지원하지 않는 파일 형식, 건너뜀: {fname}")
        continue

    # 추출된 텍스트가 비어있으면 건너뜀
    if not text.strip():
        print(f"[알림] 내용이 비어 있음, 건너뜀: {fname}")
        continue

    # 모델에 텍스트 전송하여 임베딩 생성
    try:
        response = client.models.embed_content(
            model="text-multilingual-embedding-002",#"gemini-embedding-001",
            contents=text
        )
        vec = response.embeddings[0].values
        embeddings.append((fname, vec))
        print(f"[성공] 임베딩 생성 완료: {fname}, 차원={len(vec)}")
    except Exception as e:
        print(f"[오류] 임베딩 생성 실패: {fname} - {e}")
        continue

# embeddings 리스트 확인
print(f"\n총 임베딩 생성 파일 수: {len(embeddings)}")
for fname, vec in embeddings[:3]:
    print(f"샘플 벡터 - 파일: {fname}, 벡터 길이: {len(vec)}")
    
import os
from google.cloud import bigquery

# BigQuery 클라이언트 초기화
bq_client = bigquery.Client()

# 데이터셋 및 테이블 생성(이미 존재하면 생략)
dataset_id = f"{os.environ['GOOGLE_CLOUD_PROJECT']}.book_data"
table_id = f"{dataset_id}.embeddings"

# 데이터셋 생성
from google.cloud.bigquery import Dataset, Table, SchemaField

dataset = Dataset(dataset_id)
dataset.location = 'US'
bq_client.create_dataset(dataset, exists_ok=True)

# 테이블 스키마 설정: id(STRING), embedding(ARRAY<FLOAT64>)
schema = [
    SchemaField("id", "STRING", mode="REQUIRED"),
    SchemaField("embedding", "FLOAT64", mode="REPEATED"),  # 배열 필드는 FLOAT64 + REPEATED로 정의
]
table = Table(table_id, schema=schema)
bq_client.create_table(table, exists_ok=True) 

# 임베딩 결과를 테이블에 삽입 (embeddings는 이전에 생성된 리스트)
# 예시: embeddings = [("file1.pdf", [0.1, 0.2, ...]), ...]
rows_to_insert = [
    {"id": fname, "embedding": vec}
    for fname, vec in embeddings
]
errors = bq_client.insert_rows_json(table, rows_to_insert)
if errors:
    print("BigQuery 삽입 오류:", errors)
