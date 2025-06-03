import os
import fitz  # PyMuPDF: PDF 텍스트 추출용
import json # JSON 처리를 위해 추가
from google import genai
from google.genai.types import EmbedContentConfig # EmbedContentConfig 임포트 추가
from dotenv import load_dotenv
from google.cloud import storage # GCS 연동을 위해 추가

# .env 파일에서 GCP 설정 로드
load_dotenv()
project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
project_location = os.getenv("GOOGLE_CLOUD_LOCATION") # GCS 버킷 리전 등과 관련될 수 있음
print(f"Project ID: {project_id}, Project Location: {project_location}")

# --- GCS 업로드 설정 ---
# vertex_ai_matching_engine.py의 BUCKET_NAME과 일치시키거나 환경 변수에서 가져옵니다.
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_FOR_EMBEDDINGS", "notebook-docs")
# vertex_ai_matching_engine.py의 CONTENTS_DELTA_URI의 폴더 부분과 일치시킵니다. (예: "embeddings/")
# gs://BUCKET_NAME/GCS_OUTPUT_FOLDER/내에 파일이 저장됩니다.
GCS_OUTPUT_FOLDER = "embeddings/" # '/'로 끝나야 합니다.
# 로컬에 생성될 JSONL 파일 관련 설정
LOCAL_OUTPUT_DIR = "output_embeddings" # 로컬에 저장될 디렉토리
OUTPUT_JSONL_FILENAME = "embeddings_for_matching_engine.json" # 생성될 JSONL 파일 이름

# Vertex AI GenAI 사용 설정
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"

# genai 클라이언트 초기화
try:
    client = genai.Client()
except Exception as e:
    print(f"GenAI 클라이언트 초기화 실패: {e}")
    print("GOOGLE_APPLICATION_CREDENTIALS 환경 변수가 올바르게 설정되었는지, "
          "또는 gcloud auth application-default login이 실행되었는지 확인하세요.")
    exit()

# 데이터 디렉토리에서 파일 목록 읽기
data_dir = "data"
if not os.path.isdir(data_dir):
    print(f"[오류] 데이터 디렉토리 '{data_dir}'를 찾을 수 없습니다. "
          f"스크립트와 같은 위치에 '{data_dir}' 디렉토리를 생성하고 PDF 또는 TXT 파일을 넣어주세요.")
    exit()

filenames = [
    f for f in os.listdir(data_dir)
    if os.path.isfile(os.path.join(data_dir, f))
]

embedding_records = []  # {"id": 파일명, "embedding": 벡터} 딕셔너리를 저장할 리스트

# 각 파일에 대해 파일 확장자에 맞게 텍스트 추출 후 임베딩 생성
print(f"\n총 {len(filenames)}개의 파일에 대해 임베딩 생성을 시작합니다...")
for fname in filenames:
    file_path = os.path.join(data_dir, fname)
    text_content = "" # 변수명 변경: text -> text_content

    # PDF 파일 처리 (.pdf 확장자)
    if fname.lower().endswith(".pdf"):
        try:
            pdf_document = fitz.open(file_path)
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                text_content += page.get_text()
            pdf_document.close()
            print(f"[정보] PDF 파일 텍스트 추출 완료: {fname}")
        except Exception as e:
            print(f"[오류] PDF 처리 실패: {fname} - {e}")
            continue

    # 텍스트 파일 처리 (.txt 확장자)
    elif fname.lower().endswith(".txt"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text_content = f.read()
            print(f"[정보] TXT 파일 읽기 완료: {fname}")
        except Exception as e:
            print(f"[오류] TXT 읽기 실패: {fname} - {e}")
            continue

    # 지원하지 않는 파일 형식 건너뜀
    else:
        print(f"[알림] 지원하지 않는 파일 형식 ({os.path.splitext(fname)[1]}), 건너뜀: {fname}")
        continue

    # 추출된 텍스트가 비어있으면 건너뜀
    if not text_content.strip():
        print(f"[알림] 내용이 비어 있음, 건너뜀: {fname}")
        continue

    # 모델에 텍스트 전송하여 임베딩 생성
    try:
        # EmbedContentConfig를 사용하여 task_type 설정
        config = EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        # 모델 이름을 Vertex AI 특정 모델 ID로 변경 시도
        response = client.models.embed_content(
            model="text-embedding-005", # 이전: "models/text-multilingual-embedding-002"
            contents=text_content,
            config=config
        )
        # API 응답 구조 확인: response는 딕셔너리, 실제 임베딩은 'embedding' 키에 있음
        # Vertex AI 사용 시 응답 구조가 다를 수 있음. response.embeddings[0].values 또는 response['embedding']
        if 'embedding' in response:
             vec = response['embedding']
        elif hasattr(response, 'embeddings') and response.embeddings:
             vec = response.embeddings[0].values
        else:
            print(f"[오류] 임베딩 벡터를 찾을 수 없음: {fname} - 응답: {response}")
            continue

        embedding_records.append({"id": fname, "embedding": vec})
        print(f"[성공] 임베딩 생성 완료: {fname}, 차원={len(vec)}")
    except Exception as e:
        print(f"[오류] 임베딩 생성 실패: {fname} - {e}") # 전체 예외 메시지 출력
        # 실패 시 추가 정보 출력 (예: API 응답 내용)
        if hasattr(e, 'response') and hasattr(e.response, 'text') and e.response.text:
            print(f"API 응답 본문: {e.response.text}")
        # google.api_core.exceptions.GoogleAPIError의 경우 추가 정보가 있을 수 있음
        if hasattr(e, 'errors') and e.errors:
            print(f"API 오류 상세: {e.errors}")
        if hasattr(e, 'message') and e.message: # 이미 출력되지만 명시
             print(f"오류 메시지 요약: {e.message}")
        continue

# 생성된 임베딩 레코드 확인
print(f"\n총 임베딩 생성 레코드 수: {len(embedding_records)}")
if embedding_records:
    for record in embedding_records[:2]: # 처음 2개 샘플 출력
        print(f"  샘플 레코드 - ID: {record['id']}, 벡터 미리보기 (처음 3개 차원): {record['embedding'][:3]}...")

    # --- 로컬에 JSONL 파일로 저장 ---
    if not os.path.exists(LOCAL_OUTPUT_DIR):
        os.makedirs(LOCAL_OUTPUT_DIR)
        print(f"로컬 출력 디렉토리 생성: '{LOCAL_OUTPUT_DIR}'")

    local_file_path = os.path.join(LOCAL_OUTPUT_DIR, OUTPUT_JSONL_FILENAME)
    try:
        with open(local_file_path, "w", encoding="utf-8") as f:
            for record in embedding_records:
                f.write(json.dumps(record) + "\n")
        print(f"\n[성공] 임베딩 데이터를 로컬 파일에 저장 완료: {local_file_path}")

        # --- GCS에 업로드 ---
        if not project_id:
            print("[오류] GCS 업로드를 위해 GOOGLE_CLOUD_PROJECT 환경 변수가 필요합니다. 업로드를 건너<0xEB><0><0x81>니다.")
        else:
            try:
                storage_client = storage.Client(project=project_id)
                bucket = storage_client.bucket(GCS_BUCKET_NAME)
                gcs_blob_path = f"{GCS_OUTPUT_FOLDER.strip('/')}/{OUTPUT_JSONL_FILENAME}"
                blob = bucket.blob(gcs_blob_path)

                print(f"로컬 파일 '{local_file_path}'을(를) GCS 'gs://{GCS_BUCKET_NAME}/{gcs_blob_path}'에 업로드 중...")
                blob.upload_from_filename(local_file_path)
                print(f"[성공] GCS 업로드 완료: gs://{GCS_BUCKET_NAME}/{gcs_blob_path}")
                print(f"이 GCS 폴더 경로 ('gs://{GCS_BUCKET_NAME}/{GCS_OUTPUT_FOLDER.strip('/')}/')를 "
                      f"Matching Engine의 contents_delta_uri로 사용할 수 있습니다.")

            except Exception as e:
                print(f"[오류] GCS 업로드 실패: {e}")
                print("  GCS 버킷 이름, 권한, 인증 설정을 확인하세요.")
                print(f"  - GCS 버킷: {GCS_BUCKET_NAME}")
                print(f"  - 대상 경로: {gcs_blob_path}")
                print(f"  - 로컬 파일: {local_file_path}")
                print(f"  - 프로젝트 ID: {project_id}")


    except Exception as e:
        print(f"[오류] 로컬 JSONL 파일 저장 실패: {local_file_path} - {e}")
else:
    print("\n생성된 임베딩 레코드가 없어 JSONL 파일 생성 및 GCS 업로드를 건너<0xEB><0><0x81>니다.")

 