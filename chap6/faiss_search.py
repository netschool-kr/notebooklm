from dotenv import load_dotenv
import os
import numpy as np
import google.generativeai as genai
from google.cloud import storage # For reading files from GCS
import json # For parsing JSON embedding files
import faiss # For FAISS similarity search
# import time # No longer strictly needed for ME deployment waits

# --- 기본 환경 설정 (Basic Environment Setup) ---
# .env 파일에서 환경 변수 로드 (Load environment variables from .env file)
load_dotenv()

# Google Cloud 프로젝트 ID 및 위치 설정 (Set Google Cloud Project ID and Location)
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION") # 예: "us-central1" 또는 "asia-northeast3"

# Vertex AI 사용 설정 (Generative AI SDK for embeddings)
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

# --- GCS 설정 (GCS Configuration for Embeddings) ---
BUCKET_NAME = os.getenv("GCS_BUCKET_FOR_EMBEDDINGS", "notebook-docs")
print(f"사용할 GCS 버킷: gs://{BUCKET_NAME}/ (Vertex AI 위치: {LOCATION}와 동일 리전이어야 함)")
print(f"Using GCS bucket: gs://{BUCKET_NAME}/ (Must be in the same region as Vertex AI Location: {LOCATION})")

# 임베딩 파일이 저장된 GCS 폴더 (GCS folder where embedding files are stored)
GCS_EMBEDDINGS_FOLDER = os.getenv("GCS_EMBEDDINGS_FOLDER_NAME", "embeddings/")
CONTENTS_DELTA_URI_PREFIX = f"{GCS_EMBEDDINGS_FOLDER.strip('/')}/" # Used as a prefix for listing blobs

# 임베딩 차원 (Embedding dimension)
# "textembedding-gecko-multilingual", "text-multilingual-embedding-002", "models/text-embedding-004" (Gemini) 등 모델 기준
EMBEDDING_DIMENSION = 768 # 사용하는 임베딩 모델에 맞게 조정하세요.

# --- Vertex AI 및 GenAI 클라이언트 초기화 (Initialize Vertex AI and GenAI Clients) ---
if not PROJECT_ID or not LOCATION:
    raise ValueError("GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION 환경 변수를 설정해야 합니다.\n(GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION environment variables must be set.)")

print(f"Vertex AI 초기화 중... 프로젝트: {PROJECT_ID}, 위치: {LOCATION}")
# aiplatform.init(project=PROJECT_ID, location=LOCATION) # aiplatform.init is not strictly needed if only using genai for embeddings
# However, if other aiplatform features were to be used, it would be initialized here.

print("Google GenAI 클라이언트 초기화 중 (Vertex AI 백엔드 사용)...")
# genai.configure(project=PROJECT_ID, location=LOCATION) # Implicitly configured by GOOGLE_GENAI_USE_VERTEXAI

def load_embeddings_and_ids_from_gcs(bucket_name, gcs_folder_prefix):
    """
    GCS에서 임베딩 파일들을 로드하여 임베딩 리스트와 ID 리스트를 반환합니다.
    각 파일은 JSON Lines 형식이어야 하며, 각 라인은 {"id": "...", "embedding": [...]} 형태입니다.

    Args:
        bucket_name (str): GCS 버킷 이름.
        gcs_folder_prefix (str): 버킷 내 임베딩 파일들이 있는 폴더 경로 (예: "embeddings/").

    Returns:
        tuple: (all_embeddings_np, all_doc_ids)
               all_embeddings_np (numpy.ndarray): float32 타입의 임베딩 numpy 배열.
               all_doc_ids (list): 문자열 ID 리스트.
    """
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=gcs_folder_prefix)

    all_embeddings_list = []
    all_doc_ids = []
    file_count = 0

    print(f"\nGCS에서 임베딩 로드 중: gs://{bucket_name}/{gcs_folder_prefix}")
    for blob in blobs:
        if not blob.name.endswith(".json"): # JSON 파일만 처리 (필요시 수정)
            if blob.name != gcs_folder_prefix: # 폴더 자체는 건너뛰기
                 print(f"  JSON 파일이 아닌 파일 건너뛰기: {blob.name}")
            continue
        
        file_count += 1
        print(f"  처리 중인 파일: {blob.name}")
        try:
            content = blob.download_as_text()
            lines = content.strip().split('\n')
            for line_num, line in enumerate(lines):
                if not line.strip(): # 빈 줄 건너뛰기
                    continue
                try:
                    data = json.loads(line)
                    if "id" in data and "embedding" in data:
                        all_doc_ids.append(str(data["id"])) # ID는 문자열로 저장
                        all_embeddings_list.append(data["embedding"])
                    else:
                        print(f"    경고: {blob.name} 파일의 {line_num+1}번째 줄에 'id' 또는 'embedding' 필드가 없습니다.")
                except json.JSONDecodeError as e:
                    print(f"    오류: {blob.name} 파일의 {line_num+1}번째 줄 JSON 파싱 실패: {e}")
        except Exception as e:
            print(f"  오류: {blob.name} 파일 처리 중 문제 발생: {e}")

    if not all_embeddings_list:
        print("GCS에서 임베딩을 로드하지 못했습니다. 지정된 경로에 올바른 형식의 파일이 있는지 확인하세요.")
        return None, None

    print(f"{file_count}개 파일에서 총 {len(all_embeddings_list)}개의 임베딩 로드 완료.")
    
    # 모든 임베딩이 동일한 차원을 가지는지 확인 (첫 번째 임베딩 기준)
    first_embedding_dim = len(all_embeddings_list[0])
    if first_embedding_dim != EMBEDDING_DIMENSION:
        print(f"경고: 로드된 첫 번째 임베딩 차원({first_embedding_dim})이 설정된 EMBEDDING_DIMENSION({EMBEDDING_DIMENSION})과 다릅니다. EMBEDDING_DIMENSION을 확인하세요.")
        # 필요시 여기서 EMBEDDING_DIMENSION을 업데이트하거나 오류를 발생시킬 수 있습니다.

    for i, emb in enumerate(all_embeddings_list):
        if len(emb) != EMBEDDING_DIMENSION:
            print(f"오류: ID '{all_doc_ids[i]}'의 임베딩 차원({len(emb)})이 일관되지 않습니다 (기대값: {EMBEDDING_DIMENSION}). 해당 임베딩을 건너<0xEB><0><0x81>니다.")
            # 문제가 있는 임베딩과 ID를 제거하는 로직 추가 가능
            # 여기서는 간단히 오류 메시지만 출력하고 진행 (FAISS add에서 문제 발생 가능)


    all_embeddings_np = np.array(all_embeddings_list).astype('float32')
    
    # FAISS에 추가하기 전에 차원 일관성 최종 확인
    if all_embeddings_np.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError(f"임베딩 배열의 차원({all_embeddings_np.shape[1]})이 EMBEDDING_DIMENSION({EMBEDDING_DIMENSION})과 일치하지 않습니다.")

    return all_embeddings_np, all_doc_ids

def build_faiss_index(embeddings_np, embedding_dimension):
    """
    주어진 임베딩으로 FAISS 인덱스를 빌드합니다.

    Args:
        embeddings_np (numpy.ndarray): float32 타입의 임베딩 numpy 배열.
        embedding_dimension (int): 임베딩 차원.

    Returns:
        faiss.Index: 빌드된 FAISS 인덱스.
    """
    print(f"\nFAISS 인덱스 빌드 중... (임베딩 개수: {embeddings_np.shape[0]}, 차원: {embedding_dimension})")
    # DOT_PRODUCT_DISTANCE와 유사하게 작동하도록 IndexFlatIP 사용
    # (정규화된 벡터의 경우 코사인 유사도와 동일)
    index = faiss.IndexFlatIP(embedding_dimension)
    index.add(embeddings_np)
    print(f"FAISS 인덱스 빌드 완료. 인덱스에 총 {index.ntotal}개의 벡터가 있습니다.")
    return index

# --- FAISS 인덱스 로드 및 빌드 ---
embeddings_np, doc_ids = load_embeddings_and_ids_from_gcs(BUCKET_NAME, CONTENTS_DELTA_URI_PREFIX)

if embeddings_np is not None and doc_ids:
    faiss_index = build_faiss_index(embeddings_np, EMBEDDING_DIMENSION)

    # --- 검색할 쿼리 텍스트 및 임베딩 생성 (Query Text and Embedding Generation) ---
    query_text = "RAG란 무엇인가요?"
    # query_text = "Vertex AI Matching Engine의 주요 기능은 무엇인가?"
    # query_text = "LLM 모델을 평가하는 일반적인 기준은 무엇인가?"
    
    # Gemini 1.5 Flash 모델 사용 예시 (text-embedding-004)
    # embed_model = "models/text-embedding-004" # Gemini 모델
    # 이전 textembedding-gecko 모델 사용 예시
    embed_model = "models/text-embedding-004"
    print(f"\n쿼리 텍스트: '{query_text}'")
    print(f"이 쿼리에 대한 임베딩 생성 중 (모델: {embed_model})...")
    
    try:
        response_embedding = genai.embed_content(
            model=embed_model, # Vertex AI 백엔드 사용시 모델 ID
            content=query_text,
            task_type="RETRIEVAL_QUERY", # 검색용 쿼리
            # title="Optional title for query" # 필요시 제목 추가
        )
        query_vector = response_embedding['embedding']
        print(f"쿼리 임베딩 생성 성공. 차원: {len(query_vector)}")

        # --- FAISS 인덱스에서 최근접 이웃 검색 ---
        print(f"\nFAISS 인덱스를 사용하여 유사 항목 검색 중...")
        num_neighbors_to_find = 5
        query_vector_np = np.array([query_vector]).astype('float32')

        # FAISS 검색: D는 거리(유사도 점수), I는 인덱스
        # IndexFlatIP의 경우 D는 내적값 (클수록 유사함)
        distances, indices = faiss_index.search(query_vector_np, num_neighbors_to_find)

        if indices.size > 0 and indices[0][0] != -1 : # indices[0][0] == -1 이면 검색 결과 없음
            print(f"\n'{query_text}'에 대한 검색 결과 ({len(indices[0])}개):")
            for i in range(len(indices[0])):
                neighbor_idx = indices[0][i]
                neighbor_id = doc_ids[neighbor_idx]
                neighbor_similarity = distances[0][i] # IndexFlatIP는 내적값
                print(f"  {i+1}. ID: {neighbor_id}, 유사도 (Dot Product): {neighbor_similarity:.6f}")
        else:
            print("유사한 항목을 찾지 못했습니다.")

    except Exception as e:
        print(f"임베딩 생성 또는 FAISS 검색 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\n임베딩 로드에 실패하여 FAISS 검색을 진행할 수 없습니다.")

# --- FAISS 인덱스 저장/로드 (선택 사항) ---
# 생성된 FAISS 인덱스를 저장하고 싶다면:
# if 'faiss_index' in locals() and faiss_index:
#     index_filename = "my_faiss_index.index"
#     faiss.write_index(faiss_index, index_filename)
#     print(f"\nFAISS 인덱스가 '{index_filename}' 파일로 저장되었습니다.")
#     # doc_ids도 함께 저장해야 합니다 (예: pickle 또는 json)
#     with open("my_doc_ids.json", "w") as f:
#         json.dump(doc_ids, f)
#     print(f"문서 ID가 'my_doc_ids.json' 파일로 저장되었습니다.")

# 저장된 FAISS 인덱스를 로드하고 싶다면:
# loaded_index = faiss.read_index("my_faiss_index.index")
# with open("my_doc_ids.json", "r") as f:
#     loaded_doc_ids = json.load(f)
# print("\n저장된 FAISS 인덱스와 문서 ID를 로드했습니다.")
