from dotenv import load_dotenv
import os
# import numpy as np # 이 스크립트에서는 직접 사용되지 않음
# from google.cloud import bigquery # 이 스크립트에서는 직접 사용되지 않음
# from google import genai
import google.generativeai as genai
# from google.genai.types import EmbedContentConfig # 직접 사용되지 않으나, 확장 시 일관성을 위해 참고 가능
from google.cloud import aiplatform
import time # 배포 상태 확인을 위한 time 모듈 추가

# --- 기본 환경 설정 (Basic Environment Setup) ---
# .env 파일에서 환경 변수 로드 (Load environment variables from .env file)
load_dotenv()

# Google Cloud 프로젝트 ID 및 위치 설정 (Set Google Cloud Project ID and Location)
# .env 파일 또는 환경에 GOOGLE_CLOUD_PROJECT 및 GOOGLE_CLOUD_LOCATION 환경 변수가 설정되어 있어야 합니다.
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION") # 예: "us-central1" 또는 "asia-northeast3"

# Vertex AI 사용 설정 (Generative AI SDK)
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

# --- Matching Engine 설정 (Matching Engine Configuration) ---
# GCS_BUCKET_FOR_EMBEDDINGS 환경 변수가 설정되지 않은 경우, 기본값으로 "notebook-docs-us-central1"을 사용합니다.
# 중요: 이 버킷은 Vertex AI LOCATION과 동일한 리전(예: us-central1)에 있어야 합니다.
BUCKET_NAME = os.getenv("GCS_BUCKET_FOR_EMBEDDINGS", "notebook-docs")

print(f"사용할 GCS 버킷: gs://{BUCKET_NAME}/ (Vertex AI 위치: {LOCATION}와 동일 리전이어야 함)")
print(f"Using GCS bucket: gs://{BUCKET_NAME}/ (Must be in the same region as Vertex AI Location: {LOCATION})")


# 임베딩 파일이 저장된 GCS URI (GCS URI where embedding files are stored)
# embed_store_for_matching_engine.py 스크립트의 GCS_OUTPUT_FOLDER 기본값("embeddings/")과 일치시킵니다.
GCS_EMBEDDINGS_FOLDER = "embeddings/" # 이전: os.getenv("GCS_EMBEDDINGS_FOLDER_NAME", "embeddings")
CONTENTS_DELTA_URI = f"gs://{BUCKET_NAME}/{GCS_EMBEDDINGS_FOLDER.strip('/')}/" # 폴더이므로 끝에 / 추가

# Matching Engine 인덱스 및 엔드포인트 표시 이름 (Display names for Matching Engine Index and Endpoint)
# 필요시 고유한 이름으로 수정하세요. (Modify with unique names if needed)
sanitized_project_id = PROJECT_ID.replace('-', '')[:10] if PROJECT_ID else "default"
INDEX_DISPLAY_NAME = f"my-embeddings-idx-{sanitized_project_id}"
INDEX_ENDPOINT_DISPLAY_NAME = f"my-embeddings-ep-{sanitized_project_id}"
DEPLOYED_INDEX_ID = f"deployed_idx_{sanitized_project_id}"

# 임베딩 차원 (Embedding dimension)
EMBEDDING_DIMENSION = 768 # "textembedding-gecko-multilingual" 또는 "text-multilingual-embedding-002" 모델 기준

# --- Vertex AI 및 GenAI 클라이언트 초기화 (Initialize Vertex AI and GenAI Clients) ---
if not PROJECT_ID or not LOCATION:
    raise ValueError("GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION 환경 변수를 설정해야 합니다.\n(GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION environment variables must be set.)")

print(f"Vertex AI 초기화 중... 프로젝트: {PROJECT_ID}, 위치: {LOCATION}")
aiplatform.init(project=PROJECT_ID, location=LOCATION)

print("Google GenAI 클라이언트 초기화 중 (Vertex AI 백엔드 사용)...")
# embed_client = genai.Client() # Vertex AI 설정을 따름 (사용자 환경의 라이브러리 버전에 Client 속성 없음)

# --- 인덱스 생성 또는 가져오기 (Create or Get Index) ---
print(f"\nMatching Engine 인덱스 확인/생성: {INDEX_DISPLAY_NAME}")
indexes = aiplatform.MatchingEngineIndex.list(filter=f'display_name="{INDEX_DISPLAY_NAME}"')
if indexes:
    index = indexes[0]
    print(f"기존 인덱스 사용: {index.resource_name}")
else:
    print(f"새 인덱스 생성 중... (GCS 경로: {CONTENTS_DELTA_URI})")
    try:
        index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
            display_name=INDEX_DISPLAY_NAME,
            contents_delta_uri=CONTENTS_DELTA_URI,
            dimensions=EMBEDDING_DIMENSION,
            approximate_neighbors_count=10,
            distance_measure_type="DOT_PRODUCT_DISTANCE", # 코사인 유사도와 관련 (COSINE_DISTANCE도 가능)
        )
        print(f"인덱스 생성됨: {index.resource_name}. 작업이 완료될 때까지 시간이 걸릴 수 있습니다.")
        # 인덱스 생성 완료 대기 (실제 운영 시에는 더 견고한 로직 권장)
        index.wait() # 간단한 대기 방법
        print("인덱스 생성 작업 완료됨.")
    except Exception as e:
        print(f"인덱스 생성 오류: {e}")
        raise

# --- 인덱스 엔드포인트 생성 또는 가져오기 (Create or Get Index Endpoint) ---
print(f"\nMatching Engine 인덱스 엔드포인트 확인/생성: {INDEX_ENDPOINT_DISPLAY_NAME}")
endpoints = aiplatform.MatchingEngineIndexEndpoint.list(filter=f'display_name="{INDEX_ENDPOINT_DISPLAY_NAME}"')
if endpoints:
    index_endpoint = endpoints[0]
    print(f"기존 인덱스 엔드포인트 사용: {index_endpoint.resource_name}")
else:
    print(f"새 인덱스 엔드포인트 생성 중...")
    try:
        index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
            display_name=INDEX_ENDPOINT_DISPLAY_NAME,
            public_endpoint_enabled=True
        )
        print(f"인덱스 엔드포인트 생성됨: {index_endpoint.resource_name}")
        # 엔드포인트 생성 완료 대기는 일반적으로 별도 wait()가 없으며, 배포 시 상태 확인
    except Exception as e:
        print(f"인덱스 엔드포인트 생성 오류: {e}")
        raise

# --- 엔드포인트에 인덱스 배포 (Deploy Index to Endpoint) ---
print(f"\n인덱스 배포 확인/시작: {DEPLOYED_INDEX_ID} to {index_endpoint.display_name}")
is_deployed = False
# 엔드포인트 객체를 최신 상태로 가져와서 확인
current_endpoint_state = aiplatform.MatchingEngineIndexEndpoint(index_endpoint.name)
if current_endpoint_state.deployed_indexes:
    for deployed_idx_info in current_endpoint_state.deployed_indexes:
        if deployed_idx_info.id == DEPLOYED_INDEX_ID:
            is_deployed = True
            print(f"인덱스 '{DEPLOYED_INDEX_ID}'는(은) 엔드포인트 '{index_endpoint.name}'에 이미 배포되어 있습니다.")
            break

if not is_deployed:
    print(f"인덱스 '{index.display_name}' (ID: {DEPLOYED_INDEX_ID})를 엔드포인트 '{index_endpoint.display_name}'에 배포 중...")
    try:
        index_endpoint.deploy_index(
            index=index, deployed_index_id=DEPLOYED_INDEX_ID
        )
        print(f"인덱스 배포 요청 성공. 배포 완료까지 몇 분 정도 소요될 수 있습니다. (상태 확인 중...)")

        wait_interval = 60 # 초
        max_attempts = 20  # 최대 시도 횟수 (20 * 60초 = 20분)
        for attempt in range(max_attempts):
            current_endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint.name) # 최신 정보 가져오기
            deployment_ready = False
            if current_endpoint.deployed_indexes:
                for di in current_endpoint.deployed_indexes:
                    if di.id == DEPLOYED_INDEX_ID:
                        # 실제 배포 상태 확인을 더 정확하게 하려면,
                        # di.deployment_status (사용 가능한 경우) 또는 엔드포인트의 public_domain_name 등을 확인해야 합니다.
                        # 여기서는 ID 존재 및 일정 시간 경과로 배포 간주 (프로덕션에서는 더 견고한 확인 필요)
                        print(f"시도 {attempt + 1}/{max_attempts}: 인덱스 '{DEPLOYED_INDEX_ID}'가 엔드포인트에 있는 것을 확인했습니다.")
                        deployment_ready = True # 간소화된 확인
                        break
            if deployment_ready:
                # 실제 서비스 가능 상태가 되기까지 추가 시간이 필요할 수 있음
                print(f"인덱스 '{DEPLOYED_INDEX_ID}'의 배포 프로세스가 시작되었거나 완료된 것으로 보입니다. 서비스 가능까지 몇 분 더 소요될 수 있습니다.")
                break
            print(f"시도 {attempt + 1}/{max_attempts}: 배포가 아직 완료되지 않았습니다. {wait_interval}초 후 다시 확인합니다.")
            time.sleep(wait_interval)
        else: # for-else: 루프가 break 없이 완료된 경우
            print("최대 시도 횟수를 초과했습니다. 배포 상태를 Google Cloud Console에서 수동으로 확인해주세요.")

    except Exception as e:
        print(f"인덱스 배포 오류: {e}")
else:
    print(f"인덱스 '{DEPLOYED_INDEX_ID}'가(이) 이미 배포되어 있으므로 배포를 건너<0xEB><0><0x81>니다.")


# --- 검색할 쿼리 텍스트 및 임베딩 생성 (Query Text and Embedding Generation) ---
query_text = "RAG란 무엇인가요?"
# query_text = "Vertex AI Matching Engine의 주요 기능은 무엇인가?"
embed_model = "models/text-embedding-004"
print(f"\n쿼리 텍스트: '{query_text}'")
print(f"이 쿼리에 대한 임베딩 생성 중 (모델: {embed_model})...")
try:
    # google-generativeai SDK를 통해 Vertex AI 백엔드 사용 시
    # 모델명은 "textembedding-gecko-multilingual" 또는 "models/text-multilingual-embedding-002" 등 사용 가능
    # embed_store 스크립트에서 성공한 모델명과 일치시키는 것이 좋음
    response_embedding = genai.embed_content( # embed_client.embed_content 대신 genai.embed_content 사용
        model=embed_model,
        content=query_text, # google-generativeai SDK에서는 content 사용
        task_type="RETRIEVAL_QUERY",
    )
    query_vector = response_embedding['embedding']
    print(f"쿼리 임베딩 생성 성공. 차원: {len(query_vector)}")

    # --- 배포된 인덱스에서 최근접 이웃 검색 (Nearest Neighbor Search on Deployed Index) ---
    print(f"\n배포된 인덱스 '{DEPLOYED_INDEX_ID}'를 사용하여 유사 항목 검색 중...")

    index_endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint.name) # 최신 정보 업데이트

    if index_endpoint.deployed_indexes and any(di.id == DEPLOYED_INDEX_ID for di in index_endpoint.deployed_indexes):
        num_neighbors_to_find = 5
        print(f"최근접 이웃 {num_neighbors_to_find}개 검색 요청...")

        response_neighbors = index_endpoint.find_neighbors(
            deployed_index_id=DEPLOYED_INDEX_ID,
            queries=[query_vector],
            num_neighbors=num_neighbors_to_find
        )

        if response_neighbors and response_neighbors[0]:
            print(f"\n'{query_text}'에 대한 검색 결과 ({len(response_neighbors[0])}개):")
            for i, neighbor in enumerate(response_neighbors[0]):
                print(f"  {i+1}. ID: {neighbor.id}, 거리 (Distance): {neighbor.distance:.6f}")
        else:
            print("유사한 항목을 찾지 못했거나 응답이 비어 있습니다.")
    else:
        print(f"인덱스 '{DEPLOYED_INDEX_ID}'가(이) 배포되지 않았거나 엔드포인트가 아직 준비되지 않았을 수 있습니다.")
        print("Google Cloud Console에서 Matching Engine 엔드포인트의 배포 상태를 확인하세요.")

except Exception as e:
    print(f"임베딩 생성 또는 이웃 검색 중 오류 발생: {e}")
    import traceback
    traceback.print_exc()


# --- 정리 (선택 사항) (Cleanup - Optional) ---
# print("\n--- 선택적 정리 (Optional Cleanup) ---")
# PROCEED_WITH_CLEANUP = False # True로 변경하여 정리 실행
# if PROCEED_WITH_CLEANUP:
#     try:
#         if 'index_endpoint' in locals() and index_endpoint and hasattr(index_endpoint, 'name') and index_endpoint.name:
#             print(f"엔드포인트 '{index_endpoint.name}'에서 인덱스 '{DEPLOYED_INDEX_ID}' 배포 해제 중...")
#             try:
#                 index_endpoint.undeploy_index(deployed_index_id=DEPLOYED_INDEX_ID)
#                 print("인덱스 배포 해제 완료.")
#             except Exception as e_undeploy:
#                 print(f"인덱스 배포 해제 중 오류: {e_undeploy}")

#             print("엔드포인트 삭제 전 30초 대기...")
#             time.sleep(30)

#             print(f"인덱스 엔드포인트 '{index_endpoint.name}' 삭제 중...")
#             try:
#                 index_endpoint.delete(force=True)
#                 print("인덱스 엔드포인트 삭제 완료.")
#             except Exception as e_ep_delete:
#                 print(f"인덱스 엔드포인트 삭제 중 오류: {e_ep_delete}")
#         else:
#             print("인덱스 엔드포인트 객체가 유효하지 않아 엔드포인트 관련 정리를 건너<0xEB><0><0x81>니다.")


#         if 'index' in locals() and index and hasattr(index, 'name') and index.name:
#             print("인덱스 삭제 전 30초 대기...")
#             time.sleep(30)
#             print(f"인덱스 '{index.name}' 삭제 중...")
#             try:
#                 index.delete()
#                 print("인덱스 삭제 완료.")
#             except Exception as e_idx_delete:
#                 print(f"인덱스 삭제 중 오류: {e_idx_delete}")
#         else:
#             print("인덱스 객체가 유효하지 않아 인덱스 관련 정리를 건너<0xEB><0><0x81>니다.")

#     except Exception as e_cleanup:
#         print(f"정리 중 오류 발생: {e_cleanup}")
# else:
#     print("\n정리(Cleanup)는 건너<0xEB><0><0x81>습니다. PROCEED_WITH_CLEANUP=True로 설정하여 실행할 수 있습니다.")