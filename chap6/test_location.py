import os
from dotenv import load_dotenv
from google.cloud import storage
from google.cloud import aiplatform

def check_gcs_bucket_location(bucket_name):
    """지정된 GCS 버킷의 위치를 확인하고 출력합니다."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        print(f"GCS 버킷 '{bucket.name}'의 위치: {bucket.location}")
        return bucket.location
    except Exception as e:
        print(f"GCS 버킷 '{bucket_name}' 정보 조회 중 오류 발생: {e}")
        return None

def check_vertex_ai_location():
    """설정된 Vertex AI의 위치를 확인하고 출력합니다."""
    # .env 파일 또는 환경 변수에서 GOOGLE_CLOUD_PROJECT와 GOOGLE_CLOUD_LOCATION 로드
    load_dotenv()
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION")

    if not project_id or not location:
        print("\n경고: GOOGLE_CLOUD_PROJECT 또는 GOOGLE_CLOUD_LOCATION 환경 변수가 설정되지 않았습니다.")
        print("Vertex AI 위치를 정확히 확인하려면 해당 변수들을 설정해야 합니다.")
        print("Attempting to get location from aiplatform.initializer (if already initialized by other means)...")
        try:
            # 다른 스크립트에서 aiplatform.init()이 이미 호출되었을 경우를 대비
            # 또는 기본 설정을 통해 확인 시도
            current_project = aiplatform.initializer.global_config.project
            current_location = aiplatform.initializer.global_config.location
            print(f"aiplatform.initializer에 설정된 Vertex AI 프로젝트: {current_project}")
            print(f"aiplatform.initializer에 설정된 Vertex AI 위치: {current_location}")
            return current_location
        except Exception as e:
            print(f"aiplatform.initializer에서 Vertex AI 위치 정보를 가져오는 데 실패했습니다: {e}")
            return None

    print(f"\n환경 변수에 설정된 Vertex AI 프로젝트: {project_id}")
    print(f"환경 변수에 설정된 Vertex AI 위치: {location}")

    # aiplatform.init()을 호출하여 실제 SDK가 사용하는 위치를 확인할 수도 있습니다.
    # (주의: 이 함수는 SDK의 전역 설정을 변경할 수 있습니다.)
    # try:
    #     aiplatform.init(project=project_id, location=location)
    #     print(f"aiplatform.init()을 통해 확인된 Vertex AI 프로젝트: {aiplatform.aiplatform.PROJECT_ID_SET}")
    #     print(f"aiplatform.init()을 통해 확인된 Vertex AI 위치: {aiplatform.aiplatform.LOCATION_SET}")
    # except Exception as e:
    #     print(f"aiplatform.init()으로 Vertex AI 설정 확인 중 오류 발생: {e}")
    
    return location

if __name__ == "__main__":
    print("--- GCS 버킷 위치 확인 ---")
    gcs_bucket_to_check = "notebook-docs"
    bucket_loc = check_gcs_bucket_location(gcs_bucket_to_check)

    print("\n--- Vertex AI 위치 확인 ---")
    vertex_loc = check_vertex_ai_location()

    if bucket_loc and vertex_loc:
        # GCS 버킷 위치는 대문자로 반환될 수 있으므로 소문자로 비교
        if bucket_loc.lower() == vertex_loc.lower():
            print(f"\n[성공] GCS 버킷 위치 ('{bucket_loc.lower()}')와 Vertex AI 위치 ('{vertex_loc.lower()}')가 일치합니다.")
        else:
            print(f"\n[경고] GCS 버킷 위치 ('{bucket_loc.lower()}')와 Vertex AI 위치 ('{vertex_loc.lower()}')가 일치하지 않습니다.")
            print("Matching Engine 인덱스 생성 시 오류가 발생할 수 있으니, 위치를 일치시켜 주세요.")
            if bucket_loc.lower() == "us" and "us-" in vertex_loc.lower(): # 예: 버킷은 'US', Vertex는 'us-central1'
                 print("GCS 버킷이 다중 리전('US')이고 Vertex AI 위치가 특정 미국 내 리전인 경우,")
                 print("GCS 버킷을 Vertex AI와 동일한 단일 리전으로 변경하는 것을 권장합니다.")