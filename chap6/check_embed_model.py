import google.generativeai as genai
import os
from google.cloud import aiplatform # aiplatform.init()을 위해 필요할 수 있음
from dotenv import load_dotenv

# --- 환경 설정 (기존 스크립트와 유사하게 설정) ---
load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")

# Vertex AI 사용 설정 (필수)
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

if not PROJECT_ID or not LOCATION:
    print("GOOGLE_CLOUD_PROJECT와 GOOGLE_CLOUD_LOCATION 환경 변수를 설정해야 합니다.")
else:
    try:
        # Vertex AI 초기화 (이미 스크립트에 있다면 생략 가능)
        # aiplatform.init(project=PROJECT_ID, location=LOCATION)
        # print(f"Vertex AI 초기화 완료 (프로젝트: {PROJECT_ID}, 위치: {LOCATION})")

        print("\n사용 가능한 모든 모델 목록:")
        for m in genai.list_models():
            print(f"  모델명: {m.name}")
            print(f"    표시 이름: {m.display_name}")
            print(f"    지원하는 생성 방식: {m.supported_generation_methods}")
            # 임베딩 모델은 보통 'embedContent'를 지원합니다.

        print("\n--- 그 중 'embedContent'를 지원하는 모델 (임베딩 모델): ---")
        for m in genai.list_models():
            if 'embedContent' in m.supported_generation_methods:
                print(f"  모델명 (Model Name for genai.embed_content): {m.name}")
                print(f"    표시 이름 (Display Name): {m.display_name}")
                print(f"    설명 (Description): {m.description}")
                print(f"    입력 토큰 제한 (Input Token Limit): {m.input_token_limit}")
                print(f"    출력 토큰 제한 (Output Token Limit): {m.output_token_limit}")
                print("-" * 20)

    except Exception as e:
        print(f"모델 목록을 가져오는 중 오류 발생: {e}")
        print("Vertex AI 초기화 또는 인증 문제가 있는지 확인하세요.")
        print("GOOGLE_APPLICATION_CREDENTIALS 환경 변수가 올바르게 설정되었는지,")
        print("또는 'gcloud auth application-default login' 명령으로 로그인했는지 확인하세요.")

