import os
import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv

def find_available_models_final_version():
    """
    gcloud 인증(ADC)을 사용하여 Vertex AI 백엔드에서 사용 가능한
    텍스트 생성 모델 목록을 확인합니다. (API 키 및 genai.init 불필요)
    """
    # 1. 환경 변수 로드 및 Vertex AI 백엔드 사용 설정
    try:
        load_dotenv()
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION")

        if not project_id or not location:
            print("[오류] .env 파일에 GOOGLE_CLOUD_PROJECT와 GOOGLE_CLOUD_LOCATION을 설정해주세요.")
            return

        # genai 라이브러리가 API 키가 아닌 Vertex AI 백엔드를 사용하도록 설정
        # 이 설정이 gcloud 인증 및 프로젝트/위치 설정을 자동으로 사용하게 합니다.
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
        
        print(f"프로젝트: '{project_id}', 위치: '{location}'")
        print("사용 가능한 텍스트 생성 모델을 조회합니다...")
        print("-" * 50)

    except Exception as e:
        print(f"환경 설정 중 오류 발생: {e}")
        return

    # 2. 모델 목록 조회 및 필터링
    try:
        # genai.list_models()를 호출하면 gcloud 설정을 기반으로 Vertex AI에서 목록을 가져옵니다.
        all_models = genai.list_models()

        available_text_models = []
        for model in all_models:
            if 'generateContent' in model.supported_generation_methods:
                sdk_model_id = model.name.split('/')[-1]
                available_text_models.append({
                    "모델 ID (SDK용)": sdk_model_id,
                    "모델 전체 이름": model.name,
                    "설명": model.description,
                })

        if not available_text_models:
            print("사용 가능한 텍스트 생성 모델이 없습니다.")
            return

        # 3. 결과를 Pandas DataFrame으로 변환하여 출력
        df = pd.DataFrame(available_text_models)
        df_sorted = df.sort_values(by="모델 ID (SDK용)").reset_index(drop=True)

        print("\n✅ 현재 사용 가능한 텍스트 모델 목록입니다.\n")
        print(df_sorted.to_string())

        print("\n" + "=" * 60)
        print("💡 위 '모델 ID (SDK용)' 컬럼의 이름을 에이전트 코드에 사용하세요.")
        print("=" * 60)

    except Exception as e:
        print(f"\n[오류] 모델 목록을 가져오는 데 실패했습니다: {e}")
        print("GCloud 인증 상태('gcloud auth application-default login')와")
        print("프로젝트의 'Vertex AI API' 활성화 여부를 다시 확인해 주세요.")

if __name__ == "__main__":
    find_available_models_final_version()