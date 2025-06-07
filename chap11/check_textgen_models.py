import google.generativeai as genai
import os
from dotenv import load_dotenv
import pandas as pd

def check_available_text_models():
    """
    genai 라이브러리를 사용하여 사용 가능한 텍스트 생성 모델 목록을 확인합니다.
    """
    # --- 1. 환경 설정 ---
    load_dotenv()
    PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
    LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")

    # Vertex AI 백엔드를 사용하도록 설정 (가장 중요한 부분)
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

    if not PROJECT_ID or not LOCATION:
        print("오류: .env 파일에 GOOGLE_CLOUD_PROJECT와 GOOGLE_CLOUD_LOCATION이 설정되어 있는지 확인하세요.")
        return

    print(f"프로젝트 '{PROJECT_ID}' 및 위치 '{LOCATION}'에서 사용 가능한 모델을 확인합니다...")
    print("-" * 60)

    try:
        # --- 2. 'generateContent' 지원 모델 필터링 ---
        print("\n✅ 사용 가능한 Text Generation 모델 목록입니다:\n")
        
        # genai.list_models()를 사용하여 전체 모델 목록을 가져옵니다.
        all_models = genai.list_models()

        text_model_data = []
        for model in all_models:
            # 텍스트 생성을 지원하는 'generateContent'가 포함된 모델만 필터링합니다.
            if 'generateContent' in model.supported_generation_methods:
                
                # qa_agent.py의 from_pretrained()에 필요한 ID는 'models/' 다음의 이름입니다.
                model_id_for_sdk = model.name.split('/')[-1]

                text_model_data.append({
                    "SDK 사용 ID": model_id_for_sdk,
                    "표시 이름": model.display_name,
                    "전체 이름 (genai용)": model.name,
                    "설명": model.description,
                })
        
        if not text_model_data:
            print("사용 가능한 텍스트 생성 모델을 찾을 수 없습니다.")
            print("프로젝트 ID, 위치, 권한을 다시 한번 확인해 주세요.")
            return

        # Pandas DataFrame으로 변환하여 표 형태로 깔끔하게 출력
        df = pd.DataFrame(text_model_data)
        df = df.sort_values(by="SDK 사용 ID").reset_index(drop=True)
        print(df)

        print("\n" + "="*70)
        print("📢 [사용 안내]")
        print("   위 표의 'SDK 사용 ID' 컬럼에 있는 이름을 복사하여")
        print("   qa_agent.py 코드의 from_pretrained() 함수 안에 붙여넣으세요.")
        print("\n   예시: text_model = TextGenerationModel.from_pretrained(\"text-bison-002\")")
        print("="*70)


    except Exception as e:
        print(f"\n모델 목록을 가져오는 중 오류 발생: {e}")
        print("Vertex AI 초기화 또는 인증 문제가 있는지 확인하세요.")
        print("GOOGLE_APPLICATION_CREDENTIALS 환경 변수가 올바르게 설정되었는지,")
        print("또는 'gcloud auth application-default login' 명령으로 로그인했는지 확인하세요.")

if __name__ == "__main__":
    check_available_text_models()