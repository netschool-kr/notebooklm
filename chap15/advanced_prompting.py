import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

# Vertex AI 초기화 (프로젝트 ID 및 위치 설정)
import os
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
vertexai.init(project=PROJECT_ID, location=LOCATION)

# 예시 원문 (실제로는 파일에서 읽거나 RAG를 통해 가져옴)
original_document_content = """
Vertex AI는 Google Cloud에서 제공하는 통합 머신러닝 플랫폼입니다. 
데이터 준비부터 모델 학습, 배포, 관리에 이르기까지 MLOps의 전체 수명 주기를 지원합니다. 
특히, 최근에는 Gemini와 같은 강력한 대규모 언어 모델(LLM)을 활용한 생성형 AI 애플리케이션 개발 기능이 강화되었습니다. 
사용자는 Vertex AI Studio를 통해 코딩 없이 LLM을 실험하거나, Python SDK를 사용하여 프로그래매틱하게 모델을 제어할 수 있습니다. 
또한, RAG(Retrieval-Augmented Generation) 아키텍처를 쉽게 구현할 수 있도록 Vector Search, RAG Engine 등의 도구를 제공하여, 
기업 내부 데이터를 LLM과 안전하게 연동하여 환각을 줄이고 신뢰성 높은 답변을 생성하도록 돕습니다.
"""

# 기본 요약 모델 및 프롬프트
baseline_model = GenerativeModel("gemini-2.0-flash-001") # 또는 "text-bison@latest"

# 고급 프롬프트 기법 적용
improved_prompt = f"""
당신은 숙련된 기술 작가입니다. 다음 문서를 분석하여, Vertex AI의 주요 기능과 RAG 아키텍처의 이점을 중심으로 비전문가도 이해하기 쉽게 세 문장으로 요약해 주십시오. 각 문장은 명확하고 간결해야 합니다.

원본 문서:
{original_document_content}

요약:
"""

improved_generation_config = GenerationConfig(
    temperature=0.1, # 더욱 결정적인 요약을 위해 온도 낮춤
    max_output_tokens=100 # 세 문장에 맞는 토큰 수로 조절
)

try:
    improved_response = baseline_model.generate_content( # 동일 모델 사용
        improved_prompt,
        generation_config=improved_generation_config
    )
    improved_summary_text = improved_response.text
    print("\n----- 개선된 프롬프트 기반 요약 -----")
    print(improved_summary_text)
except Exception as e:
    print(f"개선된 요약 생성 중 오류 발생: {e}")
    improved_summary_text = "오류로 인해 요약을 생성할 수 없습니다."
