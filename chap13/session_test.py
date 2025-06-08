import os
from dotenv import load_dotenv
import vertexai # vertexai 라이브러리를 직접 import 합니다.

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# 최신 라이브러리 버전에 맞는 import 경로를 사용합니다.
from vertexai.preview.reasoning_engines import AdkApp
from google.adk.agents import Agent

# --- Vertex AI SDK 초기화 ---
# 이 부분이 가장 중요합니다.
# ADK가 어떤 GCP 프로젝트와 리전에서 모델을 찾아야 할지 알려줍니다.
try:
    # 1. gcloud에서 설정한 프로젝트 ID를 사용합니다.
    PROJECT_ID = "notebookagent-1"
    # 2. 모델이 서비스되는 리전을 지정합니다. (us-central1이 일반적)
    LOCATION = "us-central1"
    
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    print(f"Vertex AI가 '{PROJECT_ID}' 프로젝트, '{LOCATION}' 리전으로 초기화되었습니다.")

except Exception as e:
    print(f"Vertex AI 초기화 중 오류 발생: {e}")


# 1. 에이전트가 사용할 도구(Tool) 정의
def get_exchange_rate(currency_code: str, date: str) -> str:
    """
    지정된 날짜와 통화 코드를 기준으로 미국 달러(USD) 대비 환율을 가져옵니다.
    """
    print(f"\n--- 툴 호출됨: get_exchange_rate(currency_code='{currency_code}', date='{date}') ---")
    
    if currency_code == "SEK" and date == "2025-04-03":
        return f"{date} 기준, 1 미국 달러는 10.52 스웨덴 크로나(SEK)입니다."
    else:
        return f"'{date}'의 '{currency_code}' 환율 정보를 찾을 수 없습니다."

# 2. 에이전트(Agent) 생성
try:
    my_agent = Agent(
        name="financial_assistant",
        # Vertex AI 플랫폼에서 인식하는 모델 이름
        model="gemini-2.0-flash-001",
        instruction="""
        당신은 환율 정보를 제공하는 전문 금융 어시스턴트입니다.
        사용자가 환율을 질문하면, 반드시 `get_exchange_rate` 도구를 사용해야 합니다.
        만약 도구 사용에 필요한 '통화 코드'나 '날짜' 정보가 부족하면, 사용자에게 정중하게 되물어보세요.
        모든 정보를 받으면 도구를 호출하고, 그 결과를 바탕으로 완전한 문장으로 답변하세요.
        """,
        tools=[get_exchange_rate],
    )
    print("에이전트가 성공적으로 생성되었습니다.")
except Exception as e:
    print(f"에이전트 생성 실패. 오류: {e}")
    my_agent = None


# 3. AdkApp으로 에이전트 실행 및 대화 시뮬레이션
def run_conversation_flow():
    """요청된 대화 시나리오를 순서대로 실행합니다."""
    
    if not my_agent:
        print("에이전트가 없어 실행을 중단합니다.")
        return

    app = AdkApp(agent=my_agent)
    
    user_id = "USER1"
    session = app.create_session(user_id=user_id)
    session_id = session.id
    print(f"\n[시스템] '{user_id}' 사용자의 세션이 생성되었습니다. (ID: {session_id})")
    
    # --- 첫 번째 사용자 질문 ---
    print("\n" + "="*25)
    print("1. 첫 번째 질문: 정보가 불충분한 경우")
    print("="*25)
    
    first_message = "2025-04-03의 미달러 대비 스웨덴 크로나 환율은?"
    print(f"사용자: {first_message}")
    print("에이전트 응답:")

    response_text = ""
    for event in app.stream_query(user_id=user_id, session_id=session_id, message=first_message):
        if hasattr(event, 'text'):
            response_text += event.text
            print(event.text, end="", flush=True)
    print()

    # --- 두 번째 사용자 입력 (세션 유지) ---
    print("\n" + "="*25)
    print("2. 두 번째 입력: 추가 정보 제공")
    print("="*25)
    
    second_message = "SEK"
    print(f"사용자: {second_message}")
    print("에이전트 응답:")

    response_text = ""
    for event in app.stream_query(user_id=user_id, session_id=session_id, message=second_message):
        if hasattr(event, 'text'):
            response_text += event.text
            print(event.text, end="", flush=True)
    print()

    print("\n\n[시스템] 대화 시나리오가 성공적으로 종료되었습니다.")


if __name__ == "__main__":
    run_conversation_flow()