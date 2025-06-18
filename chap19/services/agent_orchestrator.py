# 파일 경로: services/agent_orchestrator.py

import os
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Tool
from google.cloud.aiplatform_v1beta1 import Tool as GapicTool
# from vertexai.generative_models import (
#     GenerativeModel,
#     Tool,
# )
from langchain_core.tools import tool

from google import adk
from google.adk.agents import LlmAgent
from google.adk.sessions import VertexAiSessionService
from vertexai.generative_models import Part, Content
# 다른 서비스 모듈에서 실제 도구 구현 함수를 가져옵니다.
# 이 구조는 각 모듈이 자신의 책임에만 집중하도록 합니다.
from.visualization_service import generate_chart_base64
from.audio_service import AudioService

# --- 전역 서비스 인스턴스 ---
audio_service_instance: AudioService = None

def initialize_audio_service(instance: AudioService):
    """app.py에서 생성된 AudioService 인스턴스를 이 모듈에 주입합니다."""
    global audio_service_instance
    if not audio_service_instance:
        audio_service_instance = instance
        print("AgentOrchestrator가 AudioService 인스턴스를 성공적으로 받았습니다.")

# --- 도구(Tool) 정의 ---
# Chapter 14, 15, 17에서 설명된 멀티모달 기능을 에이전트가 사용할 수 있는 '도구'로 변환합니다.

# 1. 데이터 시각화 도구 (Chapter 15, 17)
visualization_tool = generate_chart_base64

# visualization_tool = Tool(
#     function_declarations=[generate_chart_base64]
# )

# 2. 음성 합성(TTS) 도구 (Chapter 14)
def synthesize_speech(text: str) -> str:
    """
    주어진 텍스트를 음성으로 변환하고, 재생 가능한 오디오 파일의 URL을 반환합니다.
    에이전트가 텍스트 응답을 음성으로 제공해야 할 때 이 도구를 사용합니다.
    Args:
        text (str): 음성으로 변환할 한국어 텍스트.
    Returns:
        str: 오디오 파일에 접근할 수 있는 GCS 서명된 URL.
    """
    if not audio_service_instance:
        return "오류: 오디오 서비스가 초기화되지 않았습니다."
    try:
        return audio_service_instance.synthesize_and_get_signed_url(text)
    except Exception as e:
        return f"오류: 음성 합성에 실패했습니다. {e}"

tts_tool = synthesize_speech
# tts_tool = Tool(
#     function_declarations=[synthesize_speech]
# )

# 3. Google 검색 도구 (RAG의 일종으로, 최신 정보 검색에 활용)
# [수정됨] 함수 이름의 공백을 제거하고 올바른 이름으로 변경합니다.
# search_tool = Tool.from_Google_Search_retrieval()
low_level_tool = GapicTool(
    google_search=GapicTool.GoogleSearch(),
)
# search_tool = Tool._from_gapic(raw_tool=low_level_tool)
def perform_actual_google_search(query: str) -> str:
    """실제 Google 검색을 수행하는 로직"""
    # 여기에 실제 검색 API 연동 코드를 구현합니다.
    print(f"Searching Google for: {query}")
    return f"'{query}'에 대한 실제 검색 결과입니다."
@tool
def search_tool(query: str) -> str:
    """최신 정보나 로컬 문서에서 다루지 않는 주제에 대한 질문에 답하기 위해 Google을 검색하는 데 사용됩니다."""
    return perform_actual_google_search(query)


# --- ADK 에이전트 정의 ---
doc_qa_agent = LlmAgent(
    model="gemini-2.0-flash-lite-001",
    tools=[search_tool, visualization_tool, tts_tool],
    name="doc_qa_agent",
    description="사용자 문서와 웹 검색을 기반으로 질문에 답하고, 필요시 시각화나 음성 응답을 생성하는 에이전트",
    instruction=
        "당신은 사용자의 질문을 해결하기 위해 주어진 도구를 적극적으로 사용하는 지능형 AI 조력자입니다."+
        "답변은 항상 한국어로 제공해야 합니다."+
        "정보가 부족하면 먼저 웹 검색(search_tool)을 사용하세요."+
        "사용자가 차트를 요청하면 'visualization_tool'을 사용하고, 음성 응답을 요청하면 'tts_tool'을 사용하세요."
    
)


class AgentOrchestrator:
    """
    ADK Runner와 SessionService를 사용하여 복잡한 워크플로와 영구 세션을 관리하는 서비스.
    """
    def __init__(self, project_id: str, location: str, app_name: str = "notebooklm-agent"):
        """
        서비스 초기화 시 Vertex AI, SessionService, ADK Runner를 설정합니다.
        """
        if not project_id or not location:
            raise ValueError("GCP 프로젝트 ID와 위치는 필수입니다.")
        
        print(f"AgentOrchestrator (ADK) 초기화 중... 프로젝트: {project_id}, 위치: {location}")
        vertexai.init(project=project_id, location=location)
        print("vertexai.init 완료 app_name=", app_name)        
        self.app_name = app_name
        
        # 1. Firestore를 백엔드로 사용하는 영구 세션 서비스 초기화
        self.session_service = VertexAiSessionService(project=project_id, location=location)
        print("vVertexAiSessionService 완료 self.session_service=", self.session_service)        
        
        # 2. 정의된 에이전트와 세션 서비스를 사용하여 ADK Runner 초기화
        self.runner = adk.Runner(
            agent=doc_qa_agent,
            app_name=self.app_name,
            session_service=self.session_service
        )
        print("AgentOrchestrator (ADK) 초기화 완료. Runner 및 SessionService 준비됨.")

    async def invoke_agent_streaming_async(self, session_id: str, query: str, user_id: str = "default-user"):
        """
        주어진 쿼리로 ADK Runner를 비동기적으로 실행하고, 응답 이벤트 스트림을 반환합니다.
        """
        print(f"ADK Runner 실행 시작: session_id={session_id}, query='{query[:50]}...'")
        
        # 3. 세션 가져오기 또는 생성 (Firestore와 연동)
        try:
            session = await self.session_service.get_session(app_name=self.app_name, session_id=session_id)
        except Exception:
            print(f"세션 {session_id}를 찾을 수 없어 새로 생성합니다.")
            session = await self.session_service.create_session(
                app_name=self.app_name,
                user_id=user_id,
                session_id=session_id
            )
            
        # 4. 사용자 메시지를 ADK 형식으로 변환
        new_message = Content(role='user', parts=[Part.from_text(query)])
        
        # 5. ADK Runner를 통해 에이전트 실행 및 이벤트 스트리밍
        try:
            async for event in self.runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=new_message
            ):
                yield event 
            print(f"ADK Runner 실행 완료: session_id={session_id}")

        except Exception as e:
            print(f"ADK Runner 실행 중 심각한 오류 발생: {e}")
            raise