# services/agent_service.py

import os
import uuid
import vertexai
from vertexai.preview import adk
from vertexai.preview.generative_models import (
    Tool,
    Content,
    Part,
    GenerationConfig,
    GenerativeModel
)
# 이 예제에서는 도구를 직접 정의하지만, 실제로는 별도 파일에서 가져올 수 있다.
# from.tools import all_tools 

# --- 도구 정의 (예시) ---
# 실제 애플리케이션에서는 이 도구들을 별도의 tools.py 파일에서 관리한다.
# 여기서는 설명을 위해 agent_service 내에 간단히 정의한다.
# Chapter 15, 17에서 정의한 시각화/오디오 도구 스키마를 여기에 통합한다.
# 이 예제에서는 간단한 검색 도구만 정의한다.
search_tool = Tool.from_google_search_retrieval()

class AgentService:
    """
    Vertex AI Agent와의 상호작용을 캡슐화하는 서비스 클래스.
    """
    def __init__(self, project_id: str, location: str):
        """
        서비스 초기화 시 Vertex AI SDK 및 모델을 설정한다.
        """
        if not project_id or not location:
            raise ValueError("GCP 프로젝트 ID와 위치는 필수입니다.")
        
        print(f"AgentService 초기화 중... 프로젝트: {project_id}, 위치: {location}")
        vertexai.init(project=project_id, location=location)
        
        # GenerativeModel을 에이전트로 사용. 도구를 장착한다.
        # 실제로는 Chapter 8, 12 등에서 설계한 복합 에이전트가 될 수 있다.
        self.model = GenerativeModel(
            "gemini-2.0-flash-001",
            tools=[search_tool], # 여기에 모든 커스텀 도구를 추가
            system_instruction=(
                "당신은 사용자의 문서를 기반으로 질문에 답하고, 데이터를 요약하며, 필요시 시각화나 음성 응답을 제공하는 지능형 AI 조력자입니다."
                "답변은 항상 한국어로 제공해야 합니다."
            )
        )
        print("AgentService 초기화 완료. Gemini 모델 및 도구 준비됨.")

    def start_chat(self):
        """
        새로운 대화 세션을 시작한다.
        """
        return self.model.start_chat()

    async def invoke_agent_streaming_async(self, chat_session, query: str):
        """
        주어진 쿼리로 에이전트를 비동기적으로 호출하고,
        응답 이벤트 스트림을 비동기 제너레이터로 반환한다.
        
        Args:
            chat_session: model.start_chat()으로 생성된 채팅 세션 객체.
            query: 사용자의 질문.

        Yields:
            응답 스트림의 각 부분(chunk).
        """
        print(f"에이전트 스트리밍 호출: '{query[:50]}...'")
        try:
            # send_message를 stream=True로 호출하여 비동기 제너레이터를 얻는다.
            response_stream = await chat_session.send_message_async(
                query,
                stream=True,
                generation_config=GenerationConfig(temperature=0.3)
            )
            
            # 응답 스트림을 순회하며 각 청크를 그대로 반환(yield)한다.
            async for chunk in response_stream:
                yield chunk
            print("에이전트 스트리밍 호출 완료.")

        except Exception as e:
            print(f"에이전트 스트리밍 중 오류 발생: {e}")
            # 오류 발생 시에도 제너레이터를 통해 오류 정보를 전달할 수 있다.
            # 이 예제에서는 간단히 예외를 로깅하고 종료한다.
            raise