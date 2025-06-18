import os
import asyncio
import json
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from google.cloud import storage
from google import adk # ADK 이벤트 처리를 위해 임포트

# 서비스 모듈 임포트
# agent_service는 더 이상 사용하지 않음
from services import agent_orchestrator
from services.agent_orchestrator import AgentOrchestrator
from services.audio_service import AudioService

# .env 파일에서 환경 변수 로드
load_dotenv()

# --- 전역 변수 및 설정 ---
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "a_very_secret_key_for_development")
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

# --- Flask 및 확장 프로그램 초기화 ---
app = Flask(__name__)
app.config['SECRET_KEY'] = FLASK_SECRET_KEY
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# CORS 설정: Vercel AI SDK의 useChat 훅과 통신하기 위해 필요
CORS(app, resources={r"/api/*": {"origins": "*"}})

# --- 서비스 인스턴스 생성 ---
# 애플리케이션 시작 시 한 번만 초기화하여 재사용
orchestrator_instance = None
audio_service_instance = None
storage_client = None

try:
    # 1. Google Cloud Storage 클라이언트 초기화
    storage_client = storage.Client(project=PROJECT_ID)
    print("Google Cloud Storage client initialized.")

    # 2. 오디오 서비스 초기화
    audio_service_instance = AudioService(project_id=PROJECT_ID, bucket_name=GCS_BUCKET_NAME)
    print("AudioService initialized.")

    # 3. 에이전트 오케스트레이터 초기화
    orchestrator_instance = AgentOrchestrator(project_id=PROJECT_ID, location=LOCATION)
    print("AgentOrchestrator initialized.")
    
    # 4. 오케스트레이터에 오디오 서비스 인스턴스 주입 (의존성 주입)
    # 이를 통해 오케스트레이터의 도구(tool)가 오디오 서비스를 사용할 수 있게 됨
    agent_orchestrator.initialize_audio_service(audio_service_instance)
    
    print("All services have been successfully initialized.")

except Exception as e:
    print(f"A critical error occurred during service initialization: {e}")
    # 프로덕션 환경에서는 이 부분에서 애플리케이션을 종료하거나 fallback 로직을 수행해야 할 수 있습니다.

# --- 헬퍼 함수 ---
def allowed_file(filename):
    """허용된 파일 확장자인지 확인"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- API 엔드포인트 ---
@app.route('/')
def index():
    return "NotebookLM-Style AI Agent Backend (ADK-based) is running."

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """문서 업로드를 처리하는 REST API 엔드포인트"""
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        try:
            if not storage_client:
                raise ConnectionError("Storage client is not initialized.")
            bucket = storage_client.bucket(GCS_BUCKET_NAME)
            blob = bucket.blob(f"documents/{filename}")
            blob.upload_from_file(file.stream)
            
            print(f"File '{filename}' uploaded to GCS bucket '{GCS_BUCKET_NAME}'.")
            return jsonify({"message": f"File '{filename}' uploaded successfully."}), 201
        
        except Exception as e:
            print(f"An error occurred during GCS upload: {e}")
            return jsonify({"error": "An internal server error occurred while saving the file."}), 500
            
    return jsonify({"error": "File type not allowed."}), 400


async def stream_agent_response(messages, session_id):
    """
    AgentOrchestrator 응답을 Vercel AI SDK 데이터 프로토콜에 맞춰 스트리밍하는 비동기 제너레이터.
    AgentService를 사용하던 기존 로직을 ADK 이벤트 처리 로직으로 완전히 대체합니다.
    """
    if not orchestrator_instance:
        error_data = {"error": "Agent Orchestrator service is not available."}
        yield f'x:{json.dumps(error_data)}\n'
        return

    # useChat에서 보낸 메시지 배열의 마지막 메시지가 현재 사용자 쿼리
    last_user_message = ""
    if messages and messages[-1]['role'] == 'user':
        last_user_message = messages[-1]['content']

    if not last_user_message:
        error_data = {"error": "No user message found."}
        yield f'x:{json.dumps(error_data)}\n'
        return
        
    try:
        # AgentOrchestrator의 스트리밍 메서드 호출
        response_stream = orchestrator_instance.invoke_agent_streaming_async(
            session_id=session_id,
            query=last_user_message
        )
        
        # ADK가 생성하는 이벤트를 실시간으로 처리
        async for event in response_stream:
            # 1. 텍스트 이벤트 처리
            if event.type == adk.Event.TEXT:
                text_chunk = event.data.get('text', '')
                if text_chunk:
                    # Vercel AI SDK의 텍스트 스트림 프로토콜 (0: "text_chunk")
                    yield f'0:{json.dumps(text_chunk)}\n'

            # 2. 도구 출력(Tool Output) 이벤트 처리 (멀티모달 데이터)
            elif event.type == adk.Event.TOOL_OUTPUT:
                tool_outputs = event.data.get('tool_output', [])
                for output in tool_outputs:
                    tool_name = output.get('tool_name')
                    tool_result = output.get('output')
                    
                    if not tool_result:
                        continue
                        
                    final_data = {}
                    # 시각화 도구 결과 처리
                    if tool_name == 'generate_chart_base64':
                        final_data['chart_base64'] = tool_result
                        print("Generated and streaming chart data.")
                    # 음성 합성 도구 결과 처리
                    elif tool_name == 'synthesize_speech':
                        final_data['audio_url'] = tool_result
                        print(f"Generated and streaming audio URL: {tool_result}")
                    
                    # Vercel AI SDK의 JSON 데이터 스트림 프로토콜 (j: {"key": "value"})
                    if final_data:
                        yield f'j:{json.dumps(final_data)}\n'
            
            # 이벤트 사이에 약간의 지연을 주어 클라이언트 측 렌더링을 원활하게 함
            await asyncio.sleep(0.01)

    except Exception as e:
        print(f"An error occurred during agent response streaming: {e}")
        error_data = {"message": f"An error occurred in the agent process: {str(e)}"}
        # Vercel AI SDK의 오류 스트림 프로토콜 (x: {"error_details"})
        yield f'x:{json.dumps(error_data)}\n'


@app.route('/api/chat', methods=['POST'])
async def chat_endpoint():
    """
    Vercel AI SDK의 useChat 훅과 연동되는 HTTP 스트리밍 엔드포인트.
    """
    try:
        data = await request.get_json()
        if not data or 'messages' not in data:
            return jsonify({"error": "Invalid request body, 'messages' key is missing."}), 400

        messages = data['messages']
        # Vercel AI SDK에서 보낸 'data' 객체에서 session_id를 가져옴
        # 없으면 'default-session' 사용 (단, 사용자별 세션 구분을 위해 고유 ID 사용 권장)
        session_id = data.get('data', {}).get('session_id', 'default-session')

        # 비동기 제너레이터를 스트리밍 응답으로 변환
        return Response(stream_agent_response(messages, session_id), mimetype='text/plain')

    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500

# --- 주 실행 블록 ---
if __name__ == '__main__':
    # 개발용 서버 실행. 프로덕션 환경에서는 Gunicorn과 같은 WSGI 서버 사용
    # 예: gunicorn --worker-class gevent --bind 0.0.0.0:8080 app:app
    app.run(debug=True, port=int(os.getenv("PORT", 8080)))
