import os
import asyncio
import json
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from google.cloud import storage

# --- 서비스 모듈 임포트 ---
# 1. 각 모듈에서 필요한 클래스와 함수를 가져옵니다.
from services.agent_orchestrator import AgentOrchestrator, initialize_audio_service
from services.audio_service import AudioService

#.env 파일에서 환경 변수 로드
load_dotenv()

# --- 전역 변수 및 설정 ---
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "a_very_secret_key_for_development")
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

# --- Flask 및 확장 프로그램 초기화 ---
app = Flask(__name__)
app.config = FLASK_SECRET_KEY
CORS(app, resources={r"/api/*": {"origins": "*"}}) # 프로덕션에서는 특정 도메인으로 제한

# --- 서비스 인스턴스 생성 ---
# 2. 애플리케이션 시작 시, 모든 서비스를 단 한 번만 생성합니다.
try:
    agent_service = AgentOrchestrator(project_id=PROJECT_ID, location=LOCATION)
    audio_service = AudioService(project_id=PROJECT_ID, bucket_name=GCS_BUCKET_NAME)
    storage_client = storage.Client(project=PROJECT_ID)
    
    # AgentOrchestrator가 AudioService를 사용할 수 있도록 인스턴스를 주입합니다.
    initialize_audio_service(audio_service)
    print("모든 서비스가 성공적으로 초기화되었습니다.")
except Exception as e:
    print(f"서비스 초기화 중 심각한 오류 발생: {e}")
    agent_service = None
    storage_client = None

# 인메모리 세션 관리 (프로덕션에서는 Redis 등 외부 저장소 권장)
chat_sessions = {}

# --- 헬퍼 함수 ---
def allowed_file(filename):
    """허용된 파일 확장자인지 확인하는 헬퍼 함수"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- API 엔드포인트 ---
@app.route('/')
def index():
    return "NotebookLM-Style AI Agent Backend is running."

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    NotebookLM의 핵심 기능인 문서 업로드를 처리하는 REST API 엔드포인트입니다.
    이 기능은 AI 추론과 별개로, 웹 서버의 파일 처리 책임에 해당합니다.
    """
    if 'file' not in request.files:
        return jsonify({"error": "요청에 파일 부분이 없습니다."}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "선택된 파일이 없습니다."}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        try:
            # 파일을 GCS에 업로드합니다.
            # RAG 파이프라인이 이 버킷의 'documents/' 폴더를 모니터링한다고 가정합니다.
            bucket = storage_client.bucket(GCS_BUCKET_NAME)
            blob = bucket.blob(f"documents/{filename}")
            
            blob.upload_from_file(file)
            
            print(f"파일 '{filename}'이(가) GCS 버킷 '{GCS_BUCKET_NAME}'에 업로드되었습니다.")
            
            # 실제 RAG 파이프라인에서는 이 업로드 이벤트를 트리거로
            # 자동 임베딩 프로세스를 시작해야 합니다. (예: GCS Eventarc -> Cloud Function)
            
            return jsonify({"message": f"파일 '{filename}' 업로드 성공"}), 201
        
        except Exception as e:
            print(f"GCS 업로드 중 오류 발생: {e}")
            return jsonify({"error": "파일을 저장하는 동안 서버 오류가 발생했습니다."}), 500
            
    return jsonify({"error": "허용되지 않는 파일 형식입니다."}), 400

@app.route('/api/chat', methods=['POST'])
async def chat_endpoint():
    """
    Vercel AI SDK의 useChat 훅과 연동되는 HTTP 스트리밍 엔드포인트입니다.
    실제 AI 추론 로직은 agent_service 객체에 모두 위임합니다.
    """
    if not agent_service:
        return jsonify({"error": "AI 서비스가 초기화되지 않았습니다."}), 503

    data = await request.get_json()
    messages = data.get('messages',)
    session_id = data.get('session_id', 'default-session')
    
    async def generate():
        # 세션 가져오기 또는 생성
        if session_id not in chat_sessions:
            chat_sessions[session_id] = agent_service.start_chat()
        chat = chat_sessions[session_id]

        last_user_message = messages[-1]['content'] if messages and messages[-1]['role'] == 'user' else ""

        if not last_user_message:
            yield f'x:{json.dumps({"error": "User message not found"})}\n'
            return

        try:
            # 3. AI 관련 모든 처리는 agent_service 객체에 위임합니다.
            response_stream = agent_service.invoke_agent_streaming_async(chat, last_user_message)
            async for chunk in response_stream:
                if hasattr(chunk, 'text'):
                    yield f'0:{json.dumps(chunk.text)}\n'
        except Exception as e:
            error_data = {"message": f"에이전트 처리 중 오류: {str(e)}"}
            yield f'x:{json.dumps(error_data)}\n'

    return Response(generate(), mimetype='text/plain')

# --- 주 실행 블록 ---
if __name__ == '__main__':
    # 프로덕션 환경에서는 Gunicorn과 같은 WSGI 서버를 사용해야 합니다.
    # 예: gunicorn --worker-class gevent --bind 0.0.0.0:8080 app:app
    app.run(debug=True, port=5001)
