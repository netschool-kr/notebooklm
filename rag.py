# rag.py (쿼터 초과 예외 처리 및 대체 모델 적용 버전)

import os
from dotenv import load_dotenv
from google.cloud import aiplatform
from vertexai.preview import rag
from vertexai.generative_models import GenerativeModel, Tool
import logging

logging.basicConfig(level=logging.DEBUG)
load_dotenv()

project_id = os.environ["GOOGLE_CLOUD_PROJECT"]
project_location = os.environ["GOOGLE_CLOUD_LOCATION"]
print("project_id=", project_id, " : project_location=", project_location)

# 1) Vertex AI 및 임베딩 모델 초기화
aiplatform.init(project=project_id, location=project_location)

#    ──────────────────────────────────────────────────────────────────────────
#    기본으로 'text-embedding-gecko' 모델을 사용하려다가 쿼터 문제 발생 시를 대비하여
#    대체 임베딩 모델을 지정할 수 있도록 분기 처리
#    예) "text-embedding-gecko@001" 대신 "text-embedding-gecko@latest" 또는 
#         GCP 콘솔에서 쿼터가 충분한 다른 임베딩 모델 명시
#    ──────────────────────────────────────────────────────────────────────────
try:
    embedding_endpoint = rag.VertexPredictionEndpoint(
        # 쿼터 초과 문제가 잦은 'text-embedding-gecko' 대신, 
        # 사용 가능한 다른 버전(예: '@latest')을 명시해 봅니다.
        publisher_model="publishers/google/models/text-embedding-gecko@001"
    )
    print("▶ 임베딩 모델: text-embedding-gecko@001 사용 시도")
except Exception as e:
    # 만약 위 엔드포인트 호출 자체가 실패하면, 다른 모델로 폴백
    print("⚠️ text-embedding-gecko@001 엔드포인트 생성 실패:", e)
    embedding_endpoint = rag.VertexPredictionEndpoint(
        publisher_model="publishers/google/models/text-embedding-gecko@latest"
    )
    print("▶ 임베딩 모델: text-embedding-gecko@latest 사용")

embedding_model = rag.RagEmbeddingModelConfig(
    vertex_prediction_endpoint=embedding_endpoint
)

# 2) RAG 코퍼스 생성 및 문서 가져오기
#    (코퍼스가 이미 존재할 수도 있으므로 예외 처리)
try:
    corpus = rag.create_corpus(
        display_name="notebooklm_corpus",
        backend_config=rag.RagVectorDbConfig(rag_embedding_model_config=embedding_model)
    )
    print("▶ Corpus 생성됨:", corpus.name)
except Exception as e:
    # 이미 존재하거나 다른 이유로 실패한 경우
    print("⚠️ Corpus 생성 실패 또는 이미 존재:", e)
    corpus = None

#    ──────────────────────────────────────────────────────────────────────────
#    이미 만들어진 corpus 이름을 알고 있다면, 위 예외 발생 시 바로 그 이름을 직접 지정해도 됩니다.
#    예) corpus = rag.get_corpus(name="projects/.../ragCorpora/1234567890")
#    ──────────────────────────────────────────────────────────────────────────

if corpus:
    try:
        rag.import_files(
            corpus.name,
            paths=["gs://notebook-docs/"],  # 실제 문서가 저장된 GCS 버킷 URI
            transformation_config=rag.TransformationConfig(
                chunking_config=rag.ChunkingConfig(chunk_size=512, chunk_overlap=50)
            )
        )
        print("▶ import_files 호출 성공")
    except Exception as e:
        print("⚠️ import_files 호출 중 오류:", e)

# 3) Retrieval 객체 설정
#    RAG 조회를 위해 Retrieval 객체를 미리 만들어 두지만,
#    실제 조회는 rag.retrieval_query(...)를 사용합니다.
retrieval = None
if corpus:
    try:
        retrieval = rag.Retrieval(
            source=rag.VertexRagStore(
                rag_resources=[rag.RagResource(rag_corpus=corpus.name)],
                rag_retrieval_config=rag.RagRetrievalConfig(top_k=3)
            )
        )
        print("▶ Retrieval 객체 생성됨")
    except Exception as e:
        print("⚠️ Retrieval 객체 생성 실패:", e)

# 4) 사용자 질문 정의
user_question = "이 문서에서 Vertex AI Agent Builder의 역할은 무엇인가요?"

# 5) retrieval_query를 사용하여 컨텍스트 생성
context = ""
if corpus:
    try:
        print("▶ retrieval_query 호출 시도...")
        response_rag = rag.retrieval_query(
            rag_resources=[rag.RagResource(rag_corpus=corpus.name)],
            text=user_question,
            rag_retrieval_config=rag.RagRetrievalConfig(top_k=3),
        )
        # 'response_rag.contexts.contexts'는 반복 가능한 컨텍스트 리스트
        context_texts = [ctx.text for ctx in response_rag.contexts.contexts]
        context = "\n".join(context_texts)

        print(">> RAG 조회 성공, 컨텍스트:")
        for i, txt in enumerate(context_texts, start=1):
            print(f"  [Context {i}] {txt}")

    except Exception as e:
        print("✨ RAG 조회 실패:", e)
        # 쿼터 문제 외에도, 네트워크/권한 오류일 수 있으므로
        # 예외가 발생하면 “시뮬레이션 컨텍스트”를 fallback으로 사용
        context = "\n".join([
            "Vertex AI Agent Builder는 대화형 에이전트를 생성하는 기능을 제공합니다.",
            "이는 Retrieval-Augmented Generation(RAG) 작업을 통해 더욱 풍부한 답변을 만들 수 있도록 지원합니다.",
            "Vertex AI의 임베딩 및 생성 모델과 긴밀히 통합되어 있습니다."
        ])
else:
    # corpus 자체가 없을 경우 기본 시뮬레이션 컨텍스트 사용
    context = "\n".join([
        "Vertex AI Agent Builder는 대화형 에이전트를 생성하는 기능을 제공합니다.",
        "이는 Retrieval-Augmented Generation(RAG) 작업을 통해 더욱 풍부한 답변을 만들 수 있도록 지원합니다.",
        "Vertex AI의 임베딩 및 생성 모델과 긴밀히 통합되어 있습니다."
    ])

# 6) RAG 도구 생성 (선택 사항, 모델 통합용)
rag_tool = None
if retrieval:
    try:
        rag_tool = Tool.from_retrieval(retrieval=retrieval)
        print("▶ RAG 도구(Tool) 생성됨")
    except Exception as e:
        print("⚠️ RAG 도구 생성 실패:", e)

# 7) 생성형 모델 초기화
#    - 쿼터 문제로 인해, 특정 LLM 모델이 제한될 수 있으므로
#      필요하다면 다른 LLM(예: "text-bison@001")으로 폴백할 수도 있습니다.
try:
    llm_agent = GenerativeModel("gemini-2.0-flash-001", tools=[rag_tool] if rag_tool else [])
    print("▶ LLM 에이전트(gemini-2.0-flash-001) 초기화 성공")
except Exception as e:
    print("⚠️ 기본 LLM 초기화 실패:", e)
    # 예를 들어, "text-bison@001" 으로 폴백
    llm_agent = GenerativeModel("text-bison@001", tools=[rag_tool] if rag_tool else [])
    print("▶ LLM 에이전트(text-bison@001)로 폴백")

# 8) 컨텍스트와 질문을 프롬프트로 결합
prompt = f"Context: {context}\nQuestion: {user_question}"

# 9) 응답 생성 (예외 처리 포함)
try:
    print("▶ LLM 응답 생성 시도...")
    response = llm_agent.generate_content(prompt)
    print("\n>> LLM 응답:\n", response.text)

except Exception as e:
    # 쿼터 초과(ResourceExhausted) 등 예외 처리
    print("🔥 LLM 생성 실패:", e)
    # 간단한 fallback 메시지 출력
    print("\n>> Fallback 응답:\n",
          "현재 요청이 많아 답변을 생성할 수 없습니다. 잠시 후 다시 시도해 주세요.")

