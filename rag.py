import os
from dotenv import load_dotenv
load_dotenv()
import vertexai
from vertexai import rag
from vertexai.generative_models import TextGenerationModel, Tool

project_id = os.environ["GOOGLE_CLOUD_PROJECT"]
project_location = os.environ["GOOGLE_CLOUD_LOCATION"]
# 1) Vertex AI 프로젝트 및 임베딩 설정
vertexai.init(project=project_id, location=project_location)
embedding_model = rag.RagEmbeddingModelConfig(
    vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
        publisher_model="publishers/google/models/text-embedding-005"
    )
)

# RAG 코퍼스 생성 및 문서 임포트 (이미 문서가 있다면 생략 가능)
corpus = rag.create_corpus(display_name="notebooklm_corpus", backend_config=
    rag.RagVectorDbConfig(rag_embedding_model_config=embedding_model))
rag.import_files(corpus.name, paths=["gs://my-bucket/documents/"],   # GCS 또는 Drive 경로
    transformation_config=rag.TransformationConfig(
        chunking_config=rag.ChunkingConfig(chunk_size=512, chunk_overlap=50)
    )
)
# 위 import_files가 문서를 읽어 512토막으로 chunking 후 임베딩하여 벡터DB에 저장합니다.

# 2) RAG 검색 Tool 생성 및 LLM 초기화
# 생성한 코퍼스를 사용하도록 Retrieval 설정
retrieval = rag.Retrieval(source=rag.VertexRagStore(
    rag_resources=[ rag.RagResource(rag_corpus=corpus.name) ],
    rag_retrieval_config=rag.RagRetrievalConfig(top_k=3)
))
rag_tool = Tool.from_retrieval(retrieval=retrieval)  # RAG 검색 툴 생성

# Text Generation 모델에 RAG Tool 장착 (LLM 에이전트 초기화)
llm_agent = TextGenerationModel.from_pretrained("text-bison@001", tools=[rag_tool])

# 3) 사용자 질문에 대한 답변 생성
user_question = "이 문서에서 Vertex AI Agent Builder의 역할은 무엇인가요?"
response = llm_agent.predict(user_question)
print(response.text)  # 에이전트의 답변 출력
