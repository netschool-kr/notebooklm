# Vertex AI Agent Builder와 RAG Engine을 활용한 PDF 문서 임베딩 및 요약 Python 스크립트
#
# 목적:
# 본 스크립트는 "NotebookLM 스타일 AI Agent 개발.pdf" 문서의
# "Chapter 13: Vertex AI Agent Builder와 RAG Engine으로 문서 요약 에이전트 개발" 내용을 기반으로,
# "NotebookLM" 프로젝트 컨텍스트에서 Google Cloud Storage(GCS) "notebook-docs" 버킷에 있는
# PDF 파일을 임베딩하고 요약하는 전체 과정을 시연합니다.
#
# 주요 기능:
# 1. Vertex AI RAG Corpus 설정 또는 참조
# 2. 지정된 GCS 버킷("notebook-docs")에서 PDF 파일 목록 조회
# 3. PDF 파일을 RAG Corpus로 임포트 (파싱, 청킹, 임베딩 자동 수행)
# 4. RAG Corpus 기반의 Retrieval Tool 설정
# 5. GenerativeModel과 RAG Tool을 통합한 요약 에이전트 정의
# 6. 지정된 PDF에 대해 요약 에이전트를 호출하여 요약 생성
#
# 이 스크립트는 Vertex AI Python SDK, 특히 `vertexai.rag` 및
# `vertexai.preview.generativeai` 모듈을 주로 사용합니다.

# ==============================================================================
# 섹션 1: 스크립트 설정 및 구성
# ==============================================================================
# 필요한 라이브러리 임포트
import os
import time
import uuid # 고유한 리소스 이름 생성에 사용될 수 있음

import vertexai
from vertexai.generative_models import GenerativeModel, Tool, GenerationConfig

from vertexai import rag # Chapter 13, p.136 [1]
from google.cloud import storage
from dotenv import load_dotenv #.env 파일에서 환경 변수 로드

#.env 파일에서 환경 변수 로드
load_dotenv()

# --- 주요 구성 파라미터 ---
# (참고: "NotebookLM 스타일 AI Agent 개발.pdf" 문서 Chapter 13 내용 기반)
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
GCS_BUCKET_NAME = "notebook-docs" # 사용자 요청에 명시된 버킷
GCS_PDF_PREFIX = os.getenv("GCS_PDF_PREFIX", "") # 예: "research_papers/" 또는 "" (버킷 루트)

RAG_CORPUS_DISPLAY_NAME = f"notebooklm-pdf-summary-corpus-{str(uuid.uuid4())[:8]}"

# 임베딩 모델 선택 (Chapter 13, p.135-136) [1]
EMBEDDING_MODEL_PUBLISHER_MODEL = os.getenv("EMBEDDING_MODEL_PUBLISHER_MODEL", "publishers/google/models/text-embedding-005")

# 생성 모델 선택 (Chapter 13, p.139) [1]
GENERATIVE_MODEL_NAME = os.getenv("GENERATIVE_MODEL_NAME", "text-bison@latest")

# 청킹 및 검색 파라미터 (Chapter 13, p.136-137) [1]
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "3"))

# 요약 생성 파라미터 (Chapter 13, p.138) [1]
SUMMARY_TEMPERATURE = float(os.getenv("SUMMARY_TEMPERATURE", "0.2"))
SUMMARY_MAX_OUTPUT_TOKENS = int(os.getenv("SUMMARY_MAX_OUTPUT_TOKENS", "512")) # Chapter 13에는 명시적 값 없으나 일반적 설정

# Vertex AI SDK 초기화 (Chapter 13, p.135) [1]
print(f"Vertex AI SDK 초기화 중... 프로젝트: {PROJECT_ID}, 위치: {LOCATION}")
vertexai.init(project=PROJECT_ID, location=LOCATION)
print("Vertex AI SDK 초기화 완료.")

# ==============================================================================
# 섹션 2: RAG Corpus 관리 (Chapter 13, 1단계) [1]
# ==============================================================================

def get_or_create_rag_corpus(display_name: str, embedding_model_uri: str) -> rag.RagCorpus:
    """
    지정된 표시 이름으로 RAG Corpus를 생성합니다.
    (참고: "NotebookLM 스타일 AI Agent 개발.pdf", Chapter 13, p.136 `rag.create_corpus`) [1]

    Args:
        display_name: RAG Corpus의 표시 이름.
        embedding_model_uri: 사용할 임베딩 모델의 URI.

    Returns:
        생성된 rag.RagCorpus 객체.
    """
    print(f"RAG Corpus '{display_name}' 생성 시도 중...")
    try:
        # RagEmbeddingModelConfig 설정 (Chapter 13, p.136의 embedding_config에 해당) [1]
        embedding_model_config = rag.RagEmbeddingModelConfig(
            vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
                publisher_model=embedding_model_uri
            )
        )
        # RagCorpus 생성 (Chapter 13, p.136) [1]
        # backend_config 내에 rag_embedding_model_config를 지정합니다.
        rag_corpus_instance = rag.create_corpus(
            display_name=display_name,
            backend_config=rag.RagVectorDbConfig( # Chapter 13, p.136의 backend_config [1]
                rag_embedding_model_config=embedding_model_config
            )
        )
        print(f"RAG Corpus '{rag_corpus_instance.name}' 생성 완료.")
        return rag_corpus_instance
    except Exception as e:
        print(f"RAG Corpus 생성 중 오류 발생: {e}")
        raise

def list_gcs_pdf_files(bucket_name: str, prefix: str) -> list[str]:
    """
    지정된 GCS 버킷과 접두사에서 PDF 파일 목록을 GCS URI 형태로 반환합니다.
    (Helper function, Chapter 13의 `paths` 변수 구성에 필요) [1]
    """
    print(f"GCS 버킷 '{bucket_name}'에서 '{prefix}' 접두사를 가진 PDF 파일 검색 중...")
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    pdf_uris = []# 수정된 부분: 빈 리스트로 초기화
    for blob in blobs:
        if blob.name.lower().endswith(".pdf"):
            pdf_uris.append(f"gs://{bucket_name}/{blob.name}")

    if pdf_uris:
        print(f"총 {len(pdf_uris)}개의 PDF 파일 발견.")
    else:
        print("해당 경로에서 PDF 파일을 찾을 수 없습니다.")
    return pdf_uris

def import_pdfs_to_corpus(rag_corpus_name: str,
                          gcs_pdf_uris: list[str],
                          chunk_size: int,
                          chunk_overlap: int):
    """
    지정된 GCS PDF 파일들을 RAG Corpus로 임포트합니다.
    (참고: "NotebookLM 스타일 AI Agent 개발.pdf", Chapter 13, p.136 `rag.import_files`) [1]

    Args:
        rag_corpus_name: 대상 RAG Corpus의 리소스 이름.
        gcs_pdf_uris: 임포트할 PDF 파일들의 GCS URI 목록.
        chunk_size: 청킹 시 사용할 청크 크기 (토큰 단위).
        chunk_overlap: 청킹 시 청크 간 중첩 크기 (토큰 단위).
    """
    if not gcs_pdf_uris:
        print("임포트할 PDF 파일이 없습니다. 임포트 건너뜀.")
        return

    print(f"RAG Corpus '{rag_corpus_name}'으로 PDF 파일 임포트 시작...")
    print(f"임포트 대상 파일: {gcs_pdf_uris}")

    transformation_config = rag.TransformationConfig(
        chunking_config=rag.ChunkingConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    )

    try:
        import_lro = rag.import_files(
            rag_corpus_name,
            paths=gcs_pdf_uris,
            transformation_config=transformation_config,
        )
        print(f"PDF 파일 임포트 작업 시작됨: {import_lro}. 완료까지 시간이 소요될 수 있습니다.")
        print("참고: 파일 임포트 및 인덱싱은 백그라운드에서 진행됩니다. 다음 단계 진행 전 충분한 시간을 기다려주세요.")
        # 데모를 위해 짧은 시간 대기 (실제로는 LRO 완료 확인 필요)
        time.sleep(60) # 파일 크기와 수에 따라 조절 필요.
        print("임포트 작업 진행을 위해 일정 시간 대기 완료.")

    except Exception as e:
        print(f"PDF 파일 임포트 중 오류 발생: {e}")
        raise

# ==============================================================================
# 섹션 3: RAG Retrieval Tool 구성 (Chapter 13, 2단계) [1]
# ==============================================================================

def configure_rag_retrieval(rag_corpus_resource_name: str, top_k: int = 3) -> rag.Retrieval:
    """
    지정된 RAG Corpus에 대한 Retrieval 메커니즘을 설정합니다.
    (참고: "NotebookLM 스타일 AI Agent 개발.pdf", Chapter 13, p.137) [1]

    Args:
        rag_corpus_resource_name: RAG Corpus의 리소스 이름 (rag_corpus.name).
        top_k: 검색할 상위 관련 청크의 수.

    Returns:
        설정된 rag.Retrieval 객체.
    """
    print(f"RAG Retrieval 설정 중... Corpus: {rag_corpus_resource_name}, Top K: {top_k}")

    rag_retrieval_config = rag.RagRetrievalConfig(top_k=top_k)
    rag_resources = [rag.RagResource(rag_corpus=rag_corpus_resource_name)]# Chapter 13, p.137 [1]

    retrieval_tool = rag.Retrieval(
        source=rag.VertexRagStore(rag_resources=rag_resources),
        #rag_retrieval_config=rag_retrieval_config
    )
    print("RAG Retrieval 설정 완료.")
    return retrieval_tool

# ==============================================================================
# 섹션 4: 요약 에이전트 정의 (Chapter 13, 4단계) [1]
# ==============================================================================

def create_summarization_agent(rag_retrieval_tool: rag.Retrieval,
                               generative_model_name: str) -> GenerativeModel:
    """
    RAG Retrieval Tool을 사용하여 문서 요약을 수행하는 GenerativeModel 에이전트를 생성합니다.
    (참고: "NotebookLM 스타일 AI Agent 개발.pdf", Chapter 13, p.139) [1]

    Args:
        rag_retrieval_tool: 설정된 RAG Retrieval 객체.
        generative_model_name: 사용할 생성 모델의 이름 (예: "text-bison@latest").

    Returns:
        요약 기능을 수행하는 GenerativeModel 인스턴스.
    """
    print(f"요약 에이전트 생성 중... 모델: {generative_model_name}")

    summary_rag_tool = Tool.from_retrieval(retrieval=rag_retrieval_tool) # Chapter 13, p.139 [1]

    agent = GenerativeModel(
        model_name="gemini-2.0-flash-001",#generative_model_name, # Chapter 13, p.139 [1]
        tools=[summary_rag_tool]
    )
    print("요약 에이전트 생성 완료.")
    return agent

# ==============================================================================
# 섹션 5: PDF 요약 워크플로우 (Chapter 13, 3단계 및 5단계) [1]
# ==============================================================================

def summarize_pdf_document(agent: GenerativeModel,
                           pdf_gcs_uri: str, # 문서 식별을 위해 사용
                           document_title: str, # 프롬프트에 사용될 수 있음
                           temperature: float,
                           max_output_tokens: int) -> str:
    """
    지정된 PDF 문서에 대해 요약 에이전트를 호출하여 요약을 생성합니다.
    프롬프트 설계는 Chapter 13, 3단계를 따르며, 에이전트 호출은 5단계를 따릅니다. [1]
    RAG Tool이 장착된 에이전트는 user_request를 기반으로 컨텍스트를 자동 검색합니다.

    Args:
        agent: 설정된 요약 에이전트 (GenerativeModel 인스턴스).
        pdf_gcs_uri: 요약할 PDF 파일의 GCS URI (에이전트가 특정 문서를 타겟팅하도록 유도).
        document_title: 문서 제목 (사용자 요청에 포함).
        temperature: 생성 시 사용할 온도 값.
        max_output_tokens: 생성될 요약의 최대 토큰 수.

    Returns:
        생성된 요약 텍스트.
    """
    print(f"\n--- 문서 요약 시작: '{document_title}' ({pdf_gcs_uri}) ---")

    # 사용자 요청 구성 (Chapter 13, p.140) [1]
    # 에이전트가 RAG Tool을 사용하여 특정 문서를 요약하도록 유도하는 요청.
    # Chapter 13의 "2023년 4분기 실적 보고서를 요약해줘." 와 유사한 형태에 문서 식별자 추가. [1]
    user_request = f"'{document_title}' 문서를 요약해 주십시오. (문서 위치: {pdf_gcs_uri})"
    # 또는 더 간단하게: user_request = f"'{document_title}' 문서를 요약해줘."
    # RAG Engine이 GCS URI를 직접 처리할 수 있는지 여부에 따라 프롬프트 조정 가능.
    # 여기서는 명시적으로 GCS URI를 포함하여 RAG Tool이 해당 문서를 찾도록 유도.

    print(f"요약 요청:\n{user_request}")

    generation_config = GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens
    )

    try:
        response = agent.generate_content(
            user_request, # Chapter 13, p.140 [1]
            generation_config=generation_config
        )
        summary_text = response.text
        print(f"\n생성된 요약:\n{summary_text}")
        return summary_text
    except Exception as e:
        print(f"문서 요약 중 오류 발생 ('{document_title}'): {e}")
        return "요약 생성에 실패했습니다."
    finally:
        print(f"--- 문서 요약 완료: '{document_title}' ---")

# ==============================================================================
# 메인 실행 블록
# ==============================================================================
if __name__ == "__main__":
    if not PROJECT_ID:
        raise ValueError("GOOGLE_CLOUD_PROJECT 환경 변수를 설정해야 합니다.")

    print("===== PDF 문서 임베딩 및 요약 프로세스 시작 (Chapter 13 기반) =====")

    rag_corpus_instance = get_or_create_rag_corpus(
        display_name=RAG_CORPUS_DISPLAY_NAME,
        embedding_model_uri=EMBEDDING_MODEL_PUBLISHER_MODEL
    )
    rag_corpus_resource_name = rag_corpus_instance.name

    pdf_file_uris_to_process = list_gcs_pdf_files(
        bucket_name=GCS_BUCKET_NAME,
        prefix=GCS_PDF_PREFIX
    )

    if not pdf_file_uris_to_process:
        print(f"GCS 버킷 '{GCS_BUCKET_NAME}' (접두사: '{GCS_PDF_PREFIX}')에서 처리할 PDF 파일을 찾지 못했습니다.")
    else:
        import_pdfs_to_corpus(
            rag_corpus_name=rag_corpus_resource_name,
            gcs_pdf_uris=pdf_file_uris_to_process,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        print("\nPDF 파일 임포트 요청이 완료되었습니다. 실제 인덱싱은 백그라운드에서 진행될 수 있습니다.")
        print("다음 단계로 진행하기 전에 인덱싱이 완료될 수 있도록 잠시 기다립니다...")
        # time.sleep(180) # 프로덕션에서는 LRO 완료 확인 필요

        rag_retrieval_instance = configure_rag_retrieval(
            rag_corpus_resource_name=rag_corpus_resource_name,
            top_k=RETRIEVAL_TOP_K
        )

        summarization_agent_instance = create_summarization_agent(
            rag_retrieval_tool=rag_retrieval_instance,
            generative_model_name=GENERATIVE_MODEL_NAME
        )

        print(f"\n===== 총 {len(pdf_file_uris_to_process)}개 PDF 문서에 대한 요약 생성 시작 =====")
        for pdf_uri in pdf_file_uris_to_process:
            doc_title = pdf_uri.split('/')[-1]
            summarize_pdf_document(
                agent=summarization_agent_instance,
                pdf_gcs_uri=pdf_uri,
                document_title=doc_title,
                temperature=SUMMARY_TEMPERATURE,
                max_output_tokens=SUMMARY_MAX_OUTPUT_TOKENS
            )
            print("-" * 70)

    print("\n===== PDF 문서 임베딩 및 요약 프로세스 종료 =====")