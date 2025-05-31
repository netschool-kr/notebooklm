# test_rag_retrieval_query_inspect.py

import os
from dotenv import load_dotenv
from google.cloud import aiplatform
from vertexai.preview import rag
import logging

logging.basicConfig(level=logging.DEBUG)
load_dotenv()

# (1) 환경 변수 확인
project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
project_location = os.getenv("GOOGLE_CLOUD_LOCATION")
print(f"★ project_id = {project_id}, project_location = {project_location}")

# (2) Vertex AI 초기화
try:
    aiplatform.init(project=project_id, location=project_location)
    print("Vertex AI 초기화 성공")
except Exception as e:
    print(f"Vertex AI 초기화 실패: {e}")

# (3) 임베딩 모델 및 RAG corpus 생성 (이미 존재할 경우 예외 처리)
try:
    embedding_model = rag.RagEmbeddingModelConfig(
        vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
            publisher_model="publishers/google/models/text-embedding-005"
        )
    )

    corpus = rag.create_corpus(
        display_name="notebooklm_corpus_test_query_inspect",
        backend_config=rag.RagVectorDbConfig(rag_embedding_model_config=embedding_model)
    )
    print(f"Corpus 생성됨: {corpus.name}")
except Exception as e:
    print(f"Corpus 생성 실패 또는 이미 존재: {e}")

# (4) RAG 문서 임포트 (실제 경로 대신 더미 URI 사용)
try:
    rag.import_files(
        corpus.name,
        paths=["gs://notebook-docs/"],  # 실제 경로 대신 더미 사용
        transformation_config=rag.TransformationConfig(
            chunking_config=rag.ChunkingConfig(chunk_size=512, chunk_overlap=50)
        )
    )
    print("import_files 호출 성공")
except Exception as e:
    print(f"import_files 호출 중 오류: {e}")

# (5) Retrieval 도구 생성 (선택 사항)
try:
    retrieval = rag.Retrieval(
        source=rag.VertexRagStore(
            rag_resources=[rag.RagResource(rag_corpus=corpus.name)],
            rag_retrieval_config=rag.RagRetrievalConfig(top_k=3)
        )
    )
    print("Retrieval 객체 생성됨")
except Exception as e:
    print(f"Retrieval 객체 생성 실패: {e}")
    retrieval = None

# (6) retrieve 메서드 여부 확인 (디버깅 목적으로 출력)
if retrieval:
    available_methods = [m for m in dir(retrieval) if not m.startswith("_")]
    print("Retrieval 객체의 가능한 메서드 목록:")
    for method in available_methods:
        print(f"  - {method}")
    print()

# (7) retrieval_query 호출 및 introspection
user_question = "Vertex AI Agent Builder의 역할은 무엇인가요?"

try:
    print("▶ retrieval_query 메서드 호출 시도...")
    response = rag.retrieval_query(
        rag_resources=[
            rag.RagResource(
                rag_corpus=corpus.name,
                # rag_file_ids 를 지정하려면 여기에 리스트로 추가
            )
        ],
        text=user_question,
        rag_retrieval_config=rag.RagRetrievalConfig(top_k=3),
    )

    # 1) response.contexts 객체를 직접 살펴봅니다.
    contexts_obj = response.contexts
    print("\n--- response.contexts 타입 및 속성 확인 ---")
    print(f"type(response.contexts) = {type(contexts_obj)}")
    print("dir(response.contexts) =")
    for attr in dir(contexts_obj):
        if not attr.startswith("_"):
            print(f"  - {attr}")
    print("------------------------------------------\n")

    # 2) 반복 가능한 필드를 찾아 시도해 봅니다.
    #    예를 들어, contexts_obj 안에 'contexts' 또는 'results' 같은 속성이 있을 수 있습니다.
    possible_lists = []
    for candidate in ["contexts", "results", "items", "elements"]:
        if hasattr(contexts_obj, candidate):
            possible_lists.append(candidate)

    if possible_lists:
        print(f"반복 가능한 속성 후보: {possible_lists}\n")
        # 첫 번째 후보로 반복을 시도해 봅니다.
        first_candidate = possible_lists[0]
        iterable = getattr(contexts_obj, first_candidate)
        print(f"▶ '{first_candidate}' 속성으로 반복 시도:")
        try:
            for i, ctx in enumerate(iterable, start=1):
                # ctx 객체 자체도 어떤 속성을 가지는지 확인해 봅니다.
                print(f"  [Context {i}] type: {type(ctx)}")
                # 일반적으로 ctx에는 source_uri, text 등이 있습니다.
                if hasattr(ctx, "source_uri"):
                    print(f"    source_uri: {ctx.source_uri}")
                if hasattr(ctx, "text"):
                    print(f"    text: {ctx.text}")
                # 만약 다른 속성이 있으면 추가로 출력
                extras = [x for x in dir(ctx) if not x.startswith("_") and x not in ["source_uri", "text"]]
                if extras:
                    print(f"    (추가 속성: {extras})")
        except Exception as e:
            print(f"  ▶ 반복 중 예외 발생: {e}")
    else:
        print("반복 가능한 속성을 찾지 못했습니다. 수동으로 속성 확인이 필요합니다.")

except AttributeError as ae:
    print("★ AttributeError 발생:")
    print(f"  {ae}")
    print("  👉 'retrieval_query' 메서드가 없는 것 같습니다.")
except TypeError as te:
    print("★ TypeError 발생:")
    print(f"  {te}")
    print("  👉 'retrieval_query' 호출 시 인자가 잘못되었을 수 있습니다.")
except Exception as e:
    print("★ 기타 예외 발생:")
    print(f"  {e}")
