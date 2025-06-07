from dotenv import load_dotenv
import os
import vertexai
from vertexai.preview.language_models import TextEmbeddingModel, TextEmbeddingInput
from vertexai.generative_models import GenerativeModel 
from google.cloud import bigquery

def main():
    """
    메인 실행 함수
    """
    # --- 환경 설정 (변경 없음) ---
    load_dotenv()
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION")
    vertexai.init(project=project, location=location)

    bq_client = bigquery.Client(project=project)
    DATASET_ID = "book_data"
    TABLE_ID   = "embeddings"
    TABLE_NAME = f"{project}.{DATASET_ID}.{TABLE_ID}"

    # --- 테이블 스키마 검증 (변경 없음) ---
    print("--- BigQuery 테이블 스키마 검증 ---")
    try:
        table = bq_client.get_table(TABLE_NAME)
        schema_fields = {field.name: field.field_type for field in table.schema}
        
        print(f"테이블 스키마: {schema_fields}")
        text_column_name = next((col for col, type in schema_fields.items() if type == 'STRING' and col != 'id'), None)
        
        if not all(k in schema_fields for k in ["id", "embedding"]) or not text_column_name:
            raise ValueError("필수 컬럼(id, embedding, content 등)이 누락되었습니다.")

        print(f"검증 완료: id, embedding, 텍스트 컬럼('{text_column_name}') 확인")
        print("-------------------------------------\n")
    except Exception as e:
        print(f"\n스키마 검증 오류: {e}\n'embed_store.py'를 실행했는지 확인하세요.")
        return

    # --- 사용자 질문 및 임베딩 생성 (변경 없음) ---
    user_question = input("질문을 입력하세요: ")
    if not user_question:
        print("질문이 입력되지 않았습니다.")
        return

    embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    embedding_input = TextEmbeddingInput(task_type="RETRIEVAL_QUERY", text=user_question)
    question_embedding = embedding_model.get_embeddings([embedding_input])[0].values

    # --- BigQuery 벡터 검색 (변경 없음) ---
    top_k = 3
    sql = f"""
        SELECT id, {text_column_name},
               (1 - ML.DISTANCE(embedding, @query_vector, 'COSINE')) AS cosine_sim
        FROM `{TABLE_NAME}`
        ORDER BY cosine_sim DESC
        LIMIT {top_k}
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("query_vector", "FLOAT64", question_embedding)]
    )
    try:
        results = list(bq_client.query(sql, job_config=job_config).result())
    except Exception as e:
        print(f"\n--- 쿼리 실행 중 오류 발생 ---\n오류: {e}")
        return

    # --- 5. 검색 결과로 LLM 프롬프트 구성 및 답변 생성 ---
    retrieved_texts = [getattr(row, text_column_name) for row in results]
    context_prompt = "\n\n".join(retrieved_texts) if retrieved_texts else "(관련 문서를 찾지 못했습니다.)"
    prompt = f"""다음 문서를 참고하여 질문에 답하세요:\n\n{context_prompt}\n\n질문: {user_question}\n답변:"""

    text_model = GenerativeModel("gemini-2.0-flash-lite-001") # 모델명을 최신으로 수정

    # --- 3. 답변 생성 함수 수정 ---
    # .predict() 대신 .generate_content()를 사용합니다.
    response = text_model.generate_content(prompt)
    answer = response.text

    print("\n[생성된 답변]")
    print(answer)

if __name__ == "__main__":
    main()