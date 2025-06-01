from dotenv import load_dotenv
import os
import numpy as np
from google.cloud import bigquery
from google import genai
from google.genai.types import EmbedContentConfig

# 환경변수 로드
load_dotenv()
project = os.getenv("GOOGLE_CLOUD_PROJECT")
location = os.getenv("GOOGLE_CLOUD_LOCATION")
# Vertex AI 사용 설정 (Generative AI SDK)
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
# BigQuery 클라이언트 초기화
bq_client = bigquery.Client(project=project)
# 데이터셋 및 테이블 ID 설정 (예: book_data.ㅈ)
dataset_id = "book_data"              # 실제 데이터셋명으로 수정 필요
table_id = "embeddings"         # 실제 테이블명으로 수정 필요
full_table_id = f"{project}.{dataset_id}.{table_id}"

# 검색할 질문 목록
queries = ["Rag는 뭐지", "Agent Builder 기능 설명해줘", "text embeding이 뭐지?"]

# Google GenAI 클라이언트 (Vertex AI 임베딩 모델)
embed_client = genai.Client()

for query in queries:
    # 텍스트 임베딩 생성
    response = embed_client.models.embed_content(
        model="text-multilingual-embedding-002",
        contents=query,
        config=EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    # 임베딩 벡터 추출
    embedding = list(response.embeddings[0].values)
    
    # BigQuery에서 코사인 유사도로 상위 3개 문서 조회
    # Corrected SQL query
    sql = f"""
        SELECT
            id,
            (1 - ML.DISTANCE(embedding, @query_vector, 'COSINE')) AS cosine_sim
        FROM `{full_table_id}`
        ORDER BY cosine_sim DESC
        LIMIT 3
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("query_vector", "FLOAT64", embedding)
        ]
    )
    query_job = bq_client.query(sql, job_config=job_config)
    results = query_job.result()
    
    # 결과 출력
    print(f"\n질문: {query}")
    for row in results:
        print(f"ID: {row.id}, 유사도: {row.cosine_sim:.6f}")
