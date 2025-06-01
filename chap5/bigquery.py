import os
import numpy as np
from google import genai
from dotenv import load_dotenv
from google.cloud import bigquery
# .env 파일에서 GCP 설정 로드
load_dotenv()
project_id = os.environ["GOOGLE_CLOUD_PROJECT"]
project_location = os.environ["GOOGLE_CLOUD_LOCATION"]
print("project_id=", project_id, " : project_location=", project_location)

# Vertex AI GenAI 사용 설정
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"

# genai 클라이언트 초기화
client = genai.Client()

# BigQuery 클라이언트 초기화
bq_client = bigquery.Client()
dataset_id = f"{os.environ['GOOGLE_CLOUD_PROJECT']}.book_data"
table_id = f"{dataset_id}.embeddings"

# 벡터 유사도 검색 예시
query_text = "example query"
query_response = client.models.embed_content(model="text-multilingual-embedding-002", contents=query_text)
query_vec = np.array(query_response.embeddings[0].values)

# BigQuery에서 모든 벡터 불러오기
query_job = bq_client.query(f"SELECT id, embedding FROM `{table_id}`")
rows = query_job.result()
data = [(row["id"], np.array(row["embedding"])) for row in rows]

# 코사인 유사도 계산
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

sims = [(item[0], cosine_similarity(query_vec, item[1])) for item in data]
# 유사도 상위 3개 선택
top3 = sorted(sims, key=lambda x: x[1], reverse=True)[:3]
print("유사도 상위 3개 문서:", top3)
