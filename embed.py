import os
from dotenv import load_dotenv
from google import genai
from google.genai.types import EmbedContentConfig
from scipy.spatial.distance import cosine

load_dotenv()

project_id = os.environ["GOOGLE_CLOUD_PROJECT"]
project_location = os.environ["GOOGLE_CLOUD_LOCATION"]
print("project_id=", project_id, " : project_location=", project_location)
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"

client = genai.Client()

models = ["text-embedding-005", "text-multilingual-embedding-002"]#, "gemini-embedding-001"]
sentence = "What is the role of Vertex AI in document search?"

# 각 모델로부터 임베딩 벡터 생성
embeddings = {}
for model in models:
    response = client.models.embed_content(model=model, contents=sentence, config=EmbedContentConfig())
    vector = response.embeddings[0].values  # 임베딩 벡터 값 (첫 번째 결과)
    embeddings[model] = vector

# 벡터 차원(길이) 출력
for model, vector in embeddings.items():
    print(f"{model} 벡터 길이: {len(vector)}")

# 모델 간 임베딩 벡터 코사인 유사도 계산
for i in range(len(models)):
    for j in range(i+1, len(models)):
        v1 = embeddings[models[i]]
        v2 = embeddings[models[j]]
        similarity = 1 - cosine(v1, v2)
        print(f"{models[i]} vs {models[j]} 코사인 유사도: {similarity:.3f}")
