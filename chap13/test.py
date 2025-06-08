
from vertexai.preview.reasoning_engines import AdkApp

# (생략) 모델·툴 정의 후
app = AdkApp(agent=my_agent)
session = app.create_session(user_id="USER1")
