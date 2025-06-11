import pandas as pd  # 1. 스크립트 상단에 pandas 추가
import vertexai
from vertexai.preview.evaluation import EvalTask
from vertexai.preview.evaluation import PointwiseMetric
import os
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
vertexai.init(project=PROJECT_ID, location=LOCATION)

# 1. 충실도 (Faithfulness)
# 요약이 원문 내용을 정확하게 반영하고, 원문에 없거나 모순되는 정보를 포함하지 않는지 평가
faithfulness_metric_prompt_template = """
제공된 원본 문서와 생성된 요약을 비교하여 요약의 충실도를 평가해 주십시오. 
충실한 요약은 원본 문서의 정보를 정확하게 반영하며, 원본에 없거나 모순되는 정보를 포함하지 않아야 합니다.

원본 문서:
{document_text}

생성된 요약:
{generated_summary}

다음 척도에 따라 충실도 점수를 1점에서 5점 사이로 매겨주십시오 (5점이 가장 충실함).
점수와 함께 간략한 평가 이유를 설명해 주십시오.
점수:
이유:
"""
faithfulness_metric = PointwiseMetric(
    metric="faithfulness", # 지표 이름
    metric_prompt_template=faithfulness_metric_prompt_template
    # system_instruction을 추가하여 심사 모델에 특정 지침을 줄 수도 있음 [6]
)

# 2. 관련성 (Relevance)
# 요약이 원문의 주요 주제 및 핵심 아이디어와 얼마나 관련이 있는지 평가
relevance_metric_prompt_template = """
생성된 요약이 원본 문서의 주요 주제 및 핵심 아이디어와 얼마나 관련성이 높은지 평가해 주십시오.
관련성이 높은 요약은 원문의 핵심 내용을 중심으로 작성되며, 주제에서 벗어나는 내용을 포함하지 않습니다.

원본 문서:
{document_text}

생성된 요약:
{generated_summary}

다음 척도에 따라 관련성 점수를 1점에서 5점 사이로 매겨주십시오 (5점이 가장 관련성 높음).
점수와 함께 간략한 평가 이유를 설명해 주십시오.
점수:
이유:
"""
relevance_metric = PointwiseMetric(
    metric="relevance",
    metric_prompt_template=relevance_metric_prompt_template
)

# 3. 일관성/유창성 (Coherence/Fluency)
# 요약문 자체의 문법적 정확성, 가독성, 논리적 흐름을 평가
coherence_metric_prompt_template = """
생성된 요약의 문법적 정확성, 가독성, 그리고 내용의 논리적 흐름(일관성)을 평가해 주십시오.
품질 좋은 요약은 문법적으로 정확하고, 읽기 쉬우며, 아이디어가 자연스럽게 연결되어야 합니다.

생성된 요약:
{generated_summary}

다음 척도에 따라 일관성/유창성 점수를 1점에서 5점 사이로 매겨주십시오 (5점이 가장 우수함).
점수와 함께 간략한 평가 이유를 설명해 주십시오.
점수:
이유:
"""
coherence_metric = PointwiseMetric(
    metric="coherence",
    metric_prompt_template=coherence_metric_prompt_template
    # 이 지표는 원문(document_text)을 반드시 필요로 하지는 않을 수 있으나,
    # 평가 데이터셋에는 일관성을 위해 포함하는 것이 좋음.
)

# 4. 간결성 (Conciseness)
# 요약이 불필요한 반복이나 장황함 없이 핵심 정보를 효과적으로 전달하는지 평가
conciseness_metric_prompt_template = """
생성된 요약이 원본 문서의 핵심 정보를 얼마나 간결하게 전달하는지 평가해 주십시오.
간결한 요약은 불필요한 반복이나 세부 사항 없이 주요 내용만을 명확하게 압축합니다.

원본 문서:
{document_text}

생성된 요약:
{generated_summary}

다음 척도에 따라 간결성 점수를 1점에서 5점 사이로 매겨주십시오 (5점이 가장 간결함).
점수와 함께 간략한 평가 이유를 설명해 주십시오.
점수:
이유:
"""
conciseness_metric = PointwiseMetric(
    metric="conciseness",
    metric_prompt_template=conciseness_metric_prompt_template
)

# 5. 전반적인 요약 품질 (Overall Summarization Quality)
# 위의 개별 기준들을 종합적으로 고려한 요약의 전반적인 유용성 및 품질 평가
summarization_quality_metric_prompt_template = """
제공된 원본 문서와 생성된 요약을 바탕으로, 요약의 전반적인 품질을 평가해 주십시오.
좋은 요약은 원문의 핵심 내용을 충실하고, 관련성 높고, 일관성 있게, 그리고 간결하게 전달해야 합니다.

원본 문서:
{document_text}

생성된 요약:
{generated_summary}

다음 척도에 따라 전반적인 요약 품질 점수를 1점에서 5점 사이로 매겨주십시오 (5점이 가장 우수함).
점수와 함께 간략한 평가 이유를 설명해 주십시오.
점수:
이유:
"""
summarization_quality_metric = PointwiseMetric(
    metric="summarization_quality",
    metric_prompt_template=summarization_quality_metric_prompt_template
)

from vertexai.preview.evaluation import EvalTask

original_document_content="""
Vertex AI는 Google Cloud에서 제공하는 통합 머신러닝 플랫폼입니다. 
데이터 준비부터 모델 학습, 배포, 관리에 이르기까지 MLOps의 전체 수명 주기를 지원합니다. 
특히, 최근에는 Gemini와 같은 강력한 대규모 언어 모델(LLM)을 활용한 생성형 AI 애플리케이션 개발 기능이 강화되었습니다. 
사용자는 Vertex AI Studio를 통해 코딩 없이 LLM을 실험하거나, Python SDK를 사용하여 프로그래매틱하게 모델을 제어할 수 있습니다. 
또한, RAG(Retrieval-Augmented Generation) 아키텍처를 쉽게 구현할 수 있도록 Vector Search, RAG Engine 등의 도구를 제공하여, 
기업 내부 데이터를 LLM과 안전하게 연동하여 환각을 줄이고 신뢰성 높은 답변을 생성하도록 돕습니다.
"""
baseline_summary_text="""
Vertex AI는 Google Cloud의 통합 머신러닝 플랫폼으로, MLOps 전반을 지원하며 특히 Gemini 기반  생성형 AI 개발 기능이 강화되었습니다. Vertex AI Studio, Python SDK, Vector Search, RAG Engine 등을 통해 LLM 실험, 제어, RAG 아키텍처 구현을 용이하게 하여 기업 데이터 연동 및 신뢰성 높은 답변 생성을 돕습니다.
"""
improved_summary_text="""
Vertex AI는 Google Cloud의 머신러닝 플랫폼으로, 모델 개발부터 배포까지 전 과정을 지원합니다. 특히 Gemini와 같은 LLM을 활용한 생성형 AI 개발을 쉽게 할 수 있도록 다양한 도구를 제공합니다. RAG 아키텍처를 위한 Vector Search와 RAG Engine을 통해 기업 데이터와 LLM을 연동하여 환각 현상 을 줄이고 신뢰도 높은 답변을 얻을 수 있습니다.
"""
# 평가 데이터셋 구성
# 여기서는 앞서 생성한 baseline_summary_text와 improved_summary_text를 평가 대상으로 함
EVAL_DATASET = pd.DataFrame([
    {
        "document_text": original_document_content,
        "generated_summary": baseline_summary_text,
        "candidate_model_name": "baseline_summary" # 기준 요약 식별자
    },
    {
        "document_text": original_document_content, # 동일 원문 사용
        "generated_summary": improved_summary_text,
        "candidate_model_name": "improved_prompt_summary" # 개선된 프롬프트 요약 식별자
    }
])

# 평가할 지표 리스트
metrics_to_evaluate = [
    faithfulness_metric,
    relevance_metric,
    coherence_metric,
    conciseness_metric,
    summarization_quality_metric
]

# EvalTask 인스턴스 생성
# autorater_config를 통해 심사 모델의 상세 설정을 제어할 수 있음 [6]
eval_task = EvalTask(
    dataset=EVAL_DATASET,
    metrics=metrics_to_evaluate
)

# 평가 실행
try:
    eval_result = eval_task.evaluate()
    print("\n----- 평가 결과 -----")

    # 요약된 지표 결과 출력 (평균 점수 등)
    summary_table = getattr(eval_result, 'summary_metrics_table', None)
    if summary_table is not None and not summary_table.empty:
        print("\n=== 요약 지표 테이블 ===")
        print(summary_table)
    else:
        print("\n요약 지표를 계산하지 못했습니다 (API 오류 또는 데이터 부족).")

    # 개별 인스턴스별 상세 지표 결과 출력
    metrics_table = getattr(eval_result, 'metrics_table', None)
    if metrics_table is not None and not metrics_table.empty:
        print("\n=== 상세 지표 테이블 ===")
        for index, row in metrics_table.iterrows():
            print(f"\n--- 평가 대상: {row.get('candidate_model_name', 'N/A')} ---")
            print(f"문서: {row.get('document_text', '')[:100]}...")
            print(f"요약: {row.get('generated_summary', '')[:100]}...")
            for metric in metrics_to_evaluate:
                # [최종 수정] 올바른 내부 속성인 metric._metric_name 으로 변경
                metric_name = metric.metric_name
                
                # API 오류로 특정 지표가 없을 수 있으므로 .get()으로 안전하게 접근
                score = row.get(f"{metric_name}/score", "오류/계산불가")
                explanation = row.get(f"{metric_name}/explanation", "오류/계산불가")
                print(f"  {metric_name}: 점수 = {score}, 이유 = {explanation}")
    else:
        print("\n상세 평가 결과를 찾을 수 없습니다.")

except Exception as e:
    print(f"평가 실행 중 예상치 못한 오류 발생: {e}")
    eval_result = None
