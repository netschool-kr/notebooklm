import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
from io import StringIO

def generate_chart_tool(table_data_json: str, chart_type: str, x_column: str, y_column: str, title: str = None) -> str:
    """
    주어진 표 데이터와 파라미터를 사용하여 차트를 생성하고 Base64 인코딩된 이미지 문자열을 반환합니다.
    Args:
        table_data_json (str): 표 데이터를 나타내는 JSON 문자열 (Pandas DataFrame으로 변환 가능해야 함).
        chart_type (str): 'bar', 'pie', 'line' 중 하나.
        x_column (str): X축으로 사용할 컬럼 이름.
        y_column (str): Y축으로 사용할 컬럼 이름.
        title (str, optional): 차트 제목.

    Returns:
        str: Base64 인코딩된 PNG 이미지 문자열. 오류 발생 시 에러 메시지 반환.
    """
    try:
        # StringIO를 사용하여 JSON 문자열을 파일처럼 읽도록 처리
        df = pd.read_json(StringIO(table_data_json))

        if x_column not in df.columns or y_column not in df.columns:
            return "Error: Specified x_column or y_column not found in data."

        plt.figure(figsize=(8, 6))

        if chart_type == 'bar':
            plt.bar(df[x_column], df[y_column])
            plt.xlabel(x_column)
            plt.ylabel(y_column)
        elif chart_type == 'pie':
            if not pd.api.types.is_numeric_dtype(df[y_column]):
                 return "Error: Pie chart y_column must be numeric."
            plt.pie(df[y_column], labels=df[x_column], autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
        elif chart_type == 'line':
            plt.plot(df[x_column], df[y_column], marker='o')
            plt.xlabel(x_column)
            plt.ylabel(y_column)
        else:
            return "Error: Unsupported chart_type. Choose 'bar', 'pie', or 'line'."

        if title:
            plt.title(title)
        else:
            plt.title(f"{y_column} by {x_column} ({chart_type.capitalize()} Chart)")

        plt.grid(True)
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        plt.close()
        img_buffer.seek(0)
        
        base64_image = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        img_buffer.close()
        
        return base64_image

    except Exception as e:
        plt.close() # 오류 발생 시 플롯 닫기
        return f"Error generating chart: {str(e)}"

# # 예시 사용법:
# 비어있던 문자열을 실제 데이터가 있는 JSON 형식의 문자열로 수정
sample_json_data = '''
[
  {"Month": "January", "Sales": 65000},
  {"Month": "February", "Sales": 59000},
  {"Month": "March", "Sales": 80000},
  {"Month": "April", "Sales": 81000},
  {"Month": "May", "Sales": 56000},
  {"Month": "June", "Sales": 55000}
]
'''
b64_img = generate_chart_tool(table_data_json=sample_json_data, chart_type='line', x_column='Month', y_column='Sales', title='Monthly Sales')

if not b64_img.startswith("Error:"):
    print(f"Generated Base64 Image (first 100 chars): {b64_img[:100]}...")
else:
    print(b64_img)