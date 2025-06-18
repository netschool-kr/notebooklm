# services/visualization_service.py
import os
import io
import base64
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정 (시스템에 맞는 폰트 경로 지정 필요)
# Cloud Run과 같은 환경에서는 폰트 파일을 함께 배포해야 한다.
try:
    # 예시: NanumGothic 폰트 사용
    font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    if os.path.exists(font_path):
        font_prop = fm.FontProperties(fname=font_path)
        plt.rc('font', family=font_prop.get_name())
        plt.rc('axes', unicode_minus=False) # 마이너스 기호 깨짐 방지
        print("한글 폰트가 성공적으로 설정되었습니다.")
    else:
        print("경고: 지정된 한글 폰트 파일을 찾을 수 없습니다. 기본 폰트가 사용됩니다.")
except Exception as e:
    print(f"경고: 한글 폰트 설정 중 오류 발생: {e}")


def generate_chart_base64(data_json: str, chart_type: str, x_col: str, y_col: str, title: str) -> str:
    """
    주어진 JSON 데이터로 차트를 생성하고 Base64 인코딩된 이미지 문자열을 반환한다.
    Chapter 17의 generate_chart_tool 함수를 서비스 형태로 구현.[1]
    """
    print(f"차트 생성 요청: type={chart_type}, title='{title}'")
    try:
        df = pd.read_json(io.StringIO(data_json))

        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError(f"지정된 컬럼({x_col}, {y_col})이 데이터에 없습니다.")

        fig, ax = plt.subplots(figsize=(10, 6))

        if chart_type == 'bar':
            ax.bar(df[x_col], df[y_col])
        elif chart_type == 'line':
            ax.plot(df[x_col], df[y_col], marker='o')
        elif chart_type == 'pie':
            ax.pie(df[y_col], labels=df[x_col], autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # 파이 차트를 원형으로 유지
        else:
            raise ValueError(f"지원하지 않는 차트 종류입니다: {chart_type}")

        ax.set_title(title, fontsize=16)
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()

        # 차트를 인메모리 버퍼에 PNG 형식으로 저장
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)  # 메모리 누수 방지를 위해 figure 객체를 명시적으로 닫음
        buf.seek(0)

        # 버퍼 내용을 Base64로 인코딩하여 데이터 URI 생성
        image_bytes = buf.getvalue()
        base64_encoded_string = base64.b64encode(image_bytes).decode('utf-8')
        print("차트 생성 및 Base64 인코딩 완료.")
        return f"data:image/png;base64,{base64_encoded_string}"

    except Exception as e:
        print(f"차트 생성 실패: {e}")
        plt.close('all') # 오류 발생 시 모든 figure 닫기
        raise