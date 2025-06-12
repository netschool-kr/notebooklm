from dotenv import load_dotenv
import os
import vertexai
from google.cloud import texttospeech
# --- 환경 설정
load_dotenv()
project = os.getenv("GOOGLE_CLOUD_PROJECT")
location = os.getenv("GOOGLE_CLOUD_LOCATION")
vertexai.init(project=project, location=location)

def synthesize_text_to_speech(text_to_synthesize, output_filename="output.mp3", language_code="ko-KR", voice_name="ko-KR-Neural2-A"):
    """
    주어진 텍스트를 음성으로 변환하여 오디오 파일로 저장합니다.

    Args:
        text_to_synthesize (str): 음성으로 변환할 텍스트입니다.
        output_filename (str): 저장할 오디오 파일의 이름입니다.
        language_code (str): 음성의 언어 코드입니다.
        voice_name (str): 사용할 음성의 이름입니다.
    """
    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(text=text_to_synthesize)

    voice_params = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        name=voice_name  # 예: "ko-KR-Neural2-A" (여성), "ko-KR-Neural2-C" (남성) [8]
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    try:
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice_params,
            audio_config=audio_config
        )

        with open(output_filename, "wb") as out:
            out.write(response.audio_content)
            print(f'Audio content written to file "{output_filename}"')
        return output_filename
    except Exception as e:
        print(f"Error during TTS synthesis: {e}")
        return None

# --- LLM 응답 예시 ---
llm_response_text = "안녕하세요. Vertex AI 기반 AI 에이전트입니다. 무엇을 도와드릴까요?"

# --- TTS 변환 실행 ---
# 실제 LLM 응답을 이 함수에 전달합니다.
generated_audio_file = synthesize_text_to_speech(llm_response_text, "llm_response_audio.mp3")

if generated_audio_file:
    print(f"Successfully generated audio: {generated_audio_file}")
