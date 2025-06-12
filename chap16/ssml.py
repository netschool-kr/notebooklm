from dotenv import load_dotenv
import os
import vertexai
from google.cloud import texttospeech
load_dotenv()
project = os.getenv("GOOGLE_CLOUD_PROJECT")
location = os.getenv("GOOGLE_CLOUD_LOCATION")
vertexai.init(project=project, location=location)

def synthesize_ssml_to_speech(ssml_text, output_filename="output_ssml.mp3", language_code="ko-KR", voice_name="ko-KR-Neural2-A"):
    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(ssml=ssml_text)

    voice_params = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        name=voice_name
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
            print(f'SSML audio content written to file "{output_filename}"')
        return output_filename
    except Exception as e:
        print(f"Error during SSML TTS synthesis: {e}")
        return None

# --- SSML 텍스트 예시 ---
ssml_example = """
<speak>
  안녕하세요. 지금부터 <say-as interpret-as="cardinal">3</say-as>가지 주요 사항을 안내해 드리겠습니다. <break time="1s"/>
  첫째, 오늘 날짜는 <say-as interpret-as="date" format="yyyymmdd" detail="1">2024년 7월 15일</say-as>입니다. <break time="500ms"/>
  둘째, 현재 시각은 <say-as interpret-as="time" format="hms12"> 오후 3시 20분</say-as>입니다. <break time="500ms"/>
  셋째, <prosody rate="slow" pitch="-1st">이 부분은 조금 천천히 그리고 약간 낮은 톤으로 전달합니다.</prosody> 감사합니다.
  <break time="1s"/> 이어서 영어로 <lang xml:lang="en-US">Hello, this is an example of multilingual speech.</lang>
</speak>
"""

# --- SSML TTS 변환 실행 ---
generated_ssml_audio_file = synthesize_ssml_to_speech(ssml_example, "llm_response_ssml_audio.mp3")

if generated_ssml_audio_file:
    print(f"Successfully generated SSML audio: {generated_ssml_audio_file}")
