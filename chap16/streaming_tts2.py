from dotenv import load_dotenv
import os
import vertexai
from google.cloud import texttospeech
import pyaudio # 스피커 출력을 위해 추가

# --- 환경 설정
load_dotenv()
project = os.getenv("GOOGLE_CLOUD_PROJECT")
location = os.getenv("GOOGLE_CLOUD_LOCATION")
vertexai.init(project=project, location=location)


def generate_tts_audio_stream(
    text_iterator,
    voice_name="en-US-Chirp3-HD-Charon",
    language_code="en-US"
):
    """
    텍스트 청크 스트림에서 TTS 오디오 스트림을 생성합니다.
    """
    client = texttospeech.TextToSpeechClient()

    streaming_config = texttospeech.StreamingSynthesizeConfig(
        voice=texttospeech.VoiceSelectionParams(
            name=voice_name,
            language_code=language_code,
        )
    )

    def request_generator():
        # 첫 번째 요청에는 반드시 구성 정보가 포함되어야 합니다.
        yield texttospeech.StreamingSynthesizeRequest(
            streaming_config=streaming_config
        )

        # 이후 요청들은 텍스트 청크를 포함합니다.
        for text_chunk in text_iterator:
            if text_chunk: # 비어있지 않은 텍스트인지 확인
                yield texttospeech.StreamingSynthesizeRequest(
                    input=texttospeech.StreamingSynthesisInput(text=text_chunk)
                )

    requests = request_generator()
    streaming_responses = client.streaming_synthesize(requests=requests)

    for response in streaming_responses:
        if response.audio_content:
            yield response.audio_content

if __name__ == "__main__":
    print("Starting Text-to-Speech streaming demo...")

    example_text_chunks = [
        "Hello there. ",
        "How are you ",
        "today? It's ",
        "a beautiful day to test streaming audio.",
    ]

    # --- PyAudio 설정 ---
    # Google Cloud Chirp 음성은 일반적으로 24000Hz 샘플링 레이트를 사용합니다.
    SAMPLE_RATE = 24000
    # 오디오 포맷은 16-bit PCM 입니다.
    FORMAT = pyaudio.paInt16
    # 채널은 1 (모노) 입니다.
    CHANNELS = 1

    p = pyaudio.PyAudio()
    stream = None

    try:
        # PyAudio 스트림을 엽니다.
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            output=True
        )

        print("🔊 Streaming audio to speaker...")

        # TTS 오디오 스트림을 받아옵니다.
        audio_stream = generate_tts_audio_stream(iter(example_text_chunks))

        # 오디오 청크를 받아 스트림에 직접 씁니다 (재생).
        for i, audio_chunk in enumerate(audio_stream):
            print(f"Playing audio chunk {i+1}...")
            stream.write(audio_chunk)

        print("✅ Finished playing audio.")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # 스트림과 PyAudio 인스턴스를 안전하게 종료합니다.
        if stream is not None:
            stream.stop_stream()
            stream.close()
        p.terminate()
        print("Streaming demo finished.")