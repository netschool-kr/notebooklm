# services/audio_service.py

import os
import uuid
import datetime
from google.cloud import texttospeech, storage

class AudioService:
    """
    Google Cloud TTS 및 GCS 연동을 통한 오디오 생성을 담당하는 서비스.
    """
    def __init__(self, project_id: str, bucket_name: str):
        if not project_id or not bucket_name:
            raise ValueError("GCP 프로젝트 ID와 GCS 버킷 이름은 필수입니다.")
            
        self.tts_client = texttospeech.TextToSpeechClient()
        self.storage_client = storage.Client(project=project_id)
        self.bucket_name = bucket_name
        self.bucket = self.storage_client.bucket(self.bucket_name)
        print(f"AudioService 초기화 완료. GCS 버킷: '{bucket_name}'")

    def synthesize_and_get_signed_url(self, text: str, language_code="ko-KR", voice_name="ko-KR-Neural2-A") -> str:
        """
        텍스트를 음성으로 합성하여 GCS에 업로드하고, 서명된 URL을 반환한다.
        Chapter 14, 19의 로직을 서비스 형태로 구현.[1]
        """
        print(f"오디오 합성 요청: '{text[:50]}...'")
        try:
            synthesis_input = texttospeech.SynthesisInput(text=text)

            voice = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=voice_name
            )

            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )

            response = self.tts_client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )

            # GCS에 업로드할 고유한 파일 이름 생성
            object_name = f"tts-audio/{uuid.uuid4()}.mp3"
            blob = self.bucket.blob(object_name)

            # 메모리에서 직접 GCS로 업로드
            blob.upload_from_string(response.audio_content, content_type="audio/mpeg")
            print(f"오디오 파일 GCS에 업로드 완료: gs://{self.bucket_name}/{object_name}")

            # 5분 동안 유효한 서명된 URL 생성
            signed_url = blob.generate_signed_url(
                version="v4",
                expiration=datetime.timedelta(minutes=5),
                method="GET",
            )
            print("GCS 서명된 URL 생성 완료.")
            return signed_url

        except Exception as e:
            print(f"오디오 합성 또는 업로드 실패: {e}")
            raise