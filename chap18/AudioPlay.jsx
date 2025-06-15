// TTSControls.js (React)
import React, { useState, useEffect, useRef } from 'react';

function TTSControls({ textToSpeak, onPlaybackEnd }) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);
  const audioRef = useRef(null); // <audio> 요소 참조

  useEffect(() => {
    // textToSpeak 내용이 변경되면 새로운 오디오 URL을 백엔드에서 가져옴
    if (textToSpeak) {
      // fetchAudioUrlForText는 가상의 함수, 실제로는 TTS 서비스 호출
      // 예: const url = await ttsService.synthesize(textToSpeak);
      // setAudioUrl(url);
      // 여기서는 audio_play.html의 아이디어를 차용하여 URL을 직접 설정한다고 가정
      // setAudioUrl("YOUR_FETCHED_AUDIO_URL.mp3"); // 실제 URL로 교체
    }
  },);

  useEffect(() => {
    if (audioUrl && audioRef.current) {
      audioRef.current.src = audioUrl;
      audioRef.current.load();
      // 자동으로 재생하거나, 사용자가 재생 버튼을 누르도록 할 수 있음
    }
  }, [audioUrl]);

  const togglePlayPause = () => {
    if (!audioRef.current ||!audioRef.current.src) return;
    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play().catch(error => console.error("Audio play error:", error));
    }
    setIsPlaying(!isPlaying);
  };

  useEffect(() => {
    const audioElement = audioRef.current;
    const handleEnded = () => {
      setIsPlaying(false);
      if (onPlaybackEnd) onPlaybackEnd();
    };
    if (audioElement) {
      audioElement.addEventListener('ended', handleEnded);
      return () => {
        audioElement.removeEventListener('ended', handleEnded);
      };
    }
  }, [onPlaybackEnd]);


  return (
    <div>
      <audio ref={audioRef} />
      {audioUrl && (
        <button onClick={togglePlayPause}>
          {isPlaying? '일시정지' : '음성 듣기'}
        </button>
      )}
    </div>
  );
}
export default TTSControls;
