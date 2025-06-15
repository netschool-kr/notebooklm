import React, { useState } from 'react';

function DocumentUploader({ onUploadSuccess }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files);
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    setIsUploading(true);
    const formData = new FormData();
    formData.append('document', selectedFile);

    try {
      // 'YOUR_BACKEND_UPLOAD_ENDPOINT'는 실제 백엔드 API 엔드포인트로 대체
      const response = await fetch('YOUR_BACKEND_UPLOAD_ENDPOINT', {
        method: 'POST',
        body: formData,
        // 필요한 경우 헤더 추가 (예: 인증 토큰)
      });
      if (response.ok) {
        const result = await response.json();
        onUploadSuccess(result); // 업로드 성공 시 상위 컴포넌트에 알림
        alert('문서 업로드 성공!');
      } else {
        alert('문서 업로드 실패.');
      }
    } catch (error) {
      console.error('Upload error:', error);
      alert('업로드 중 오류 발생.');
    } finally {
      setIsUploading(false);
      setSelectedFile(null);
    }
  };

  return (
    <div>
      <input type="file" onChange={handleFileChange} disabled={isUploading} />
      <button onClick={handleUpload} disabled={!selectedFile || isUploading}>
        {isUploading? '업로드 중...' : '업로드'}
      </button>
    </div>
  );
}
export default DocumentUploader;
