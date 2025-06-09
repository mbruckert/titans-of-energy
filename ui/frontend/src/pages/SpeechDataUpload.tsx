import React, { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import UploadProgress from '../components/UploadProgress';

const SpeechDataUpload = () => {
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    setFile(droppedFile);
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
    }
  };

  const handleSubmit = () => {
    // Handle file upload logic here
    console.log('Uploading file:', file);
    // Navigate to next step
    navigate('/style-upload');
  };

  return (
    <div className="container mx-auto p-4 max-w-2xl">
      <div className="text-center mb-8">
        <h1 className="text-2xl font-bold mb-2">Upload Speech Data</h1>
      </div>

      <div 
        className="border-2 border-dashed border-gray-300 rounded-lg p-8 mb-4 text-center"
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
      >
        <div className="mb-4">
          <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
          </svg>
        </div>
        <p className="text-gray-600 mb-2">Select your file or drag and drop</p>
        <input
          type="file"
          id="file-upload"
          className="hidden"
          onChange={handleFileSelect}
        />
        <button
          onClick={() => document.getElementById('file-upload')?.click()}
          className="bg-black text-white px-4 py-2 rounded-md hover:bg-gray-800"
        >
          Browse
        </button>
        {file && (
          <p className="mt-2 text-sm text-gray-600">
            Selected file: {file.name}
          </p>
        )}
      </div>

      <div className="flex justify-end space-x-2">
        <button
          onClick={() => navigate('/tts-upload')}
          className="px-4 py-2 border border-gray-300 rounded-md hover:bg-gray-50"
        >
          Back
        </button>
        <button
          onClick={handleSubmit}
          disabled={!file}
          className={`px-4 py-2 rounded-md ${
            file ? 'bg-black text-white hover:bg-gray-800' : 'bg-gray-300 text-gray-500 cursor-not-allowed'
          }`}
        >
          Upload
        </button>
      </div>

      <UploadProgress currentStep={3} />
    </div>
  );
};

export default SpeechDataUpload; 