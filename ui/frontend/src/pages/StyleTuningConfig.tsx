import React, { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import UploadProgress from '../components/UploadProgress';
import { X } from 'lucide-react';

const StyleTuningConfig = () => {
  const navigate = useNavigate();
  const [files, setFiles] = useState<File[]>([]);

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const droppedFiles = Array.from(e.dataTransfer.files);
    setFiles(prevFiles => [...prevFiles, ...droppedFiles]);
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files || []);
    setFiles(prevFiles => [...prevFiles, ...selectedFiles]);
  };

  const removeFile = (index: number) => {
    setFiles(prevFiles => prevFiles.filter((_, i) => i !== index));
  };

  const handleSubmit = () => {
    // file upload logic TODO
    console.log('Uploading files:', files);
    navigate('/tts-config');
  };

  return (
    <div className="container mx-auto p-4 max-w-2xl">
      <div className="text-center mb-8">
        <h1 className="text-2xl font-bold mb-2">Style Tuning</h1>
        <p className="text-gray-600">Configure your character's personality and response style</p>
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
        <p className="text-gray-600 mb-2">Select your style configuration files or drag and drop</p>
        <input
          type="file"
          id="file-upload"
          className="hidden"
          onChange={handleFileSelect}
          accept=".json,.yaml,.yml"
          multiple
        />
        <button
          onClick={() => document.getElementById('file-upload')?.click()}
          className="bg-black text-white px-4 py-2 rounded-md hover:bg-gray-800"
        >
          Browse
        </button>
        {files.length > 0 && (
          <div className="mt-4 space-y-2">
            {files.map((file, index) => (
              <div key={index} className="flex items-center justify-between bg-gray-50 p-2 rounded">
                <span className="text-sm truncate">{file.name}</span>
                <button
                  onClick={() => removeFile(index)}
                  className="text-gray-500 hover:text-red-500"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="flex justify-end space-x-2">
        <button
          onClick={() => navigate('/voice-cloning')}
          className="px-4 py-2 border border-gray-300 rounded-md hover:bg-gray-50"
        >
          Back
        </button>
        <button
          onClick={handleSubmit}
          disabled={files.length === 0}
          className={`px-4 py-2 rounded-md ${
            files.length > 0 ? 'bg-black text-white hover:bg-gray-800' : 'bg-gray-300 text-gray-500 cursor-not-allowed'
          }`}
        >
          Upload
        </button>
      </div>

      <UploadProgress currentStep={4} />
    </div>
  );
};

export default StyleTuningConfig; 