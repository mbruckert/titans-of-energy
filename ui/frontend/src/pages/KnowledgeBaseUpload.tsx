import React, { useState, useCallback, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import UploadProgress from '../components/UploadProgress';
import VectorModelSelector from '../components/VectorModelSelector';
import { X } from 'lucide-react';
import { 
  getCharacterCreationData, 
  saveCharacterCreationData, 
  getCharacterCreationFiles, 
  saveCharacterCreationFiles 
} from '../utils/characterCreation';

interface VectorConfig {
  model_type: 'openai' | 'sentence_transformers';
  model_name: string;
  config: {
    api_key?: string;
    device?: string;
  };
}

const KnowledgeBaseUpload = () => {
  const navigate = useNavigate();
  const [files, setFiles] = useState<File[]>([]);
  const [embeddingConfig, setEmbeddingConfig] = useState<VectorConfig>({
    model_type: 'sentence_transformers',
    model_name: 'all-MiniLM-L6-v2',
    config: { device: 'auto' }
  });

  // Restore previous selections when component mounts
  useEffect(() => {
    const existingData = getCharacterCreationData();
    
    // Restore embedding configuration
    if (existingData.knowledgeBaseEmbeddingConfig) {
      setEmbeddingConfig(existingData.knowledgeBaseEmbeddingConfig);
    }
    
    // Restore files from global storage
    const files = getCharacterCreationFiles();
    if (files.knowledgeBaseFiles) {
      setFiles(files.knowledgeBaseFiles);
    }
  }, []);

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

  // Save current progress before navigating
  const saveProgress = () => {
    // Store knowledge base files in global storage
    if (files.length > 0) {
      saveCharacterCreationFiles({ knowledgeBaseFiles: files });
      console.log('Knowledge base files stored:', files.map(f => f.name));
    }
    
    // Save knowledge base configuration
    saveCharacterCreationData({
      hasKnowledgeBase: files.length > 0,
      knowledgeBaseFileCount: files.length,
      knowledgeBaseEmbeddingConfig: embeddingConfig
    });
  };

  const handleSubmit = () => {
    saveProgress();
    navigate('/voice-cloning');
  };

  const handleBack = () => {
    // Save current progress before going back
    saveProgress();
    navigate('/model-selection');
  };

  return (
    <div className="container mx-auto p-4 max-w-2xl">
      <div className="text-center mb-8">
        <h1 className="text-2xl font-bold mb-2">Upload Knowledge Base</h1>
        <p className="text-gray-600">Upload documents to train your character's knowledge</p>
        <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <p className="text-sm text-blue-800">
            💡 <strong>Tip:</strong> Upload text documents containing information about your character's life, background, experiences, and knowledge they should possess. This helps the AI respond authentically as your character.
          </p>
        </div>
      </div>

      {/* Vector Model Configuration */}
      <div className="mb-6">
        <VectorModelSelector
          label="Knowledge Base Embedding Model"
          description="Choose the embedding model for processing your knowledge base documents. OpenAI models offer higher quality but require an API key, while open source models are free and run locally."
          value={embeddingConfig}
          onChange={setEmbeddingConfig}
          className="bg-gray-50"
        />
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
        <p className="text-gray-600 mb-2">Select your files or drag and drop</p>
        <p className="text-xs text-gray-500 mb-4">Accepted formats: TXT, PDF, DOC, DOCX</p>
        <input
          type="file"
          id="file-upload"
          className="hidden"
          onChange={handleFileSelect}
          accept=".txt,.pdf,.doc,.docx"
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

      <div className="flex justify-between">
        <button
          onClick={handleBack}
          className="px-4 py-2 border border-gray-300 rounded-md hover:bg-gray-50"
        >
          Back
        </button>
        <button
          onClick={handleSubmit}
          className="px-4 py-2 bg-black text-white rounded-md hover:bg-gray-800"
        >
          Continue
        </button>
      </div>

      <UploadProgress currentStep={2} />
    </div>
  );
};

export default KnowledgeBaseUpload; 