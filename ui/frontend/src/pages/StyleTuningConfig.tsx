import React, { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import UploadProgress from '../components/UploadProgress';
import { X } from 'lucide-react';
import { API_BASE_URL, API_ENDPOINTS } from '../config/api';

// Global file storage for the character creation flow
declare global {
  interface Window {
    characterCreationFiles: {
      imageFile?: File;
      knowledgeBaseFiles?: File[];
      voiceCloningFiles?: File[];
      styleTuningFiles?: File[];
    };
  }
}

const StyleTuningConfig = () => {
  const navigate = useNavigate();
  const [files, setFiles] = useState<File[]>([]);
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

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

  const createCharacter = async () => {
    setIsCreating(true);
    setError(null);

    try {
      // Get all character data from session storage
      const characterData = JSON.parse(sessionStorage.getItem('newCharacterData') || '{}');
      
      if (!characterData.name) {
        throw new Error('Character name is missing');
      }

      // Get files from global storage
      const storedFiles = window.characterCreationFiles || {};

      // Create FormData for multipart upload
      const formData = new FormData();
      
      // Add basic character info
      formData.append('name', characterData.name);
      
      // Add LLM configuration
      if (characterData.llm_model) {
        formData.append('llm_model', characterData.llm_model);
        formData.append('llm_config', JSON.stringify(characterData.llm_config || {}));
      }

      // Add voice cloning settings (if configured)
      if (characterData.voice_cloning_settings) {
        formData.append('voice_cloning_settings', JSON.stringify(characterData.voice_cloning_settings));
      }

      // Add character image if available
      if (storedFiles.imageFile) {
        formData.append('character_image', storedFiles.imageFile);
      }

      // Add knowledge base files if available
      if (storedFiles.knowledgeBaseFiles && storedFiles.knowledgeBaseFiles.length > 0) {
        formData.append('knowledge_base_file', storedFiles.knowledgeBaseFiles[0]);
      }

      // Add voice cloning audio files if available
      if (storedFiles.voiceCloningFiles && storedFiles.voiceCloningFiles.length > 0) {
        formData.append('voice_cloning_audio', storedFiles.voiceCloningFiles[0]);
      }

      // Add style tuning files (current step)
      if (files.length > 0) {
        formData.append('style_tuning_file', files[0]);
      }

      // Call the API
      const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.CREATE_CHARACTER}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to create character');
      }

      const result = await response.json();

      if (result.status === 'success') {
        // Clear session storage and global file storage
        sessionStorage.removeItem('newCharacterData');
        window.characterCreationFiles = {};
        
        // Navigate back to character selection
        navigate('/character-selection');
      } else {
        throw new Error(result.error || 'Character creation failed');
      }

    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsCreating(false);
    }
  };

  const handleSubmit = () => {
    // Initialize global file storage if it doesn't exist
    if (!window.characterCreationFiles) {
      window.characterCreationFiles = {};
    }
    
    // Store style tuning files in global storage
    if (files.length > 0) {
      window.characterCreationFiles.styleTuningFiles = files;
      console.log('Style tuning files stored:', files.map(f => f.name));
    }
    
    // Get existing character data from session storage
    const existingData = JSON.parse(sessionStorage.getItem('newCharacterData') || '{}');
    
    // Add style tuning configuration (not the actual files)
    const updatedData = {
      ...existingData,
      hasStyleTuning: files.length > 0,
      styleTuningFileCount: files.length
    };

    // Store updated data (without File objects)
    sessionStorage.setItem('newCharacterData', JSON.stringify(updatedData));
    
    // Create the character now (no more steps)
    createCharacter();
  };



  return (
    <div className="container mx-auto p-4 max-w-2xl">
      <div className="text-center mb-8">
        <h1 className="text-2xl font-bold mb-2">Style Tuning</h1>
        <p className="text-gray-600">Configure your character's personality and response style</p>
      </div>

      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          {error}
        </div>
      )}

      {/* Debug: Show stored files */}
      {/* {window.characterCreationFiles && Object.keys(window.characterCreationFiles).length > 0 && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
          <h3 className="text-sm font-semibold text-blue-800 mb-2">Files Ready for Upload:</h3>
          <div className="space-y-1 text-sm text-blue-700">
            {window.characterCreationFiles.imageFile && (
              <div>📷 Image: {window.characterCreationFiles.imageFile.name}</div>
            )}
            {window.characterCreationFiles.knowledgeBaseFiles && window.characterCreationFiles.knowledgeBaseFiles.length > 0 && (
              <div>📚 Knowledge Base: {window.characterCreationFiles.knowledgeBaseFiles.map(f => f.name).join(', ')}</div>
            )}
            {window.characterCreationFiles.voiceCloningFiles && window.characterCreationFiles.voiceCloningFiles.length > 0 && (
              <div>🎤 Voice Cloning: {window.characterCreationFiles.voiceCloningFiles.map(f => f.name).join(', ')}</div>
            )}
          </div>
        </div>
      )} */}

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
        <p className="text-sm text-gray-500 mb-4">Upload text files with example conversations or personality descriptions</p>
        <input
          type="file"
          id="file-upload"
          className="hidden"
          onChange={handleFileSelect}
          accept=".txt,.json,.yaml,.yml"
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
          onClick={() => navigate('/voice-cloning')}
          className="px-4 py-2 border border-gray-300 rounded-md hover:bg-gray-50"
          disabled={isCreating}
        >
          Back
        </button>
        <button
          onClick={handleSubmit}
          className="px-4 py-2 bg-black text-white rounded-md hover:bg-gray-800 disabled:bg-gray-400"
          disabled={isCreating || files.length === 0}
        >
          {isCreating ? (
            <div className="flex items-center">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
              Creating...
            </div>
          ) : (
            'Create Character'
          )}
        </button>
      </div>

      <UploadProgress currentStep={4} />
    </div>
  );
};

export default StyleTuningConfig; 