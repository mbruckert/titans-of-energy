import React, { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import UploadProgress from '../components/UploadProgress';
import { X, GripVertical } from 'lucide-react';
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

const VoiceCloningUpload = () => {
  const navigate = useNavigate();
  const [files, setFiles] = useState<File[]>([]);
  const [referenceText, setReferenceText] = useState<string>('');
  const [selectedModel, setSelectedModel] = useState<string>('f5tts');
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Preprocessing configuration state
  const [preprocessingConfig, setPreprocessingConfig] = useState({
    preprocess_audio: true,
    clean_audio: true,
    remove_silence: true,
    enhance_audio: true,
    skip_all_processing: false,
    preprocessing_order: ['clean', 'remove_silence', 'enhance'],
    top_db: 40.0,
    fade_length_ms: 50,
    bass_boost: true,
    treble_boost: true,
    compression: true
  });

  // Drag and drop state for reordering
  const [draggedIndex, setDraggedIndex] = useState<number | null>(null);

  const ttsModels = [
    { id: 'f5tts', name: 'F5-TTS', description: 'High quality, fast inference' },
    { id: 'xtts', name: 'XTTS-v2', description: 'Multilingual support' },
  ];

  const processingSteps = [
    { id: 'clean', name: 'Clean Audio', description: 'Noise reduction' },
    { id: 'remove_silence', name: 'Remove Silence', description: 'Remove quiet segments' },
    { id: 'enhance', name: 'Enhance Audio', description: 'Quality enhancement' },
  ];

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

  const handlePreprocessingChange = (key: string, value: any) => {
    setPreprocessingConfig(prev => ({
      ...prev,
      [key]: value
    }));
  };

  // Drag and drop handlers for reordering
  const handleDragStart = (e: React.DragEvent, index: number) => {
    setDraggedIndex(index);
    e.dataTransfer.effectAllowed = 'move';
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
  };

  const handleDragDrop = (e: React.DragEvent, dropIndex: number) => {
    e.preventDefault();
    
    if (draggedIndex === null || draggedIndex === dropIndex) {
      setDraggedIndex(null);
      return;
    }

    const newOrder = [...preprocessingConfig.preprocessing_order];
    const draggedItem = newOrder[draggedIndex];
    
    // Remove dragged item
    newOrder.splice(draggedIndex, 1);
    
    // Insert at new position
    newOrder.splice(dropIndex, 0, draggedItem);
    
    setPreprocessingConfig(prev => ({
      ...prev,
      preprocessing_order: newOrder
    }));
    
    setDraggedIndex(null);
  };

  const handleDragEnd = () => {
    setDraggedIndex(null);
  };

  const getStepInfo = (stepId: string) => {
    return processingSteps.find(step => step.id === stepId);
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

      // Add voice cloning settings with preprocessing configuration
      const voiceSettings = {
        model: selectedModel,
        reference_text: referenceText || 'Hello, how can I help you today?',
        language: 'en',
        ...preprocessingConfig
      };
      
      formData.append('voice_cloning_settings', JSON.stringify(voiceSettings));

      // Add character image if available
      if (storedFiles.imageFile) {
        formData.append('character_image', storedFiles.imageFile);
      }

      // Add knowledge base files if available
      if (storedFiles.knowledgeBaseFiles && storedFiles.knowledgeBaseFiles.length > 0) {
        formData.append('knowledge_base_file', storedFiles.knowledgeBaseFiles[0]);
      }

      // Add voice cloning audio files if available
      if (files.length > 0) {
        formData.append('voice_cloning_audio', files[0]);
      }

      // Add style tuning files if available
      if (storedFiles.styleTuningFiles && storedFiles.styleTuningFiles.length > 0) {
        formData.append('style_tuning_file', storedFiles.styleTuningFiles[0]);
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
    
    // Store voice cloning files in global storage
    if (files.length > 0) {
      window.characterCreationFiles.voiceCloningFiles = files;
      console.log('Voice cloning files stored:', files.map(f => f.name));
      console.log('Reference text:', referenceText);
    }
    
    // Get existing character data from session storage
    const existingData = JSON.parse(sessionStorage.getItem('newCharacterData') || '{}');
    
    // Add voice cloning configuration with preprocessing
    const voiceSettings = {
      model: selectedModel,
      reference_text: referenceText || 'Hello, how can I help you today?',
      language: 'en',
      ...preprocessingConfig
    };

    const updatedData = {
      ...existingData,
      hasVoiceCloning: files.length > 0,
      voiceCloningFileCount: files.length,
      voice_cloning_settings: voiceSettings
    };

    // Store updated data (without File objects)
    sessionStorage.setItem('newCharacterData', JSON.stringify(updatedData));
    
    // Navigate to style tuning (skip TTS config since it's now integrated)
    navigate('/style-tuning');
  };

  const handleCreateCharacter = () => {
    createCharacter();
  };

  return (
    <div className="container mx-auto p-4 max-w-2xl">
      <div className="text-center mb-8">
        <h1 className="text-2xl font-bold mb-2">Voice Cloning & TTS Configuration</h1>
        <p className="text-gray-600">Configure voice cloning and text-to-speech settings</p>
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
            {window.characterCreationFiles.styleTuningFiles && window.characterCreationFiles.styleTuningFiles.length > 0 && (
              <div>🎭 Style Tuning: {window.characterCreationFiles.styleTuningFiles.map(f => f.name).join(', ')}</div>
            )}
          </div>
        </div>
      )} */}

      {/* TTS Model Selection */}
      <div className="bg-white rounded-lg shadow p-6 mb-6">
        <h3 className="text-lg font-semibold mb-4">Select TTS Model</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {ttsModels.map((model) => (
            <button
              key={model.id}
              onClick={() => setSelectedModel(model.id)}
              className={`p-4 border rounded-lg text-left transition-all ${
                selectedModel === model.id
                  ? 'border-black bg-gray-50'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
            >
              <h4 className="font-semibold mb-1">{model.name}</h4>
              <p className="text-sm text-gray-600">{model.description}</p>
            </button>
          ))}
        </div>
      </div>

      {/* Audio Upload */}
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
        <p className="text-gray-600 mb-2">Select your audio files or drag and drop</p>
        <input
          type="file"
          id="file-upload"
          className="hidden"
          onChange={handleFileSelect}
          accept=".mp3,.wav,.m4a"
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

      {/* Reference Text */}
      {files.length > 0 && (
        <div className="mb-6">
          <label htmlFor="referenceText" className="block text-sm font-medium text-gray-700 mb-2">
            Reference Text (what is being said in the audio)
          </label>
          <textarea
            id="referenceText"
            value={referenceText}
            onChange={(e) => setReferenceText(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-black"
            rows={3}
            placeholder="Enter the text that matches your audio sample..."
          />
        </div>
      )}

      {/* Audio Preprocessing Configuration */}
      {files.length > 0 && (
        <div className="bg-white border border-gray-200 rounded-lg p-6 mb-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Audio Preprocessing Settings</h3>
          <p className="text-sm text-gray-600 mb-4">Configure how audio files are processed for voice cloning</p>
          
          {/* Main preprocessing toggle */}
          <div className="mb-4">
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={preprocessingConfig.preprocess_audio}
                onChange={(e) => handlePreprocessingChange('preprocess_audio', e.target.checked)}
                className="rounded border-gray-300 text-black focus:ring-black"
              />
              <span className="ml-2 text-sm font-medium text-gray-700">Enable audio preprocessing</span>
            </label>
            <p className="text-xs text-gray-500 mt-1">Turn off to use raw audio files without any processing</p>
          </div>

          {preprocessingConfig.preprocess_audio && (
            <div className="space-y-4 pl-4 border-l-2 border-gray-200">
              {/* Skip all processing option */}
              <div>
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={preprocessingConfig.skip_all_processing}
                    onChange={(e) => handlePreprocessingChange('skip_all_processing', e.target.checked)}
                    className="rounded border-gray-300 text-black focus:ring-black"
                  />
                  <span className="ml-2 text-sm font-medium text-gray-700">Skip all processing (copy files as-is)</span>
                </label>
              </div>

              {!preprocessingConfig.skip_all_processing && (
                <>
                  {/* Processing Order */}
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <h4 className="text-sm font-medium text-gray-700 mb-3">Processing Order</h4>
                    <p className="text-xs text-gray-500 mb-3">Drag and drop to reorder the processing steps</p>
                    <div className="space-y-2">
                      {preprocessingConfig.preprocessing_order.map((stepId, index) => {
                        const stepInfo = getStepInfo(stepId);
                        const isEnabled = preprocessingConfig[stepId === 'clean' ? 'clean_audio' : stepId === 'remove_silence' ? 'remove_silence' : 'enhance_audio'];
                        
                        return (
                          <div
                            key={stepId}
                            draggable
                            onDragStart={(e) => handleDragStart(e, index)}
                            onDragOver={handleDragOver}
                            onDrop={(e) => handleDragDrop(e, index)}
                            onDragEnd={handleDragEnd}
                            className={`flex items-center p-3 border rounded-md cursor-move transition-all ${
                              draggedIndex === index 
                                ? 'opacity-50 transform scale-95' 
                                : 'hover:bg-gray-50'
                            } ${
                              isEnabled 
                                ? 'border-gray-300 bg-white' 
                                : 'border-gray-200 bg-gray-100 opacity-60'
                            }`}
                          >
                            <GripVertical className="w-4 h-4 text-gray-400 mr-3" />
                            <div className="flex-1">
                              <div className="flex items-center">
                                <span className="text-sm font-medium text-gray-900 mr-2">
                                  {index + 1}. {stepInfo?.name}
                                </span>
                                {!isEnabled && (
                                  <span className="text-xs text-gray-500 bg-gray-200 px-2 py-1 rounded">
                                    Disabled
                                  </span>
                                )}
                              </div>
                              <p className="text-xs text-gray-500 mt-1">{stepInfo?.description}</p>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Individual processing options */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <label className="flex items-center">
                      <input
                        type="checkbox"
                        checked={preprocessingConfig.clean_audio}
                        onChange={(e) => handlePreprocessingChange('clean_audio', e.target.checked)}
                        className="rounded border-gray-300 text-black focus:ring-black"
                      />
                      <span className="ml-2 text-sm text-gray-700">Clean audio (noise reduction)</span>
                    </label>
                    
                    <label className="flex items-center">
                      <input
                        type="checkbox"
                        checked={preprocessingConfig.remove_silence}
                        onChange={(e) => handlePreprocessingChange('remove_silence', e.target.checked)}
                        className="rounded border-gray-300 text-black focus:ring-black"
                      />
                      <span className="ml-2 text-sm text-gray-700">Remove silence</span>
                    </label>
                    
                    <label className="flex items-center">
                      <input
                        type="checkbox"
                        checked={preprocessingConfig.enhance_audio}
                        onChange={(e) => handlePreprocessingChange('enhance_audio', e.target.checked)}
                        className="rounded border-gray-300 text-black focus:ring-black"
                      />
                      <span className="ml-2 text-sm text-gray-700">Enhance audio quality</span>
                    </label>
                  </div>

                  {/* Enhancement options */}
                  {preprocessingConfig.enhance_audio && (
                    <div className="bg-gray-50 p-4 rounded-lg">
                      <h4 className="text-sm font-medium text-gray-700 mb-3">Enhancement Options</h4>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <label className="flex items-center">
                          <input
                            type="checkbox"
                            checked={preprocessingConfig.bass_boost}
                            onChange={(e) => handlePreprocessingChange('bass_boost', e.target.checked)}
                            className="rounded border-gray-300 text-black focus:ring-black"
                          />
                          <span className="ml-2 text-sm text-gray-600">Bass boost</span>
                        </label>
                        
                        <label className="flex items-center">
                          <input
                            type="checkbox"
                            checked={preprocessingConfig.treble_boost}
                            onChange={(e) => handlePreprocessingChange('treble_boost', e.target.checked)}
                            className="rounded border-gray-300 text-black focus:ring-black"
                          />
                          <span className="ml-2 text-sm text-gray-600">Treble boost</span>
                        </label>
                        
                        <label className="flex items-center">
                          <input
                            type="checkbox"
                            checked={preprocessingConfig.compression}
                            onChange={(e) => handlePreprocessingChange('compression', e.target.checked)}
                            className="rounded border-gray-300 text-black focus:ring-black"
                          />
                          <span className="ml-2 text-sm text-gray-600">Dynamic compression</span>
                        </label>
                      </div>
                    </div>
                  )}

                  {/* Advanced settings */}
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <h4 className="text-sm font-medium text-gray-700 mb-3">Advanced Settings</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm text-gray-600 mb-1">Silence threshold (dB)</label>
                        <input
                          type="number"
                          value={preprocessingConfig.top_db}
                          onChange={(e) => handlePreprocessingChange('top_db', parseFloat(e.target.value))}
                          className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                          min="10"
                          max="80"
                          step="5"
                        />
                        <p className="text-xs text-gray-500 mt-1">Higher values remove more silence</p>
                      </div>
                      
                      <div>
                        <label className="block text-sm text-gray-600 mb-1">Fade length (ms)</label>
                        <input
                          type="number"
                          value={preprocessingConfig.fade_length_ms}
                          onChange={(e) => handlePreprocessingChange('fade_length_ms', parseInt(e.target.value))}
                          className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                          min="10"
                          max="200"
                          step="10"
                        />
                        <p className="text-xs text-gray-500 mt-1">Smooth transitions between segments</p>
                      </div>
                    </div>
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      )}

      <div className="flex justify-between">
        <button
          onClick={() => navigate('/knowledge-base')}
          className="px-4 py-2 border border-gray-300 rounded-md hover:bg-gray-50"
          disabled={isCreating}
        >
          Back
        </button>
        <div className="space-x-2">
          <button
            onClick={handleSubmit}
            disabled={isCreating || files.length === 0 || !referenceText.trim()}
            className={`px-4 py-2 rounded-md ${
              !isCreating && files.length > 0 && referenceText.trim()
                ? 'bg-black text-white hover:bg-gray-800'
                : 'bg-gray-300 text-gray-500 cursor-not-allowed'
            }`}
          >
            Continue to Style Tuning
          </button>
        </div>
      </div>

      <UploadProgress currentStep={3} />
    </div>
  );
};

export default VoiceCloningUpload; 