import React, { useState, useEffect, ChangeEvent } from 'react';
import { API_BASE_URL, API_ENDPOINTS } from '../config/api';
import { GripVertical } from 'lucide-react';
import VectorModelSelector from './VectorModelSelector';

interface Character {
  id: number;
  name: string;
  image_base64?: string;
  llm_model?: string;
  llm_config?: any;
  voice_cloning_settings?: any;
  voice_cloning_reference_text?: string;
  created_at?: string;
}

interface EditCharacterModalProps {
  isOpen: boolean;
  character: Character | null;
  onClose: () => void;
  onSuccess: () => void;
}

interface VectorConfig {
  model_type: 'openai' | 'sentence_transformers';
  model_name: string;
  config: {
    api_key?: string;
    device?: string;
  };
}

const EditCharacterModal: React.FC<EditCharacterModalProps> = ({ 
  isOpen, 
  character, 
  onClose, 
  onSuccess 
}) => {
  const [name, setName] = useState('');
  const [image, setImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [wakeword, setWakeword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingCharacter, setIsLoadingCharacter] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Model selection state
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [apiKey, setApiKey] = useState<string>('');
  const [systemPrompt, setSystemPrompt] = useState<string>('');

  // Voice cloning state
  const [voiceFile, setVoiceFile] = useState<File | null>(null);
  const [referenceText, setReferenceText] = useState<string>('');
  const [selectedTTSModel, setSelectedTTSModel] = useState<string>('f5tts');

  // File upload state
  const [knowledgeBaseFile, setKnowledgeBaseFile] = useState<File | null>(null);
  const [styleTuningFile, setStyleTuningFile] = useState<File | null>(null);

  // Vector model configuration state
  const [knowledgeBaseEmbeddingConfig, setKnowledgeBaseEmbeddingConfig] = useState<VectorConfig>({
    model_type: 'sentence_transformers',
    model_name: 'all-MiniLM-L6-v2',
    config: { device: 'auto' }
  });
  
  const [styleTuningEmbeddingConfig, setStyleTuningEmbeddingConfig] = useState<VectorConfig>({
    model_type: 'sentence_transformers',
    model_name: 'all-MiniLM-L6-v2',
    config: { device: 'auto' }
  });

  // Current file info (from existing character)
  const [currentFiles, setCurrentFiles] = useState({
    hasImage: false,
    hasKnowledgeBase: false,
    hasVoiceCloning: false,
    hasStyleTuning: false
  });

  // Voice cloning preprocessing configuration
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

  const models = [
    { id: 'gemma-3-4b', name: 'Gemma3 4B', requiresKey: false },
    { id: 'llama-3.2-3b', name: 'Llama 3.2 3B', requiresKey: false },
    { id: 'gpt-4o', name: 'GPT-4o', requiresKey: true },
    { id: 'gpt-4o-mini', name: 'GPT-4o-mini', requiresKey: true },
  ];

  const ttsModels = [
    { id: 'f5tts', name: 'F5-TTS' },
    { id: 'xtts', name: 'XTTS-v2' },
  ];

  const processingSteps = [
    { id: 'clean', name: 'Clean Audio', description: 'Noise reduction' },
    { id: 'remove_silence', name: 'Remove Silence', description: 'Remove quiet segments' },
    { id: 'enhance', name: 'Enhance Audio', description: 'Quality enhancement' },
  ];

  useEffect(() => {
    const fetchCharacterDetails = async () => {
      if (character) {
        setIsLoadingCharacter(true);
        try {
          // Fetch full character details
          const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.GET_CHARACTER}/${character.id}`);
          if (response.ok) {
            const data = await response.json();
            const fullCharacter = data.character;
            
            setName(fullCharacter.name);
            setPreviewUrl(fullCharacter.image_base64 || null);
            setWakeword(fullCharacter.wakeword || `hey ${fullCharacter.name.toLowerCase()}`);
            
            // Set current file status
            setCurrentFiles({
              hasImage: fullCharacter.has_image || !!fullCharacter.image_base64,
              hasKnowledgeBase: fullCharacter.has_knowledge_base || false,
              hasVoiceCloning: fullCharacter.has_voice_cloning || false,
              hasStyleTuning: fullCharacter.has_style_tuning || false
            });
            
            // Set LLM model info
            setSelectedModel(fullCharacter.llm_model || '');
            if (fullCharacter.llm_config) {
              setApiKey(fullCharacter.llm_config.api_key || '');
              setSystemPrompt(fullCharacter.llm_config.system_prompt || '');
            }
            
            // Set voice cloning info
            if (fullCharacter.voice_cloning_settings) {
              setSelectedTTSModel(fullCharacter.voice_cloning_settings.model || 'f5tts');
              // Set preprocessing config from existing settings
              setPreprocessingConfig({
                preprocess_audio: fullCharacter.voice_cloning_settings.preprocess_audio ?? true,
                clean_audio: fullCharacter.voice_cloning_settings.clean_audio ?? true,
                remove_silence: fullCharacter.voice_cloning_settings.remove_silence ?? true,
                enhance_audio: fullCharacter.voice_cloning_settings.enhance_audio ?? true,
                skip_all_processing: fullCharacter.voice_cloning_settings.skip_all_processing ?? false,
                preprocessing_order: fullCharacter.voice_cloning_settings.preprocessing_order ?? ['clean', 'remove_silence', 'enhance'],
                top_db: fullCharacter.voice_cloning_settings.top_db ?? 40.0,
                fade_length_ms: fullCharacter.voice_cloning_settings.fade_length_ms ?? 50,
                bass_boost: fullCharacter.voice_cloning_settings.bass_boost ?? true,
                treble_boost: fullCharacter.voice_cloning_settings.treble_boost ?? true,
                compression: fullCharacter.voice_cloning_settings.compression ?? true
              });
            }
            setReferenceText(fullCharacter.voice_cloning_reference_text || '');
          } else {
            // Fallback to basic character info
            setName(character.name);
            setPreviewUrl(character.image_base64 || null);
            setWakeword(`hey ${character.name.toLowerCase()}`);
            setSelectedModel(character.llm_model || '');
          }
        } catch (err) {
          console.error('Failed to fetch character details:', err);
          // Fallback to basic character info
          setName(character.name);
          setPreviewUrl(character.image_base64 || null);
          setWakeword(`hey ${character.name.toLowerCase()}`);
          setSelectedModel(character.llm_model || '');
        } finally {
          setIsLoadingCharacter(false);
        }
      }
    };

    fetchCharacterDetails();
  }, [character]);

  const handleImageChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && (file.type === 'image/jpeg' || file.type === 'image/png')) {
      setImage(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    }
  };

  const handleVoiceFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setVoiceFile(file);
    }
  };

  const handleKnowledgeBaseFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setKnowledgeBaseFile(file);
    }
  };

  const handleStyleTuningFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setStyleTuningFile(file);
    }
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

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || !character) return;

    setIsLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('name', name);
      formData.append('wakeword', wakeword);
      formData.append('llm_model', selectedModel);
      formData.append('llm_config', JSON.stringify({ 
        api_key: apiKey, 
        system_prompt: systemPrompt 
      }));
      formData.append('voice_cloning_settings', JSON.stringify({
        ...preprocessingConfig,
        model: selectedTTSModel
      }));
      formData.append('voice_cloning_reference_text', referenceText);

      // Add files if provided
      if (image) {
        formData.append('character_image', image);
      }
      if (voiceFile) {
        formData.append('voice_cloning_audio', voiceFile);
      }
      if (knowledgeBaseFile) {
        formData.append('knowledge_base_file', knowledgeBaseFile);
        formData.append('knowledge_base_embedding_config', JSON.stringify(knowledgeBaseEmbeddingConfig));
      }
      if (styleTuningFile) {
        formData.append('style_tuning_file', styleTuningFile);
        formData.append('style_tuning_embedding_config', JSON.stringify(styleTuningEmbeddingConfig));
      }

      const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.UPDATE_CHARACTER}/${character.id}`, {
        method: 'PUT',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to update character');
      }

      onSuccess();
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  if (!isOpen || !character) return null;

  return (
    <div className="fixed inset-0 bg-gray-800 bg-opacity-50 flex items-center justify-center p-4 z-50">
      <div className="bg-white rounded-lg p-6 max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        <h2 className="text-2xl font-bold mb-4">Edit Character</h2>
        
        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
            {error}
          </div>
        )}

        {isLoadingCharacter && (
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
            <span className="ml-2 text-gray-600">Loading character details...</span>
          </div>
        )}

        {!isLoadingCharacter && (
          <>
            {/* Update Summary */}
            {(image || voiceFile || knowledgeBaseFile || styleTuningFile) && (
              <div className="mb-6 p-4 bg-gradient-to-r from-amber-50 to-orange-50 border border-amber-200 rounded-lg">
                <h3 className="text-sm font-semibold text-amber-800 mb-2">📋 Changes Summary</h3>
                <div className="text-sm text-amber-700 space-y-1">
                  {image && <div>🔄 Character image will be replaced</div>}
                  {voiceFile && <div>🔄 Voice audio will be replaced and re-processed</div>}
                  {knowledgeBaseFile && <div>🔄 Knowledge base will be replaced and rebuilt</div>}
                  {styleTuningFile && <div>🔄 Style tuning data will be replaced and rebuilt</div>}
                  <div className="mt-2 pt-2 border-t border-amber-300 text-xs">
                    💡 Files not selected above will remain unchanged
                  </div>
                </div>
              </div>
            )}
            
            <form onSubmit={handleSubmit} className="space-y-6">
            {/* Basic Info */}
            <div>
              <h3 className="text-lg font-semibold mb-2">Basic Information</h3>
              <div className="mb-4">
                <label htmlFor="name" className="block text-sm font-medium text-gray-700 mb-1">
                  Character Name
                </label>
                <input
                  type="text"
                  id="name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  required
                />
              </div>
              
              <div className="mb-4">
                <label htmlFor="wakeword" className="block text-sm font-medium text-gray-700 mb-1">
                  Wake Word
                </label>
                <input
                  type="text"
                  id="wakeword"
                  value={wakeword}
                  onChange={(e) => setWakeword(e.target.value)}
                  placeholder={`hey ${name.toLowerCase() || 'character'}`}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <p className="text-xs text-gray-500 mt-1">
                  The phrase to activate voice interaction (e.g., "hey {name.toLowerCase() || 'character'}")
                </p>
              </div>
              
              <div className="mb-4">
                <label htmlFor="image" className="block text-sm font-medium text-gray-700 mb-1">
                  Character Image (PNG or JPG)
                </label>
                {currentFiles.hasImage && !image && (
                  <div className="mb-2 p-2 bg-blue-50 border border-blue-200 rounded text-sm text-blue-700">
                    ℹ️ Current image will be kept. Upload a new image to replace it.
                  </div>
                )}
                {image && (
                  <div className="mb-2 p-2 bg-amber-50 border border-amber-200 rounded text-sm text-amber-700">
                    🔄 New image selected - will replace current image when saved.
                  </div>
                )}
                <input
                  type="file"
                  id="image"
                  accept="image/jpeg,image/png"
                  onChange={handleImageChange}
                  className="w-full"
                  required={!currentFiles.hasImage}
                />
                {previewUrl && (
                  <div className="mt-2">
                    <img
                      src={previewUrl}
                      alt="Preview"
                      className="w-32 h-32 object-cover rounded-md"
                    />
                  </div>
                )}
              </div>
            </div>

            {/* LLM Model Configuration */}
            <div>
              <h3 className="text-lg font-semibold mb-2">Language Model</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                {models.map((model) => (
                  <button
                    key={model.id}
                    type="button"
                    onClick={() => setSelectedModel(model.id)}
                    className={`p-3 border rounded-lg text-left transition-all ${
                      selectedModel === model.id
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <h4 className="font-semibold text-sm">{model.name}</h4>
                    {model.requiresKey && (
                      <span className="text-xs text-gray-500">Requires API Key</span>
                    )}
                  </button>
                ))}
              </div>

              {models.find(m => m.id === selectedModel)?.requiresKey && (
                <div className="mb-4">
                  <label htmlFor="apiKey" className="block text-sm font-medium text-gray-700 mb-1">
                    API Key - Optional
                  </label>
                  <input
                    type="password"
                    id="apiKey"
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="Enter your API key"
                  />
                </div>
              )}

              <div className="mb-4">
                <label htmlFor="systemPrompt" className="block text-sm font-medium text-gray-700 mb-1">
                  System Prompt
                </label>
                <textarea
                  id="systemPrompt"
                  value={systemPrompt}
                  onChange={(e) => setSystemPrompt(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  rows={3}
                  placeholder="Enter the system prompt..."
                />
              </div>
            </div>

            {/* Voice Cloning */}
            <div>
              <h3 className="text-lg font-semibold mb-2">Voice Cloning</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                {ttsModels.map((model) => (
                  <button
                    key={model.id}
                    type="button"
                    onClick={() => setSelectedTTSModel(model.id)}
                    className={`p-3 border rounded-lg text-left transition-all ${
                      selectedTTSModel === model.id
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <h4 className="font-semibold text-sm">{model.name}</h4>
                  </button>
                ))}
              </div>

              <div className="mb-4">
                <label htmlFor="voiceFile" className="block text-sm font-medium text-gray-700 mb-1">
                  Voice Audio File
                </label>
                {currentFiles.hasVoiceCloning && !voiceFile && (
                  <div className="mb-2 p-2 bg-blue-50 border border-blue-200 rounded text-sm text-blue-700">
                    ℹ️ Current voice audio will be kept. Upload a new audio file to replace it and re-process.
                  </div>
                )}
                {voiceFile && (
                  <div className="mb-2 p-2 bg-amber-50 border border-amber-200 rounded text-sm text-amber-700">
                    🔄 New voice audio selected - will replace current audio and re-process when saved.
                  </div>
                )}
                <input
                  type="file"
                  id="voiceFile"
                  accept=".mp3,.wav,.m4a"
                  onChange={handleVoiceFileChange}
                  className="w-full"
                  required={!currentFiles.hasVoiceCloning}
                />
                {voiceFile && (
                  <div className="mt-1 text-sm text-gray-600">
                    Selected: {voiceFile.name}
                  </div>
                )}
              </div>

              <div className="mb-4">
                <label htmlFor="referenceText" className="block text-sm font-medium text-gray-700 mb-1">
                  Reference Text
                </label>
                <textarea
                  id="referenceText"
                  value={referenceText}
                  onChange={(e) => setReferenceText(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  rows={2}
                  placeholder="Text that matches the audio..."
                  required
                />
              </div>

              {/* Audio Preprocessing Configuration */}
              <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 mb-4">
                <h4 className="text-md font-semibold text-gray-900 mb-3">Audio Preprocessing Settings</h4>
                
                {/* Main preprocessing toggle */}
                <div className="mb-4">
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={preprocessingConfig.preprocess_audio}
                      onChange={(e) => handlePreprocessingChange('preprocess_audio', e.target.checked)}
                      className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
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
                          className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                        />
                        <span className="ml-2 text-sm font-medium text-gray-700">Skip all processing (copy files as-is)</span>
                      </label>
                    </div>

                    {!preprocessingConfig.skip_all_processing && (
                      <>
                        {/* Processing Order */}
                        <div className="bg-white p-3 rounded border">
                          <h5 className="text-sm font-medium text-gray-700 mb-2">Processing Order</h5>
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
                                  className={`flex items-center p-2 border rounded cursor-move transition-all ${
                                    draggedIndex === index 
                                      ? 'opacity-50 transform scale-95' 
                                      : 'hover:bg-gray-50'
                                  } ${
                                    isEnabled 
                                      ? 'border-gray-300 bg-white' 
                                      : 'border-gray-200 bg-gray-100 opacity-60'
                                  }`}
                                >
                                  <GripVertical className="w-3 h-3 text-gray-400 mr-2" />
                                  <div className="flex-1">
                                    <div className="flex items-center">
                                      <span className="text-xs font-medium text-gray-900 mr-2">
                                        {index + 1}. {stepInfo?.name}
                                      </span>
                                      {!isEnabled && (
                                        <span className="text-xs text-gray-500 bg-gray-200 px-1 py-0.5 rounded">
                                          Disabled
                                        </span>
                                      )}
                                    </div>
                                    <p className="text-xs text-gray-500">{stepInfo?.description}</p>
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
                              className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                            />
                            <span className="ml-2 text-sm text-gray-700">Clean audio</span>
                          </label>
                          
                          <label className="flex items-center">
                            <input
                              type="checkbox"
                              checked={preprocessingConfig.remove_silence}
                              onChange={(e) => handlePreprocessingChange('remove_silence', e.target.checked)}
                              className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                            />
                            <span className="ml-2 text-sm text-gray-700">Remove silence</span>
                          </label>
                          
                          <label className="flex items-center">
                            <input
                              type="checkbox"
                              checked={preprocessingConfig.enhance_audio}
                              onChange={(e) => handlePreprocessingChange('enhance_audio', e.target.checked)}
                              className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                            />
                            <span className="ml-2 text-sm text-gray-700">Enhance audio</span>
                          </label>
                        </div>

                        {/* Enhancement options */}
                        {preprocessingConfig.enhance_audio && (
                          <div className="bg-white p-3 rounded border">
                            <h5 className="text-sm font-medium text-gray-700 mb-2">Enhancement Options</h5>
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                              <label className="flex items-center">
                                <input
                                  type="checkbox"
                                  checked={preprocessingConfig.bass_boost}
                                  onChange={(e) => handlePreprocessingChange('bass_boost', e.target.checked)}
                                  className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                                />
                                <span className="ml-2 text-xs text-gray-600">Bass boost</span>
                              </label>
                              
                              <label className="flex items-center">
                                <input
                                  type="checkbox"
                                  checked={preprocessingConfig.treble_boost}
                                  onChange={(e) => handlePreprocessingChange('treble_boost', e.target.checked)}
                                  className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                                />
                                <span className="ml-2 text-xs text-gray-600">Treble boost</span>
                              </label>
                              
                              <label className="flex items-center">
                                <input
                                  type="checkbox"
                                  checked={preprocessingConfig.compression}
                                  onChange={(e) => handlePreprocessingChange('compression', e.target.checked)}
                                  className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                                />
                                <span className="ml-2 text-xs text-gray-600">Compression</span>
                              </label>
                            </div>
                          </div>
                        )}

                        {/* Advanced settings */}
                        <div className="bg-white p-3 rounded border">
                          <h5 className="text-sm font-medium text-gray-700 mb-2">Advanced Settings</h5>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                            <div>
                              <label className="block text-xs text-gray-600 mb-1">Silence threshold (dB)</label>
                              <input
                                type="number"
                                value={preprocessingConfig.top_db}
                                onChange={(e) => handlePreprocessingChange('top_db', parseFloat(e.target.value))}
                                className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
                                min="10"
                                max="80"
                                step="5"
                              />
                            </div>
                            
                            <div>
                              <label className="block text-xs text-gray-600 mb-1">Fade length (ms)</label>
                              <input
                                type="number"
                                value={preprocessingConfig.fade_length_ms}
                                onChange={(e) => handlePreprocessingChange('fade_length_ms', parseInt(e.target.value))}
                                className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
                                min="10"
                                max="200"
                                step="10"
                              />
                            </div>
                          </div>
                        </div>
                      </>
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* Additional Files */}
            <div>
              <h3 className="text-lg font-semibold mb-2">Additional Files</h3>
              
              {/* Knowledge Base Section */}
              <div className="mb-6">
                <label htmlFor="knowledgeBase" className="block text-sm font-medium text-gray-700 mb-1">
                  Knowledge Base File
                </label>
                {currentFiles.hasKnowledgeBase && !knowledgeBaseFile && (
                  <div className="mb-2 p-2 bg-blue-50 border border-blue-200 rounded text-sm text-blue-700">
                    ℹ️ Current knowledge base will be kept. Upload new files to replace and rebuild the knowledge base.
                  </div>
                )}
                {knowledgeBaseFile && (
                  <div className="mb-2 p-2 bg-amber-50 border border-amber-200 rounded text-sm text-amber-700">
                    🔄 New knowledge base file selected - will replace current knowledge base and rebuild when saved.
                  </div>
                )}
                <input
                  type="file"
                  id="knowledgeBase"
                  accept=".txt,.pdf,.doc,.docx"
                  onChange={handleKnowledgeBaseFileChange}
                  className="w-full mb-3"
                  required={!currentFiles.hasKnowledgeBase}
                />
                {knowledgeBaseFile && (
                  <div className="mt-1 mb-3 text-sm text-gray-600">
                    Selected: {knowledgeBaseFile.name}
                  </div>
                )}
                
                {/* Knowledge Base Embedding Configuration */}
                {(knowledgeBaseFile || currentFiles.hasKnowledgeBase) && (
                  <VectorModelSelector
                    label="Knowledge Base Embedding Model"
                    description="Configure the embedding model for processing knowledge base documents."
                    value={knowledgeBaseEmbeddingConfig}
                    onChange={setKnowledgeBaseEmbeddingConfig}
                    className="bg-gray-50"
                  />
                )}
              </div>

              {/* Style Tuning Section */}
              <div className="mb-4">
                <label htmlFor="styleTuning" className="block text-sm font-medium text-gray-700 mb-1">
                  Style Tuning File
                </label>
                {currentFiles.hasStyleTuning && !styleTuningFile && (
                  <div className="mb-2 p-2 bg-blue-50 border border-blue-200 rounded text-sm text-blue-700">
                    ℹ️ Current style tuning data will be kept. Upload a new file to replace and rebuild the style database.
                  </div>
                )}
                {styleTuningFile && (
                  <div className="mb-2 p-2 bg-amber-50 border border-amber-200 rounded text-sm text-amber-700">
                    🔄 New style tuning file selected - will replace current style data and rebuild when saved.
                  </div>
                )}
                <input
                  type="file"
                  id="styleTuning"
                  accept=".txt,.json,.yaml,.yml"
                  onChange={handleStyleTuningFileChange}
                  className="w-full mb-3"
                  required={!currentFiles.hasStyleTuning}
                />
                {styleTuningFile && (
                  <div className="mt-1 mb-3 text-sm text-gray-600">
                    Selected: {styleTuningFile.name}
                  </div>
                )}
                
                {/* Style Tuning Embedding Configuration */}
                {(styleTuningFile || currentFiles.hasStyleTuning) && (
                  <VectorModelSelector
                    label="Style Tuning Embedding Model"
                    description="Configure the embedding model for processing style tuning data."
                    value={styleTuningEmbeddingConfig}
                    onChange={setStyleTuningEmbeddingConfig}
                    className="bg-gray-50"
                  />
                )}
              </div>
            </div>

            <div className="flex justify-end gap-2">
              <button
                type="button"
                onClick={onClose}
                className="px-4 py-2 text-gray-600 hover:text-gray-800"
                disabled={isLoading}
              >
                Cancel
              </button>
              <button
                type="submit"
                className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 disabled:bg-gray-400"
                disabled={isLoading || !name.trim() || !referenceText.trim()}
              >
                {isLoading ? (
                  <div className="flex items-center">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    Updating...
                  </div>
                ) : (
                  'Update Character'
                )}
              </button>
            </div>
          </form>
          </>
        )}
      </div>
    </div>
  );
};

export default EditCharacterModal; 