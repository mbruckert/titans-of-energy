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

interface ModelInfo {
  id: string;
  name: string;
  type: string;
  repo: string;
  requiresKey: boolean;
  available: boolean;
  downloaded: boolean;
  custom?: boolean;
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
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loadingModels, setLoadingModels] = useState(false);

  // Voice cloning state
  const [voiceFile, setVoiceFile] = useState<File | null>(null);
  const [referenceText, setReferenceText] = useState<string>('');
  const [selectedTTSModel, setSelectedTTSModel] = useState<string>('f5tts');
  const [isTranscribing, setIsTranscribing] = useState(false);

  // File upload state
  const [knowledgeBaseFiles, setKnowledgeBaseFiles] = useState<File[]>([]);
  const [styleTuningFiles, setStyleTuningFiles] = useState<File[]>([]);

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

  // Track original configurations for change detection
  const [originalConfigs, setOriginalConfigs] = useState({
    knowledgeBaseEmbedding: null as VectorConfig | null,
    styleTuningEmbedding: null as VectorConfig | null,
    preprocessing: null as any
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

  // Model-specific settings with tracking of user modifications
  const [f5Settings, setF5Settings] = useState({});
  const [f5SettingsModified, setF5SettingsModified] = useState(new Set<string>());

  const [xttsSettings, setXttsSettings] = useState({
    language: 'en',
    repetition_penalty: 1.0,
    top_k: 50,
    top_p: 0.8,
    speed: 1.0,
    enable_text_splitting: true
  });
  const [xttsSettingsModified, setXttsSettingsModified] = useState(new Set<string>());

  const [zonosSettings, setZonosSettings] = useState({
    e1: 0.5, // happiness
    e2: 0.5, // sadness
    e3: 0.5, // disgust
    e4: 0.5, // fear
    e5: 0.5, // surprise
    e6: 0.5, // anger
    e7: 0.5, // other
    e8: 0.5, // neutral
    seed: 42,
    cfg_scale: 1.0,
    speaking_rate: 20,
    frequency_max: 12000,
    pitch_standard_deviation: 100,
    language: 'en'
  });
  const [zonosSettingsModified, setZonosSettingsModified] = useState(new Set<string>());

  // Drag and drop state for reordering
  const [draggedIndex, setDraggedIndex] = useState<number | null>(null);

  const ttsModels = [
    { id: 'f5tts', name: 'F5-TTS' },
    { id: 'xtts', name: 'XTTS-v2' },
    { id: 'zonos', name: 'Zonos-v0.1' },
  ];

  const processingSteps = [
    { id: 'clean', name: 'Clean Audio', description: 'Noise reduction' },
    { id: 'remove_silence', name: 'Remove Silence', description: 'Remove quiet segments' },
    { id: 'enhance', name: 'Enhance Audio', description: 'Quality enhancement' },
  ];

  // Helper function to determine if a model is hosted (API-based) or local
  const isHostedModel = (model: ModelInfo | undefined) => {
    if (!model) return false;
    return model.type === 'openai_api';
  };

  // Helper function to generate system prompt based on model type
  const generateSystemPrompt = (characterName: string, modelInfo: ModelInfo | undefined, existingPrompt?: string) => {
    // If there's an existing prompt, check if it needs updating
    if (existingPrompt && existingPrompt.trim()) {
      const basePromptPattern = /^You are .+ answering questions in their style, so answer in the first person\. Output at MOST 30 words\.$/;
      const localPromptPattern = /^You are .+ answering questions in their style, so answer in the first person\. Output at MOST 30 words\. MAKE SURE YOU ONLY ANSWER THE REQUESTED QUESTION AND NOTHING ELSE \+ DO NOT ASK FURTHER QUESTIONS$/;
      
      // If it's a custom prompt (doesn't match our patterns), keep it as-is
      if (!basePromptPattern.test(existingPrompt) && !localPromptPattern.test(existingPrompt)) {
        return existingPrompt;
      }
    }
    
    const basePrompt = `You are ${characterName} answering questions in their style, so answer in the first person. Output at MOST 30 words.`;
    
    if (isHostedModel(modelInfo)) {
      // Hosted models: use the base prompt as-is
      return basePrompt;
    } else {
      // Local models: add the additional instruction
      return `${basePrompt} MAKE SURE YOU ONLY ANSWER THE REQUESTED QUESTION AND NOTHING ELSE + DO NOT ASK FURTHER QUESTIONS`;
    }
  };

  // Load models from API
  useEffect(() => {
    const loadModels = async () => {
      try {
        setLoadingModels(true);
        const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.GET_LLM_MODELS}`);
        
        if (!response.ok) {
          throw new Error('Failed to load models');
        }
        
        const data = await response.json();
        
        if (data.status === 'success') {
          setModels(data.models);
        } else {
          throw new Error(data.error || 'Failed to load models');
        }
      } catch (err) {
        console.error('Error loading models:', err);
        // Fallback to default models if API fails
        setModels([
          { id: 'google-gemma-3-4b-it-qat-q4_0-gguf', name: 'Gemma3 4B', type: 'gguf', repo: 'google/gemma-3-4b-it-qat-q4_0-gguf:gemma-3-4b-it-q4_0.gguf', requiresKey: false, available: false, downloaded: false },
          { id: 'llama-3.2-3b', name: 'Llama 3.2 3B', type: 'huggingface', repo: 'meta-llama/Llama-3.2-3B', requiresKey: false, available: false, downloaded: false },
          { id: 'gpt-4o', name: 'GPT-4o', type: 'openai_api', repo: 'gpt-4o', requiresKey: true, available: true, downloaded: true },
          { id: 'gpt-4o-mini', name: 'GPT-4o-mini', type: 'openai_api', repo: 'gpt-4o-mini', requiresKey: true, available: true, downloaded: true },
        ]);
      } finally {
        setLoadingModels(false);
      }
    };

    if (isOpen) {
      loadModels();
    }
  }, [isOpen]);

  // Update system prompt when model selection changes
  useEffect(() => {
    if (selectedModel && models.length > 0) {
      const selectedModelInfo = models.find(m => m.id === selectedModel);
      const characterName = name || 'the character';
      const currentPrompt = systemPrompt;
      
      const newSystemPrompt = generateSystemPrompt(characterName, selectedModelInfo, currentPrompt);
      
      // Only update if the prompt has changed
      if (newSystemPrompt !== currentPrompt) {
        setSystemPrompt(newSystemPrompt);
      }
    }
  }, [selectedModel, models, name]);

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
              const existingPrompt = fullCharacter.llm_config.system_prompt || '';
              setSystemPrompt(existingPrompt);
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

              // Load model-specific settings
              if (fullCharacter.voice_cloning_settings.f5_settings) {
                setF5Settings(fullCharacter.voice_cloning_settings.f5_settings);
                if (fullCharacter.voice_cloning_settings.f5_settings_modified) {
                  setF5SettingsModified(new Set(fullCharacter.voice_cloning_settings.f5_settings_modified));
                }
              }

              if (fullCharacter.voice_cloning_settings.xtts_settings) {
                setXttsSettings(prev => ({
                  ...prev,
                  ...fullCharacter.voice_cloning_settings.xtts_settings
                }));
                if (fullCharacter.voice_cloning_settings.xtts_settings_modified) {
                  setXttsSettingsModified(new Set(fullCharacter.voice_cloning_settings.xtts_settings_modified));
                }
              }

              if (fullCharacter.voice_cloning_settings.zonos_settings) {
                setZonosSettings(prev => ({
                  ...prev,
                  ...fullCharacter.voice_cloning_settings.zonos_settings
                }));
                if (fullCharacter.voice_cloning_settings.zonos_settings_modified) {
                  setZonosSettingsModified(new Set(fullCharacter.voice_cloning_settings.zonos_settings_modified));
                }
              }
            }
            setReferenceText(fullCharacter.voice_cloning_reference_text || '');
            
            // Set embedding configurations
            if (fullCharacter.knowledge_base_embedding_config) {
              setKnowledgeBaseEmbeddingConfig(fullCharacter.knowledge_base_embedding_config);
            }
            if (fullCharacter.style_tuning_embedding_config) {
              setStyleTuningEmbeddingConfig(fullCharacter.style_tuning_embedding_config);
            }
            
            // Store original configurations for change detection
            setOriginalConfigs({
              knowledgeBaseEmbedding: fullCharacter.knowledge_base_embedding_config || null,
              styleTuningEmbedding: fullCharacter.style_tuning_embedding_config || null,
              preprocessing: fullCharacter.voice_cloning_settings || null
            });
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
    const files = Array.from(e.target.files || []);
    setKnowledgeBaseFiles(files);
  };

  const handleStyleTuningFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    setStyleTuningFiles(files);
  };

  const transcribeAudio = async () => {
    if (!voiceFile) return;

    setIsTranscribing(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('audio', voiceFile);

      const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.TRANSCRIBE_AUDIO}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to transcribe audio');
      }

      const result = await response.json();
      
      if (result.transcript) {
        // Append or replace the reference text
        if (referenceText.trim()) {
          setReferenceText(prev => prev + '\n\n' + result.transcript);
        } else {
          setReferenceText(result.transcript);
        }
      } else {
        throw new Error('No transcription received');
      }
    } catch (err) {
      console.error('Transcription failed:', err);
      setError(err instanceof Error ? err.message : 'Failed to transcribe audio');
    } finally {
      setIsTranscribing(false);
    }
  };

  const handlePreprocessingChange = (key: string, value: any) => {
    setPreprocessingConfig(prev => ({
      ...prev,
      [key]: value
    }));
  };

  // Model-specific settings handlers
  const handleF5SettingChange = (key: string, value: any) => {
    setF5Settings(prev => ({
      ...prev,
      [key]: value
    }));
    setF5SettingsModified(prev => new Set([...prev, key]));
  };

  const handleXttsSettingChange = (key: string, value: any) => {
    setXttsSettings(prev => ({
      ...prev,
      [key]: value
    }));
    setXttsSettingsModified(prev => new Set([...prev, key]));
  };

  const handleZonosSettingChange = (key: string, value: any) => {
    setZonosSettings(prev => ({
      ...prev,
      [key]: value
    }));
    setZonosSettingsModified(prev => new Set([...prev, key]));
  };

  // Function to filter settings to only include modified values
  const getModifiedSettings = (settings: any, modifiedKeys: Set<string>) => {
    const filtered: any = {};
    for (const key of modifiedKeys) {
      if (settings[key] !== undefined) {
        filtered[key] = settings[key];
      }
    }
    return filtered;
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
        model: selectedTTSModel,
        // Always include all model-specific settings and their modification state
        f5_settings: f5Settings,
        f5_settings_modified: Array.from(f5SettingsModified),
        xtts_settings: xttsSettings,
        xtts_settings_modified: Array.from(xttsSettingsModified),
        zonos_settings: zonosSettings,
        zonos_settings_modified: Array.from(zonosSettingsModified),
        // Also include the filtered settings for the current model
        ...(selectedTTSModel === 'f5tts' && getModifiedSettings(f5Settings, f5SettingsModified)),
        ...(selectedTTSModel === 'xtts' && getModifiedSettings(xttsSettings, xttsSettingsModified)),
        ...(selectedTTSModel === 'zonos' && getModifiedSettings(zonosSettings, zonosSettingsModified))
      }));
      formData.append('voice_cloning_reference_text', referenceText);

      // Add files if provided
      if (image) {
        formData.append('character_image', image);
      }
      if (voiceFile) {
        formData.append('voice_cloning_audio', voiceFile);
      }
      // Check if embedding configurations have changed
      const hasKnowledgeBaseConfigChanged = JSON.stringify(knowledgeBaseEmbeddingConfig) !== JSON.stringify(originalConfigs.knowledgeBaseEmbedding);
      const hasStyleTuningConfigChanged = JSON.stringify(styleTuningEmbeddingConfig) !== JSON.stringify(originalConfigs.styleTuningEmbedding);

      if (knowledgeBaseFiles.length > 0) {
        knowledgeBaseFiles.forEach(file => {
          formData.append('knowledge_base_file', file);
        });
        formData.append('knowledge_base_embedding_config', JSON.stringify(knowledgeBaseEmbeddingConfig));
      } else if (hasKnowledgeBaseConfigChanged) {
        // Send embedding config even without new files if it changed
        formData.append('knowledge_base_embedding_config', JSON.stringify(knowledgeBaseEmbeddingConfig));
      }
      
      if (styleTuningFiles.length > 0) {
        styleTuningFiles.forEach(file => {
          formData.append('style_tuning_file', file);
        });
        formData.append('style_tuning_embedding_config', JSON.stringify(styleTuningEmbeddingConfig));
      } else if (hasStyleTuningConfigChanged) {
        // Send embedding config even without new files if it changed
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
      <div className="bg-gray-50 rounded-lg p-6 max-w-2xl w-full max-h-[90vh] overflow-y-auto">
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
            {(() => {
              const hasKnowledgeBaseConfigChanged = JSON.stringify(knowledgeBaseEmbeddingConfig) !== JSON.stringify(originalConfigs.knowledgeBaseEmbedding);
              const hasStyleTuningConfigChanged = JSON.stringify(styleTuningEmbeddingConfig) !== JSON.stringify(originalConfigs.styleTuningEmbedding);
              const hasPreprocessingConfigChanged = JSON.stringify(preprocessingConfig) !== JSON.stringify(originalConfigs.preprocessing);
              const hasModelSpecificSettingsChanged = f5SettingsModified.size > 0 || xttsSettingsModified.size > 0 || zonosSettingsModified.size > 0;
              
              const hasChanges = image || voiceFile || knowledgeBaseFiles.length > 0 || styleTuningFiles.length > 0 || 
                                hasKnowledgeBaseConfigChanged || hasStyleTuningConfigChanged || hasPreprocessingConfigChanged || hasModelSpecificSettingsChanged;
              
              return hasChanges ? (
                <div className="mb-6 p-4 bg-gradient-to-r from-amber-50 to-orange-50 border border-amber-200 rounded-lg">
                  <h3 className="text-sm font-semibold text-amber-800 mb-2">📋 Changes Summary</h3>
                  <div className="text-sm text-amber-700 space-y-1">
                    {image && <div>🔄 Character image will be replaced</div>}
                    {voiceFile && <div>🔄 Voice audio will be replaced and re-processed</div>}
                    {hasPreprocessingConfigChanged && !voiceFile && <div>⚙️ Voice preprocessing settings changed - existing audio will be re-processed</div>}
                    {hasModelSpecificSettingsChanged && <div>🎛️ TTS model-specific settings changed</div>}
                    {knowledgeBaseFiles.length > 0 && <div>🔄 Knowledge base will be replaced and rebuilt ({knowledgeBaseFiles.length} files)</div>}
                    {hasKnowledgeBaseConfigChanged && knowledgeBaseFiles.length === 0 && <div>⚙️ Knowledge base embedding model changed - vector database will be recreated</div>}
                    {styleTuningFiles.length > 0 && <div>🔄 Style tuning data will be replaced and rebuilt ({styleTuningFiles.length} files)</div>}
                    {hasStyleTuningConfigChanged && styleTuningFiles.length === 0 && <div>⚙️ Style tuning embedding model changed - vector database will be recreated</div>}
                    {(() => {
                      const oldTTSModel = originalConfigs.preprocessing?.model || 'f5tts';
                      const newTTSModel = selectedTTSModel;
                      const oldReferenceText = character?.voice_cloning_reference_text || '';
                      const newReferenceText = referenceText;
                      
                      const shouldRegenerateThinking = voiceFile || 
                                                     hasPreprocessingConfigChanged || 
                                                     oldTTSModel !== newTTSModel || 
                                                     oldReferenceText !== newReferenceText;
                      
                      return shouldRegenerateThinking ? <div>🤔 Thinking audio will be regenerated</div> : null;
                    })()}
                    <div className="mt-2 pt-2 border-t border-amber-300 text-xs">
                      💡 Items not listed above will remain unchanged
                    </div>
                  </div>
                </div>
              ) : null;
            })()}
            
            <form onSubmit={handleSubmit} className="space-y-6">
            {/* Basic Info */}
            <div className="bg-white p-4 rounded-lg border border-gray-200">
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
            <div className="bg-white p-4 rounded-lg border border-gray-200">
              <h3 className="text-lg font-semibold mb-2">Language Model</h3>
              {loadingModels ? (
                <div className="flex items-center justify-center py-8">
                  <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-gray-500"></div>
                  <span className="ml-2 text-gray-600">Loading models...</span>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                  {models.map((model) => (
                    <button
                      key={model.id}
                      type="button"
                      onClick={() => model.available && setSelectedModel(model.id)}
                      disabled={!model.available}
                      className={`p-3 border rounded-lg text-left transition-all ${
                        selectedModel === model.id
                          ? 'border-gray-500 bg-gray-50'
                          : model.available
                          ? 'border-gray-200 hover:border-gray-300'
                          : 'border-gray-200 bg-gray-100 opacity-50 cursor-not-allowed'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <h4 className="font-semibold text-sm">{model.name}</h4>
                          {model.requiresKey && (
                            <span className="text-xs text-gray-500">Requires API Key</span>
                          )}
                          {model.custom && (
                            <span className="text-xs text-blue-600">Custom Model</span>
                          )}
                        </div>
                        <div className="flex flex-col items-end">
                          {model.available ? (
                            <span className="text-xs text-green-600">Available</span>
                          ) : (
                            <span className="text-xs text-red-600">Not Available</span>
                          )}
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              )}

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
                <div className="mb-2">
                  <div className="text-xs text-gray-500">
                    {selectedModel && models.find(m => m.id === selectedModel) ? (
                      isHostedModel(models.find(m => m.id === selectedModel)) ? (
                        <span className="text-blue-600">📡 Hosted model - using standard prompt</span>
                      ) : (
                        <span className="text-green-600">💻 Local model - using enhanced prompt with additional constraints</span>
                      )
                    ) : (
                      <span>Select a model to see prompt type</span>
                    )}
                  </div>
                </div>
                <textarea
                  id="systemPrompt"
                  value={systemPrompt}
                  onChange={(e) => setSystemPrompt(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  rows={4}
                  placeholder="Enter the system prompt..."
                />
                <p className="text-xs text-gray-500 mt-1">
                  The system prompt is automatically adjusted based on the model type. You can customize it if needed.
                </p>
              </div>
            </div>

            {/* Voice Cloning */}
            <div className="bg-white p-4 rounded-lg border border-gray-200">
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
                
                {/* Transcription Helper */}
                <div className="mt-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="text-sm font-medium text-blue-900">🎤 Auto-transcribe Voice File</h4>
                    <span className="text-xs text-blue-600">Optional</span>
                  </div>
                  <p className="text-xs text-blue-700 mb-3">
                    Transcribe your uploaded voice file to automatically generate reference text. This can help create accurate text that matches your audio.
                  </p>
                  
                  {voiceFile ? (
                    <button
                      type="button"
                      onClick={transcribeAudio}
                      disabled={isTranscribing}
                      className="w-full py-2 px-3 bg-white text-gray-700 border border-gray-300 rounded-md hover:bg-gray-50 disabled:bg-gray-50 disabled:text-gray-400 disabled:cursor-not-allowed transition-colors text-sm"
                    >
                      {isTranscribing ? (
                        <div className="flex items-center justify-center">
                          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-gray-500 mr-2"></div>
                          Transcribing...
                        </div>
                      ) : (
                        '🎤 Transcribe Audio'
                      )}
                    </button>
                  ) : (
                    <div className="text-xs text-gray-500 italic">
                      Upload a voice file above to enable transcription
                    </div>
                  )}
                  
                  <div className="mt-2 text-xs text-blue-600">
                    💡 <strong>Note:</strong> Manual transcription or review is still recommended to make the TTS more accurate. 
                    The auto-transcription is a helpful starting point, but you should verify and edit the text as needed.
                  </div>
                </div>
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

              {/* Model-specific Settings */}
              <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 mb-4">
                <h4 className="text-md font-semibold text-gray-900 mb-3">Model-specific Settings</h4>
                
                {/* F5-TTS Settings */}
                {selectedTTSModel === 'f5tts' && (
                  <div className="space-y-4">
                    <p className="text-sm text-gray-600 mb-4">F5-TTS specific configuration options</p>
                    <div className="bg-white p-4 rounded-lg border">
                      <p className="text-sm text-gray-600">No additional configuration needed for F5-TTS.</p>
                    </div>
                  </div>
                )}

                {/* XTTS-v2 Settings */}
                {selectedTTSModel === 'xtts' && (
                  <div className="space-y-4">
                    <p className="text-sm text-gray-600 mb-4">XTTS-v2 specific configuration options</p>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Language
                          {xttsSettingsModified.has('language') && (
                            <span className="ml-2 text-xs text-blue-600">(Modified)</span>
                          )}
                        </label>
                        <select
                          value={xttsSettings.language}
                          onChange={(e) => handleXttsSettingChange('language', e.target.value)}
                          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                        >
                          <option value="en">English</option>
                          <option value="es">Spanish</option>
                          <option value="fr">French</option>
                          <option value="de">German</option>
                          <option value="it">Italian</option>
                          <option value="pt">Portuguese</option>
                          <option value="pl">Polish</option>
                          <option value="tr">Turkish</option>
                          <option value="ru">Russian</option>
                          <option value="nl">Dutch</option>
                          <option value="cs">Czech</option>
                          <option value="ar">Arabic</option>
                          <option value="zh-cn">Chinese (Simplified)</option>
                          <option value="ja">Japanese</option>
                          <option value="ko">Korean</option>
                          <option value="hi">Hindi</option>
                        </select>
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Repetition Penalty (0-10)
                          {xttsSettingsModified.has('repetition_penalty') && (
                            <span className="ml-2 text-xs text-blue-600">(Modified)</span>
                          )}
                        </label>
                        <input
                          type="number"
                          value={xttsSettings.repetition_penalty}
                          onChange={(e) => handleXttsSettingChange('repetition_penalty', parseFloat(e.target.value))}
                          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                          min="0"
                          max="10"
                          step="0.1"
                        />
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Top K (0-100)
                          {xttsSettingsModified.has('top_k') && (
                            <span className="ml-2 text-xs text-blue-600">(Modified)</span>
                          )}
                        </label>
                        <input
                          type="number"
                          value={xttsSettings.top_k}
                          onChange={(e) => handleXttsSettingChange('top_k', parseInt(e.target.value))}
                          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                          min="0"
                          max="100"
                          step="1"
                        />
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Top P (0-1)
                          {xttsSettingsModified.has('top_p') && (
                            <span className="ml-2 text-xs text-blue-600">(Modified)</span>
                          )}
                        </label>
                        <input
                          type="number"
                          value={xttsSettings.top_p}
                          onChange={(e) => handleXttsSettingChange('top_p', parseFloat(e.target.value))}
                          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                          min="0"
                          max="1"
                          step="0.01"
                        />
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Speed (0-1)
                          {xttsSettingsModified.has('speed') && (
                            <span className="ml-2 text-xs text-blue-600">(Modified)</span>
                          )}
                        </label>
                        <input
                          type="number"
                          value={xttsSettings.speed}
                          onChange={(e) => handleXttsSettingChange('speed', parseFloat(e.target.value))}
                          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                          min="0"
                          max="1"
                          step="0.1"
                        />
                      </div>
                    </div>

                    <div className="mt-4">
                      <label className="flex items-center">
                        <input
                          type="checkbox"
                          checked={xttsSettings.enable_text_splitting}
                          onChange={(e) => handleXttsSettingChange('enable_text_splitting', e.target.checked)}
                          className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                        />
                        <span className="ml-2 text-sm font-medium text-gray-700">
                          Enable text splitting
                          {xttsSettingsModified.has('enable_text_splitting') && (
                            <span className="ml-2 text-xs text-blue-600">(Modified)</span>
                          )}
                        </span>
                      </label>
                    </div>
                  </div>
                )}

                {/* Zonos Settings */}
                {selectedTTSModel === 'zonos' && (
                  <div className="space-y-4">
                    <p className="text-sm text-gray-600 mb-4">Zonos-v0.1 specific configuration options</p>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">Language</label>
                        <select
                          value={zonosSettings.language}
                          onChange={(e) => handleZonosSettingChange('language', e.target.value)}
                          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                        >
                          <option value="en">English</option>
                          <option value="es">Spanish</option>
                          <option value="fr">French</option>
                          <option value="de">German</option>
                          <option value="it">Italian</option>
                          <option value="pt">Portuguese</option>
                          <option value="pl">Polish</option>
                          <option value="tr">Turkish</option>
                          <option value="ru">Russian</option>
                          <option value="nl">Dutch</option>
                          <option value="cs">Czech</option>
                          <option value="ar">Arabic</option>
                          <option value="zh-cn">Chinese</option>
                          <option value="ja">Japanese</option>
                          <option value="hu">Hungarian</option>
                          <option value="ko">Korean</option>
                          <option value="hi">Hindi</option>
                        </select>
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">Seed</label>
                        <input
                          type="number"
                          value={zonosSettings.seed}
                          onChange={(e) => handleZonosSettingChange('seed', parseInt(e.target.value))}
                          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                        />
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">CFG Scale</label>
                        <input
                          type="number"
                          value={zonosSettings.cfg_scale}
                          onChange={(e) => handleZonosSettingChange('cfg_scale', parseFloat(e.target.value))}
                          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                          min="0"
                          max="10"
                          step="0.1"
                        />
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">Speaking Rate (5-35)</label>
                        <input
                          type="number"
                          value={zonosSettings.speaking_rate}
                          onChange={(e) => handleZonosSettingChange('speaking_rate', parseInt(e.target.value))}
                          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                          min="5"
                          max="35"
                          step="1"
                        />
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">Frequency Max (0-24000)</label>
                        <input
                          type="number"
                          value={zonosSettings.frequency_max}
                          onChange={(e) => handleZonosSettingChange('frequency_max', parseInt(e.target.value))}
                          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                          min="0"
                          max="24000"
                          step="100"
                        />
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">Pitch Standard Deviation (0-500)</label>
                        <input
                          type="number"
                          value={zonosSettings.pitch_standard_deviation}
                          onChange={(e) => handleZonosSettingChange('pitch_standard_deviation', parseInt(e.target.value))}
                          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                          min="0"
                          max="500"
                          step="10"
                        />
                      </div>
                    </div>

                    {/* Emotion Parameters */}
                    <div className="bg-white p-4 rounded-lg border">
                      <h4 className="text-sm font-medium text-gray-700 mb-3">Emotion Parameters (0-1)</h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <label className="block text-sm text-gray-600 mb-1">Happiness (e1)</label>
                          <input
                            type="range"
                            value={zonosSettings.e1}
                            onChange={(e) => handleZonosSettingChange('e1', parseFloat(e.target.value))}
                            className="w-full"
                            min="0"
                            max="1"
                            step="0.01"
                          />
                          <span className="text-xs text-gray-500">{zonosSettings.e1}</span>
                        </div>

                        <div>
                          <label className="block text-sm text-gray-600 mb-1">Sadness (e2)</label>
                          <input
                            type="range"
                            value={zonosSettings.e2}
                            onChange={(e) => handleZonosSettingChange('e2', parseFloat(e.target.value))}
                            className="w-full"
                            min="0"
                            max="1"
                            step="0.01"
                          />
                          <span className="text-xs text-gray-500">{zonosSettings.e2}</span>
                        </div>

                        <div>
                          <label className="block text-sm text-gray-600 mb-1">Disgust (e3)</label>
                          <input
                            type="range"
                            value={zonosSettings.e3}
                            onChange={(e) => handleZonosSettingChange('e3', parseFloat(e.target.value))}
                            className="w-full"
                            min="0"
                            max="1"
                            step="0.01"
                          />
                          <span className="text-xs text-gray-500">{zonosSettings.e3}</span>
                        </div>

                        <div>
                          <label className="block text-sm text-gray-600 mb-1">Fear (e4)</label>
                          <input
                            type="range"
                            value={zonosSettings.e4}
                            onChange={(e) => handleZonosSettingChange('e4', parseFloat(e.target.value))}
                            className="w-full"
                            min="0"
                            max="1"
                            step="0.01"
                          />
                          <span className="text-xs text-gray-500">{zonosSettings.e4}</span>
                        </div>

                        <div>
                          <label className="block text-sm text-gray-600 mb-1">Surprise (e5)</label>
                          <input
                            type="range"
                            value={zonosSettings.e5}
                            onChange={(e) => handleZonosSettingChange('e5', parseFloat(e.target.value))}
                            className="w-full"
                            min="0"
                            max="1"
                            step="0.01"
                          />
                          <span className="text-xs text-gray-500">{zonosSettings.e5}</span>
                        </div>

                        <div>
                          <label className="block text-sm text-gray-600 mb-1">Anger (e6)</label>
                          <input
                            type="range"
                            value={zonosSettings.e6}
                            onChange={(e) => handleZonosSettingChange('e6', parseFloat(e.target.value))}
                            className="w-full"
                            min="0"
                            max="1"
                            step="0.01"
                          />
                          <span className="text-xs text-gray-500">{zonosSettings.e6}</span>
                        </div>

                        <div>
                          <label className="block text-sm text-gray-600 mb-1">Other (e7)</label>
                          <input
                            type="range"
                            value={zonosSettings.e7}
                            onChange={(e) => handleZonosSettingChange('e7', parseFloat(e.target.value))}
                            className="w-full"
                            min="0"
                            max="1"
                            step="0.01"
                          />
                          <span className="text-xs text-gray-500">{zonosSettings.e7}</span>
                        </div>

                        <div>
                          <label className="block text-sm text-gray-600 mb-1">Neutral (e8)</label>
                          <input
                            type="range"
                            value={zonosSettings.e8}
                            onChange={(e) => handleZonosSettingChange('e8', parseFloat(e.target.value))}
                            className="w-full"
                            min="0"
                            max="1"
                            step="0.01"
                          />
                          <span className="text-xs text-gray-500">{zonosSettings.e8}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Additional Files */}
            <div className="bg-white p-4 rounded-lg border border-gray-200">
              <h3 className="text-lg font-semibold mb-2">Additional Files</h3>
              
              {/* Knowledge Base Section */}
              <div className="mb-6">
                <label htmlFor="knowledgeBase" className="block text-sm font-medium text-gray-700 mb-1">
                  Knowledge Base Files
                </label>
                {currentFiles.hasKnowledgeBase && knowledgeBaseFiles.length === 0 && (
                  <div className="mb-2 p-2 bg-blue-50 border border-blue-200 rounded text-sm text-blue-700">
                    ℹ️ Current knowledge base will be kept. Upload new files to replace and rebuild the knowledge base.
                  </div>
                )}
                {knowledgeBaseFiles.length > 0 && (
                  <div className="mb-2 p-2 bg-amber-50 border border-amber-200 rounded text-sm text-amber-700">
                    🔄 {knowledgeBaseFiles.length} new knowledge base file(s) selected - will replace current knowledge base and rebuild when saved.
                  </div>
                )}
                <input
                  type="file"
                  id="knowledgeBase"
                  accept=".txt,.pdf,.doc,.docx,.json"
                  onChange={handleKnowledgeBaseFileChange}
                  className="w-full mb-3"
                  multiple
                  required={!currentFiles.hasKnowledgeBase}
                />
                {knowledgeBaseFiles.length > 0 && (
                  <div className="mt-1 mb-3 text-sm text-gray-600">
                    Selected: {knowledgeBaseFiles.map(f => f.name).join(', ')}
                  </div>
                )}
                
                {/* Knowledge Base Embedding Configuration */}
                {(knowledgeBaseFiles.length > 0 || currentFiles.hasKnowledgeBase) && (
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
                  Style Tuning Files
                </label>
                {currentFiles.hasStyleTuning && styleTuningFiles.length === 0 && (
                  <div className="mb-2 p-2 bg-blue-50 border border-blue-200 rounded text-sm text-blue-700">
                    ℹ️ Current style tuning data will be kept. Upload new files to replace and rebuild the style database.
                  </div>
                )}
                {styleTuningFiles.length > 0 && (
                  <div className="mb-2 p-2 bg-amber-50 border border-amber-200 rounded text-sm text-amber-700">
                    🔄 {styleTuningFiles.length} new style tuning file(s) selected - will replace current style data and rebuild when saved.
                  </div>
                )}
                <input
                  type="file"
                  id="styleTuning"
                  accept=".txt,.json,.yaml,.yml"
                  onChange={handleStyleTuningFileChange}
                  className="w-full mb-3"
                  multiple
                  required={!currentFiles.hasStyleTuning}
                />
                {styleTuningFiles.length > 0 && (
                  <div className="mt-1 mb-3 text-sm text-gray-600">
                    Selected: {styleTuningFiles.map(f => f.name).join(', ')}
                  </div>
                )}
                
                {/* Style Tuning Embedding Configuration */}
                {(styleTuningFiles.length > 0 || currentFiles.hasStyleTuning) && (
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