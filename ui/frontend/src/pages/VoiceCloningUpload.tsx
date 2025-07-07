import React, { useState, useCallback, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import UploadProgress from '../components/UploadProgress';
import { X, GripVertical } from 'lucide-react';
import { API_BASE_URL, API_ENDPOINTS } from '../config/api';
import { 
  getCharacterCreationData, 
  saveCharacterCreationData, 
  getCharacterCreationFiles, 
  saveCharacterCreationFiles,
  clearCharacterCreationData 
} from '../utils/characterCreation';

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
  const [file, setFile] = useState<File | null>(null);
  const [referenceText, setReferenceText] = useState<string>('');
  const [selectedModel, setSelectedModel] = useState<string>('f5tts');
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isTranscribing, setIsTranscribing] = useState(false);

  // Restore previous selections when component mounts
  useEffect(() => {
    const existingData = getCharacterCreationData();
    
    // Restore voice cloning settings
    if (existingData.voice_cloning_settings) {
      const settings = existingData.voice_cloning_settings;
      if (settings.model) {
        setSelectedModel(settings.model);
      }
      if (settings.ref_text) {
        setReferenceText(settings.ref_text);
      }
      
      // Restore preprocessing config
      if (settings.preprocess_audio !== undefined) {
        setPreprocessingConfig(prev => ({
          ...prev,
          preprocess_audio: settings.preprocess_audio,
          clean_audio: settings.clean_audio !== undefined ? settings.clean_audio : prev.clean_audio,
          remove_silence: settings.remove_silence !== undefined ? settings.remove_silence : prev.remove_silence,
          enhance_audio: settings.enhance_audio !== undefined ? settings.enhance_audio : prev.enhance_audio,
          skip_all_processing: settings.skip_all_processing !== undefined ? settings.skip_all_processing : prev.skip_all_processing,
          preprocessing_order: settings.preprocessing_order || prev.preprocessing_order,
          top_db: settings.top_db !== undefined ? settings.top_db : prev.top_db,
          fade_length_ms: settings.fade_length_ms !== undefined ? settings.fade_length_ms : prev.fade_length_ms,
          bass_boost: settings.bass_boost !== undefined ? settings.bass_boost : prev.bass_boost,
          treble_boost: settings.treble_boost !== undefined ? settings.treble_boost : prev.treble_boost,
          compression: settings.compression !== undefined ? settings.compression : prev.compression
        }));
      }
      
      // Restore F5 settings and modifications
      if (settings.f5_settings) {
        setF5Settings(settings.f5_settings);
      }
      if (settings.f5_settings_modified) {
        setF5SettingsModified(new Set(settings.f5_settings_modified));
      }
      
      // Restore XTTS settings and modifications
      if (settings.xtts_settings) {
        setXttsSettings(settings.xtts_settings);
      }
      if (settings.xtts_settings_modified) {
        setXttsSettingsModified(new Set(settings.xtts_settings_modified));
      }
      
      // Restore Zonos settings and modifications
      if (settings.zonos_settings) {
        setZonosSettings(settings.zonos_settings);
      }
      if (settings.zonos_settings_modified) {
        setZonosSettingsModified(new Set(settings.zonos_settings_modified));
      }
      
      // Fallback: Restore model-specific settings from direct properties (backward compatibility)
      if (!settings.xtts_settings && settings.model === 'xtts') {
        const xttsKeys = ['language', 'repetition_penalty', 'top_k', 'top_p', 'speed', 'enable_text_splitting'];
        xttsKeys.forEach(key => {
          if (settings[key] !== undefined) {
            setXttsSettings(prev => ({ ...prev, [key]: settings[key] }));
            setXttsSettingsModified(prev => new Set([...prev, key]));
          }
        });
      }
      
      if (!settings.zonos_settings && settings.model === 'zonos') {
        const zonosKeys = ['language', 'seed', 'cfg_scale', 'speaking_rate', 'frequency_max', 'pitch_standard_deviation', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8'];
        zonosKeys.forEach(key => {
          if (settings[key] !== undefined) {
            setZonosSettings(prev => ({ ...prev, [key]: settings[key] }));
            setZonosSettingsModified(prev => new Set([...prev, key]));
          }
        });
      }
    }
    
    // Restore files from global storage
    const files = getCharacterCreationFiles();
    if (files.voiceCloningFiles && files.voiceCloningFiles.length > 0) {
      setFile(files.voiceCloningFiles[0]); // Only take the first file
    }
  }, []);
  
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
    { id: 'f5tts', name: 'F5-TTS', description: 'High quality, fast inference' },
    { id: 'xtts', name: 'XTTS-v2', description: 'Multilingual support' },
    { id: 'zonos', name: 'Zonos-v0.1', description: 'Early experimental TTS model' },
  ];

  const processingSteps = [
    { id: 'clean', name: 'Clean Audio', description: 'Noise reduction' },
    { id: 'remove_silence', name: 'Remove Silence', description: 'Remove quiet segments' },
    { id: 'enhance', name: 'Enhance Audio', description: 'Quality enhancement' },
  ];

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const droppedFiles = Array.from(e.dataTransfer.files);
    
    // Only take the first file
    const firstFile = droppedFiles[0];
    if (!firstFile) return;
    
    // Check file format
    if (!firstFile.name.toLowerCase().endsWith('.wav')) {
      setError(`${firstFile.name}: Only WAV files are allowed`);
      return;
    }
    
    // Check file duration (we'll do this with audio element)
    const audio = new Audio();
    const objectUrl = URL.createObjectURL(firstFile);
    
    audio.onloadedmetadata = () => {
      const duration = audio.duration;
      if (duration < 5 || duration > 30) {
        setError(`${firstFile.name}: Duration must be between 5-30 seconds (current: ${duration.toFixed(1)}s)`);
      } else {
        setFile(firstFile);
        setError(null);
      }
      URL.revokeObjectURL(objectUrl);
    };
    
    audio.onerror = () => {
      setError(`${firstFile.name}: Could not read audio file`);
      URL.revokeObjectURL(objectUrl);
    };
    
    audio.src = objectUrl;
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files || []);
    
    // Only take the first file
    const firstFile = selectedFiles[0];
    if (!firstFile) return;
    
    // Check file format
    if (!firstFile.name.toLowerCase().endsWith('.wav')) {
      setError(`${firstFile.name}: Only WAV files are allowed`);
      return;
    }
    
    // Check file duration (we'll do this with audio element)
    const audio = new Audio();
    const objectUrl = URL.createObjectURL(firstFile);
    
    audio.onloadedmetadata = () => {
      const duration = audio.duration;
      if (duration < 5 || duration > 30) {
        setError(`${firstFile.name}: Duration must be between 5-30 seconds (current: ${duration.toFixed(1)}s)`);
      } else {
        setFile(firstFile);
        setError(null);
      }
      URL.revokeObjectURL(objectUrl);
    };
    
    audio.onerror = () => {
      setError(`${firstFile.name}: Could not read audio file`);
      URL.revokeObjectURL(objectUrl);
    };
    
    audio.src = objectUrl;
  };

  const removeFile = () => {
    setFile(null);
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

  const transcribeAudio = async () => {
    if (!file) return;

    setIsTranscribing(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('audio', file);

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

  const createCharacter = async () => {
    setIsCreating(true);
    setError(null);

    try {
      // Get all character data from utility
      const characterData = getCharacterCreationData();
      
      if (!characterData.name) {
        throw new Error('Character name is missing');
      }

      // Get files from global storage
      const storedFiles = getCharacterCreationFiles();

      // Create FormData for multipart upload
      const formData = new FormData();
      
      // Add basic character info
      formData.append('name', characterData.name);
      
      // Add wakeword if provided
      if (characterData.wakeword) {
        formData.append('wakeword', characterData.wakeword);
      }
      
      // Add LLM configuration
      if (characterData.llm_model) {
        formData.append('llm_model', characterData.llm_model);
        formData.append('llm_config', JSON.stringify(characterData.llm_config || {}));
      }

      // Add voice cloning settings with preprocessing configuration
      const voiceSettings = {
        model: selectedModel,
        ref_audio: file ? file : null,
        ref_text: referenceText || 'Hello, how can I help you today?',
        ...preprocessingConfig,
        // Always include all model-specific settings and their modification state
        f5_settings: f5Settings,
        f5_settings_modified: Array.from(f5SettingsModified),
        xtts_settings: xttsSettings,
        xtts_settings_modified: Array.from(xttsSettingsModified),
        zonos_settings: zonosSettings,
        zonos_settings_modified: Array.from(zonosSettingsModified),
        // Also include the filtered settings for the current model
        ...(selectedModel === 'f5tts' && getModifiedSettings(f5Settings, f5SettingsModified)),
        ...(selectedModel === 'xtts' && getModifiedSettings(xttsSettings, xttsSettingsModified)),
        ...(selectedModel === 'zonos' && getModifiedSettings(zonosSettings, zonosSettingsModified))
      };
      
      formData.append('voice_cloning_settings', JSON.stringify(voiceSettings));

      // Add character image if available
      if (storedFiles.imageFile) {
        formData.append('character_image', storedFiles.imageFile);
      }

      // Add knowledge base files if available (support multiple files)
      if (storedFiles.knowledgeBaseFiles && storedFiles.knowledgeBaseFiles.length > 0) {
        storedFiles.knowledgeBaseFiles.forEach(file => {
          formData.append('knowledge_base_file', file);
        });
      }

      // Add voice cloning audio files if available
      if (file) {
        formData.append('voice_cloning_audio', file);
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
        // Clear all character creation data
        clearCharacterCreationData();
        
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

  // Save current progress before navigating
  const saveProgress = () => {
    // Store voice cloning files in global storage
    if (file) {
      saveCharacterCreationFiles({ voiceCloningFiles: [file] });
      console.log('Voice cloning file stored:', file.name);
      console.log('Reference text:', referenceText);
    }
    
    // Save comprehensive voice cloning configuration
    const voiceSettings = {
      model: selectedModel,
      ref_text: referenceText || 'Hello, how can I help you today?',
      ...preprocessingConfig,
      // Always save all model-specific settings and their modification state
      f5_settings: f5Settings,
      f5_settings_modified: Array.from(f5SettingsModified),
      xtts_settings: xttsSettings,
      xtts_settings_modified: Array.from(xttsSettingsModified),
      zonos_settings: zonosSettings,
      zonos_settings_modified: Array.from(zonosSettingsModified),
      // Also include the filtered settings for backward compatibility
      ...(selectedModel === 'f5tts' && getModifiedSettings(f5Settings, f5SettingsModified)),
      ...(selectedModel === 'xtts' && getModifiedSettings(xttsSettings, xttsSettingsModified)),
      ...(selectedModel === 'zonos' && getModifiedSettings(zonosSettings, zonosSettingsModified))
    };

    saveCharacterCreationData({
      hasVoiceCloning: !!file,
      voiceCloningFileCount: file ? 1 : 0,
      voice_cloning_settings: voiceSettings
    });
  };

  const handleSubmit = () => {
    saveProgress();
    navigate('/style-tuning');
  };

  const handleBack = () => {
    // Save current progress before going back
    saveProgress();
    navigate('/knowledge-base');
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
        <p className="text-gray-600 mb-2">Select your audio file or drag and drop</p>
        <p className="text-xs text-gray-500 mb-2">Accepted formats: WAV only</p> 
        <p className="text-xs text-gray-500 mb-4">Duration must be between 5-30 seconds for best results</p>
        <input
          type="file"
          id="file-upload"
          className="hidden"
          onChange={handleFileSelect}
          accept=".wav"
        />
        <button
          onClick={() => document.getElementById('file-upload')?.click()}
          className="bg-black text-white px-4 py-2 rounded-md hover:bg-gray-800"
        >
          Browse
        </button>
        {file && (
          <div className="mt-4">
            <div className="flex items-center justify-between bg-gray-50 p-2 rounded">
              <span className="text-sm truncate">{file.name}</span>
              <button
                onClick={removeFile}
                className="text-gray-500 hover:text-red-500"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Reference Text */}
      {file && (
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
          
          {/* Transcription Helper */}
          <div className="mt-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-sm font-medium text-blue-900">🎤 Auto-transcribe Voice File</h4>
              <span className="text-xs text-blue-600">Optional</span>
            </div>
            <p className="text-xs text-blue-700 mb-3">
              Transcribe your uploaded voice file to automatically generate reference text. This can help create accurate text that matches your audio.
            </p>
            
            {file ? (
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
      )}

      {/* Audio Preprocessing Configuration */}
      {file && (
        <div className="bg-white border border-gray-200 rounded-lg p-6 mb-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Audio Preprocessing Settings</h3>
          <p className="text-sm text-gray-600 mb-4">Configure how the audio file is processed for voice cloning</p>
          
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

      {/* Model-Specific Settings */}
      {file && (
        <div className="bg-white border border-gray-200 rounded-lg p-6 mb-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">{selectedModel.toUpperCase()} Model Settings</h3>
          <div className="flex justify-between items-center mb-4">
            <p className="text-sm text-gray-600">
              Only settings that you modify from their defaults will be included in the configuration. 
              This helps keep the configuration clean and efficient.
            </p>
            <button
              onClick={() => {
                setXttsSettingsModified(new Set());
                setZonosSettingsModified(new Set());
                setF5SettingsModified(new Set());
              }}
              className="text-xs text-gray-500 hover:text-gray-700 underline"
            >
              Reset modifications
            </button>
          </div>
          
          {/* Debug: Show what settings will be sent */}
          {false && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 mb-4">
              <h4 className="text-sm font-medium text-blue-900 mb-2">Debug: Settings to be sent</h4>
              <div className="text-xs text-blue-700 space-y-1">
                {selectedModel === 'f5tts' && (
                  <div>F5 Settings: {Object.keys(getModifiedSettings(f5Settings, f5SettingsModified)).length > 0 ? 
                    JSON.stringify(getModifiedSettings(f5Settings, f5SettingsModified)) : 'None (using defaults)'}</div>
                )}
                {selectedModel === 'xtts' && (
                  <div>XTTS Settings: {Object.keys(getModifiedSettings(xttsSettings, xttsSettingsModified)).length > 0 ? 
                    JSON.stringify(getModifiedSettings(xttsSettings, xttsSettingsModified)) : 'None (using defaults)'}</div>
                )}
                {selectedModel === 'zonos' && (
                  <div>Zonos Settings: {Object.keys(getModifiedSettings(zonosSettings, zonosSettingsModified)).length > 0 ? 
                    JSON.stringify(getModifiedSettings(zonosSettings, zonosSettingsModified)) : 'None (using defaults)'}</div>
                )}
              </div>
            </div>
          )}
          
          {/* F5-TTS Settings */}
          {selectedModel === 'f5tts' && (
            <div className="space-y-4">
              <p className="text-sm text-gray-600 mb-4">F5-TTS specific configuration options</p>
              <div className="bg-gray-50 p-4 rounded-lg">
                <p className="text-sm text-gray-600">No additional configuration needed.</p>
              </div>
            </div>
          )}

          {/* XTTS-v2 Settings */}
          {selectedModel === 'xtts' && (
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
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-black"
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
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-black"
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
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-black"
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
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-black"
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
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-black"
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
                    className="rounded border-gray-300 text-black focus:ring-black"
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
          {selectedModel === 'zonos' && (
            <div className="space-y-4">
              <p className="text-sm text-gray-600 mb-4">Zonos-v0.1 specific configuration options</p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Language</label>
                  <select
                    value={zonosSettings.language}
                    onChange={(e) => handleZonosSettingChange('language', e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-black"
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
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-black"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">CFG Scale</label>
                  <input
                    type="number"
                    value={zonosSettings.cfg_scale}
                    onChange={(e) => handleZonosSettingChange('cfg_scale', parseFloat(e.target.value))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-black"
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
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-black"
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
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-black"
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
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-black"
                    min="0"
                    max="500"
                    step="10"
                  />
                </div>
              </div>

              {/* Emotion Parameters */}
              <div className="bg-gray-50 p-4 rounded-lg">
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
      )}

      <div className="flex justify-between">
        <button
          onClick={handleBack}
          className="px-4 py-2 border border-gray-300 rounded-md hover:bg-gray-50"
          disabled={isCreating}
        >
          Back
        </button>
        <div className="space-x-2">
          <button
            onClick={handleSubmit}
            disabled={isCreating || !file || !referenceText.trim()}
            className={`px-4 py-2 rounded-md ${
              !isCreating && file && referenceText.trim()
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