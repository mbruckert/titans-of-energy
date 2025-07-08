import React, { useState, useCallback, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import UploadProgress from '../components/UploadProgress';
import VectorModelSelector from '../components/VectorModelSelector';
import { X } from 'lucide-react';
import { API_BASE_URL, API_ENDPOINTS } from '../config/api';
import { 
  getCharacterCreationData, 
  saveCharacterCreationData, 
  getCharacterCreationFiles, 
  saveCharacterCreationFiles,
  clearCharacterCreationData 
} from '../utils/characterCreation';

interface VectorConfig {
  model_type: 'openai' | 'sentence_transformers';
  model_name: string;
  config: {
    api_key?: string;
    device?: string;
  };
}

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
  const [embeddingConfig, setEmbeddingConfig] = useState<VectorConfig>({
    model_type: 'sentence_transformers',
    model_name: 'all-MiniLM-L6-v2',
    config: { device: 'auto' }
  });

  // Restore previous selections when component mounts
  useEffect(() => {
    const existingData = getCharacterCreationData();
    
    // Restore embedding configuration
    if (existingData.styleTuningEmbeddingConfig) {
      setEmbeddingConfig(existingData.styleTuningEmbeddingConfig);
    }
    
    // Restore files from global storage
    const files = getCharacterCreationFiles();
    if (files.styleTuningFiles) {
      setFiles(files.styleTuningFiles);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const droppedFiles = Array.from(e.dataTransfer.files);
    
    // Validate JSON files
    const validFiles: File[] = [];
    const invalidFiles: string[] = [];
    
    droppedFiles.forEach(file => {
      // Check file format
      if (!file.name.toLowerCase().endsWith('.json')) {
        invalidFiles.push(`${file.name}: Only JSON files are allowed`);
        return;
      }
      
      // Validate JSON content
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const content = e.target?.result as string;
          const parsed = JSON.parse(content);
          
          // Check if it's an array of objects with question and response
          if (Array.isArray(parsed) && parsed.length > 0) {
            const isValid = parsed.every(item => 
              typeof item === 'object' && 
              item !== null && 
              'question' in item && 
              'response' in item &&
              typeof item.question === 'string' &&
              typeof item.response === 'string'
            );
            
            if (isValid) {
              validFiles.push(file);
              setFiles(prevFiles => [...prevFiles, file]);
            } else {
              invalidFiles.push(`${file.name}: Invalid format. Must be an array of objects with "question" and "response" fields`);
              setError(invalidFiles.join(', '));
            }
          } else {
            invalidFiles.push(`${file.name}: Must be an array of question-answer pairs`);
            setError(invalidFiles.join(', '));
          }
        } catch (err) {
          invalidFiles.push(`${file.name}: Invalid JSON format`);
          setError(invalidFiles.join(', '));
        }
      };
      
      reader.readAsText(file);
    });
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files || []);
    
    // Validate JSON files
    const validFiles: File[] = [];
    const invalidFiles: string[] = [];
    
    selectedFiles.forEach(file => {
      // Check file format
      if (!file.name.toLowerCase().endsWith('.json')) {
        invalidFiles.push(`${file.name}: Only JSON files are allowed`);
        return;
      }
      
      // Validate JSON content
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const content = e.target?.result as string;
          const parsed = JSON.parse(content);
          
          // Check if it's an array of objects with question and response
          if (Array.isArray(parsed) && parsed.length > 0) {
            const isValid = parsed.every(item => 
              typeof item === 'object' && 
              item !== null && 
              'question' in item && 
              'response' in item &&
              typeof item.question === 'string' &&
              typeof item.response === 'string'
            );
            
            if (isValid) {
              validFiles.push(file);
              setFiles(prevFiles => [...prevFiles, file]);
            } else {
              invalidFiles.push(`${file.name}: Invalid format. Must be an array of objects with "question" and "response" fields`);
              setError(invalidFiles.join(', '));
            }
          } else {
            invalidFiles.push(`${file.name}: Must be an array of question-answer pairs`);
            setError(invalidFiles.join(', '));
          }
        } catch (err) {
          invalidFiles.push(`${file.name}: Invalid JSON format`);
          setError(invalidFiles.join(', '));
        }
      };
      
      reader.readAsText(file);
    });
  };

  const removeFile = (index: number) => {
    setFiles(prevFiles => prevFiles.filter((_, i) => i !== index));
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
      
      // Add wakeword if available
      if (characterData.wakeword) {
        formData.append('wakeword', characterData.wakeword);
      }
      
      // Add LLM configuration
      if (characterData.llm_model) {
        formData.append('llm_model', characterData.llm_model);
        formData.append('llm_config', JSON.stringify(characterData.llm_config || {}));
      }

      // Add voice cloning settings (if configured)
      if (characterData.voice_cloning_settings) {
        formData.append('voice_cloning_settings', JSON.stringify(characterData.voice_cloning_settings));
      }

      // Add embedding configurations
      if (characterData.knowledgeBaseEmbeddingConfig) {
        formData.append('knowledge_base_embedding_config', JSON.stringify(characterData.knowledgeBaseEmbeddingConfig));
      }
      
      // Add style tuning embedding config
      if (files.length > 0) {
        formData.append('style_tuning_embedding_config', JSON.stringify(embeddingConfig));
      }

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
      if (storedFiles.voiceCloningFiles && storedFiles.voiceCloningFiles.length > 0) {
        formData.append('voice_cloning_audio', storedFiles.voiceCloningFiles[0]);
      }

      // Add style tuning files (current step)
      if (files.length > 0) {
        formData.append('style_tuning_file', files[0]);
      }

      // Call the API with timeout (5 minutes for character creation with thinking audio)
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5 * 60 * 1000); // 5 minutes

      console.log('🚀 Starting character creation request...');
      const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.CREATE_CHARACTER}`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      console.log('📡 Received response:', response.status, response.statusText);

      if (!response.ok) {
        let errorMessage = 'Failed to create character';
        try {
          const errorData = await response.json();
          errorMessage = errorData.error || errorMessage;
        } catch (parseError) {
          console.error('Failed to parse error response:', parseError);
          errorMessage = `Server error: ${response.status} ${response.statusText}`;
        }
        throw new Error(errorMessage);
      }

      const result = await response.json();
      console.log('✅ Character creation result:', result);

      if (result.status === 'success') {
        console.log('🎉 Character created successfully!');
        // Clear all character creation data
        clearCharacterCreationData();
        
        // Navigate back to character selection
        navigate('/character-selection');
      } else {
        throw new Error(result.error || 'Character creation failed');
      }

    } catch (err) {
      console.error('❌ Character creation failed:', err);
      if (err instanceof Error) {
        if (err.name === 'AbortError') {
          setError('Character creation timed out. This may happen with complex voice cloning models. Please try again or use a simpler configuration.');
        } else {
          setError(err.message);
        }
      } else {
        setError('An error occurred during character creation');
      }
    } finally {
      setIsCreating(false);
    }
  };

  // Save current progress before navigating
  const saveProgress = () => {
    // Store style tuning files in global storage
    if (files.length > 0) {
      saveCharacterCreationFiles({ styleTuningFiles: files });
      console.log('Style tuning files stored:', files.map(f => f.name));
    }
    
    // Save style tuning configuration
    saveCharacterCreationData({
      hasStyleTuning: files.length > 0,
      styleTuningFileCount: files.length,
      styleTuningEmbeddingConfig: embeddingConfig
    });
  };

  const handleSubmit = () => {
    saveProgress();
    // Create the character now (no more steps)
    createCharacter();
  };

  const handleBack = () => {
    // Save current progress before going back
    saveProgress();
    navigate('/voice-cloning');
  };

  return (
    <div className="container mx-auto p-4 max-w-2xl">
      <div className="text-center mb-8">
        <h1 className="text-2xl font-bold mb-2">Style Tuning Configuration</h1>
        <p className="text-gray-600">Upload conversation examples to fine-tune your character's response style</p>
        <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <p className="text-sm text-blue-800">
            💡 <strong>Tip:</strong> Upload JSON files containing question-answer pairs that demonstrate how your character should respond. Format: {`[{"question": "...", "response": "..."}]`}
          </p>
        </div>
      </div>

      {/* Vector Model Configuration */}
      <div className="mb-6">
        <VectorModelSelector
          label="Style Tuning Embedding Model"
          description="Choose the embedding model for processing your style tuning data. This model will be used to create vector representations of your conversation examples."
          value={embeddingConfig}
          onChange={setEmbeddingConfig}
          className="bg-gray-50"
        />
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
        <p className="text-sm text-gray-500 mb-4">Upload JSON files with question-answer pairs to define your character's response style</p>
        <p className="text-xs text-gray-500 mb-4">Accepted formats: JSON only</p>
        <input
          type="file"
          id="file-upload"
          className="hidden"
          onChange={handleFileSelect}
          accept=".json"
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