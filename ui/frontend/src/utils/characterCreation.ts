// Utility functions for character creation flow

export interface CharacterCreationData {
  name?: string;
  wakeword?: string;
  llm_model?: string;
  llm_config?: {
    model_path?: string;
    system_prompt?: string;
    api_key?: string;
  };
  hasKnowledgeBase?: boolean;
  knowledgeBaseFileCount?: number;
  knowledgeBaseEmbeddingConfig?: any;
  hasVoiceCloning?: boolean;
  voiceCloningFileCount?: number;
  voice_cloning_settings?: any;
  hasStyleTuning?: boolean;
  styleTuningFileCount?: number;
  styleTuningEmbeddingConfig?: any;
}

export interface CharacterCreationFiles {
  imageFile?: File;
  knowledgeBaseFiles?: File[];
  voiceCloningFiles?: File[];
  styleTuningFiles?: File[];
}

// Global file storage
declare global {
  interface Window {
    characterCreationFiles: CharacterCreationFiles;
  }
}

/**
 * Initialize character creation storage
 */
export const initializeCharacterCreation = () => {
  // Clear any existing data
  clearCharacterCreationData();
  
  // Initialize global file storage
  if (!window.characterCreationFiles) {
    window.characterCreationFiles = {};
  }
};

/**
 * Clear all character creation data
 */
export const clearCharacterCreationData = () => {
  // Clear session storage
  sessionStorage.removeItem('newCharacterData');
  
  // Clear global file storage
  if (window.characterCreationFiles) {
    window.characterCreationFiles = {};
  }
};

/**
 * Get current character creation data
 */
export const getCharacterCreationData = (): CharacterCreationData => {
  const data = sessionStorage.getItem('newCharacterData');
  return data ? JSON.parse(data) : {};
};

/**
 * Save character creation data
 */
export const saveCharacterCreationData = (data: Partial<CharacterCreationData>) => {
  const existingData = getCharacterCreationData();
  const updatedData = { ...existingData, ...data };
  sessionStorage.setItem('newCharacterData', JSON.stringify(updatedData));
};

/**
 * Get character creation files
 */
export const getCharacterCreationFiles = (): CharacterCreationFiles => {
  return window.characterCreationFiles || {};
};

/**
 * Save character creation files
 */
export const saveCharacterCreationFiles = (files: Partial<CharacterCreationFiles>) => {
  if (!window.characterCreationFiles) {
    window.characterCreationFiles = {};
  }
  
  Object.assign(window.characterCreationFiles, files);
};

/**
 * Check if we're in the middle of character creation
 */
export const isCharacterCreationInProgress = (): boolean => {
  const data = getCharacterCreationData();
  const files = getCharacterCreationFiles();
  
  return !!(data.name || Object.keys(files).length > 0);
};

/**
 * Get summary of current progress
 */
export const getCharacterCreationProgress = () => {
  const data = getCharacterCreationData();
  const files = getCharacterCreationFiles();
  
  return {
    hasBasicInfo: !!(data.name),
    hasModel: !!(data.llm_model),
    hasKnowledgeBase: !!(data.hasKnowledgeBase && files.knowledgeBaseFiles?.length),
    hasVoiceCloning: !!(data.hasVoiceCloning && files.voiceCloningFiles?.length),
    hasStyleTuning: !!(data.hasStyleTuning && files.styleTuningFiles?.length),
    hasImage: !!(files.imageFile)
  };
}; 