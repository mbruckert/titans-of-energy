export const API_BASE_URL = 'http://localhost:5000';

export const API_ENDPOINTS = {
  // Character endpoints
  GET_CHARACTERS: '/get-characters',
  GET_CHARACTER: '/get-character',
  GET_CHARACTER_WAKEWORD: '/get-character-wakeword',
  GET_CHARACTER_THINKING_AUDIO: '/get-character-thinking-audio',
  CREATE_CHARACTER: '/create-character',
  UPDATE_CHARACTER: '/update-character',
  DELETE_CHARACTER: '/delete-character',
  LOAD_CHARACTER: '/load-character',
  
  // Question endpoints
  ASK_QUESTION_TEXT: '/ask-question-text',
  ASK_QUESTION_AUDIO: '/ask-question-audio',
  TRANSCRIBE_AUDIO: '/transcribe-audio',
  
  // Chat history endpoints
  GET_CHAT_HISTORY: '/get-chat-history',
  CLEAR_CHAT_HISTORY: '/clear-chat-history',
  
  // Model endpoints
  DOWNLOAD_MODEL: '/download-model',
  GET_TTS_MODELS: '/get-tts-models',
  GET_LLM_MODELS: '/get-llm-models',
  GET_LOADED_MODELS: '/get-loaded-models',
  UNLOAD_MODELS: '/unload-models',
  GET_LOADED_LLM_MODELS: '/get-loaded-llm-models',
  UNLOAD_LLM_MODELS: '/unload-llm-models',
  UNLOAD_ALL_MODELS: '/unload-all-models',
  
  // File serving endpoints
  SERVE_AUDIO: '/serve-audio',
  SERVE_IMAGE: '/serve-image'
}; 