import { API_BASE_URL, API_ENDPOINTS } from '../config/api';

interface VoiceCloningSettings {
  model?: string;
  zonos_model?: string;
  torch_device?: string;
  [key: string]: any;
}

interface Character {
  voice_cloning_settings?: VoiceCloningSettings | string;
  [key: string]: any;
}

/**
 * Check if a character uses Zonos TTS
 */
export const isZonosCharacter = (character: Character): boolean => {
  if (!character.voice_cloning_settings) {
    return false;
  }

  let settings: VoiceCloningSettings;
  
  // Handle both string and object formats
  if (typeof character.voice_cloning_settings === 'string') {
    try {
      settings = JSON.parse(character.voice_cloning_settings);
    } catch (e) {
      console.warn('Failed to parse voice_cloning_settings:', e);
      return false;
    }
  } else {
    settings = character.voice_cloning_settings;
  }

  return settings.model?.toLowerCase() === 'zonos';
};

/**
 * Get Zonos configuration from character settings
 */
export const getZonosConfig = (character: Character): { model: string; device: string } | null => {
  if (!isZonosCharacter(character)) {
    return null;
  }

  let settings: VoiceCloningSettings;
  
  if (typeof character.voice_cloning_settings === 'string') {
    try {
      settings = JSON.parse(character.voice_cloning_settings);
    } catch (e) {
      console.warn('Failed to parse voice_cloning_settings:', e);
      return null;
    }
  } else {
    settings = character.voice_cloning_settings!;
  }

  return {
    model: settings.zonos_model || 'Zyphra/Zonos-v0.1-transformer',
    device: settings.torch_device || 'auto'
  };
};

/**
 * Preload Zonos worker for a character
 */
export const preloadZonosWorker = async (character: Character): Promise<boolean> => {
  const config = getZonosConfig(character);
  
  if (!config) {
    console.log('Character does not use Zonos, skipping worker preload');
    return false;
  }

  try {
    console.log(`🚀 Preloading Zonos worker for model: ${config.model}`);
    
    const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.PRELOAD_ZONOS_WORKER}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(config),
    });

    if (response.ok) {
      const data = await response.json();
      console.log('✅ Zonos worker preloaded successfully:', data);
      return true;
    } else {
      const errorData = await response.json();
      console.warn('⚠️ Failed to preload Zonos worker:', errorData);
      return false;
    }
  } catch (error) {
    console.error('❌ Error preloading Zonos worker:', error);
    return false;
  }
};

/**
 * Get status of Zonos workers
 */
export const getZonosWorkerStatus = async (): Promise<any> => {
  try {
    const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.GET_ZONOS_WORKER_STATUS}`);
    
    if (response.ok) {
      const data = await response.json();
      return data.status;
    } else {
      console.warn('Failed to get Zonos worker status');
      return null;
    }
  } catch (error) {
    console.error('Error getting Zonos worker status:', error);
    return null;
  }
};

/**
 * Cleanup all Zonos workers
 */
export const cleanupZonosWorkers = async (): Promise<boolean> => {
  try {
    const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.CLEANUP_ZONOS_WORKERS}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (response.ok) {
      console.log('✅ Zonos workers cleaned up successfully');
      return true;
    } else {
      console.warn('⚠️ Failed to cleanup Zonos workers');
      return false;
    }
  } catch (error) {
    console.error('❌ Error cleaning up Zonos workers:', error);
    return false;
  }
}; 