import React, { useState, useEffect } from 'react';
import { API_BASE_URL } from '../config/api';

interface EmbeddingModel {
  description: string;
  dimensions: number;
  performance: string;
}

interface EmbeddingModels {
  openai: { [key: string]: EmbeddingModel };
  sentence_transformers: { [key: string]: EmbeddingModel };
}

interface VectorConfig {
  model_type: 'openai' | 'sentence_transformers';
  model_name: string;
  config: {
    api_key?: string;
    device?: string;
  };
}

interface VectorModelSelectorProps {
  label: string;
  description?: string;
  value: VectorConfig;
  onChange: (config: VectorConfig) => void;
  className?: string;
}

const VectorModelSelector: React.FC<VectorModelSelectorProps> = ({
  label,
  description,
  value,
  onChange,
  className = ""
}) => {
  const [availableModels, setAvailableModels] = useState<EmbeddingModels | null>(null);
  const [loading, setLoading] = useState(true);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<{ success: boolean; message: string } | null>(null);

  useEffect(() => {
    fetchAvailableModels();
  }, []);

  const fetchAvailableModels = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/get-embedding-models`);
      const data = await response.json();
      
      if (response.ok && data.success) {
        setAvailableModels(data.models);
      } else {
        console.error('Failed to fetch embedding models:', data.error);
      }
    } catch (error) {
      console.error('Error fetching embedding models:', error);
    } finally {
      setLoading(false);
    }
  };

  const testConfiguration = async () => {
    setTesting(true);
    setTestResult(null);
    
    try {
      const response = await fetch(`${API_BASE_URL}/test-embedding-config`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_type: value.model_type,
          model_name: value.model_name,
          config: value.config
        }),
      });
      
      const data = await response.json();
      
      if (response.ok && data.success) {
        setTestResult({
          success: true,
          message: `✓ Configuration valid! Embedding dimensions: ${data.embedding_dimensions}`
        });
      } else {
        setTestResult({
          success: false,
          message: `✗ ${data.error || 'Configuration test failed'}`
        });
      }
    } catch (error) {
      setTestResult({
        success: false,
        message: `✗ Error testing configuration: ${error}`
      });
    } finally {
      setTesting(false);
    }
  };

  const handleModelTypeChange = (modelType: 'openai' | 'sentence_transformers') => {
    const defaultModelName = modelType === 'openai' 
      ? 'text-embedding-ada-002' 
      : 'all-MiniLM-L6-v2';
    
    onChange({
      model_type: modelType,
      model_name: defaultModelName,
      config: modelType === 'openai' ? { api_key: '' } : { device: 'auto' }
    });
    setTestResult(null);
  };

  const handleModelNameChange = (modelName: string) => {
    onChange({
      ...value,
      model_name: modelName
    });
    setTestResult(null);
  };

  const handleConfigChange = (key: string, val: string) => {
    onChange({
      ...value,
      config: {
        ...value.config,
        [key]: val
      }
    });
    setTestResult(null);
  };

  if (loading) {
    return (
      <div className={`p-4 border rounded-lg ${className}`}>
        <div className="animate-pulse">
          <div className="h-4 bg-gray-200 rounded mb-2"></div>
          <div className="h-8 bg-gray-200 rounded"></div>
        </div>
      </div>
    );
  }

  return (
    <div className={`p-4 border rounded-lg ${className}`}>
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-lg font-medium text-gray-900">{label}</h3>
        <button
          type="button"
          onClick={testConfiguration}
          disabled={testing || (value.model_type === 'openai' && !value.config.api_key)}
          className="px-3 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed"
        >
          {testing ? 'Testing...' : 'Test Config'}
        </button>
      </div>
      
      {description && (
        <p className="text-sm text-gray-600 mb-4">{description}</p>
      )}

      {/* Model Type Selection */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Model Type
        </label>
        <div className="grid grid-cols-2 gap-2">
          <button
            type="button"
            onClick={() => handleModelTypeChange('sentence_transformers')}
            className={`p-3 text-left border rounded-lg ${
              value.model_type === 'sentence_transformers'
                ? 'border-blue-500 bg-blue-50'
                : 'border-gray-300 hover:border-gray-400'
            }`}
          >
            <div className="font-medium">Open Source</div>
            <div className="text-xs text-gray-600">Free, runs locally</div>
          </button>
          <button
            type="button"
            onClick={() => handleModelTypeChange('openai')}
            className={`p-3 text-left border rounded-lg ${
              value.model_type === 'openai'
                ? 'border-blue-500 bg-blue-50'
                : 'border-gray-300 hover:border-gray-400'
            }`}
          >
            <div className="font-medium">OpenAI</div>
            <div className="text-xs text-gray-600">Requires API key</div>
          </button>
        </div>
      </div>

      {/* Model Selection */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Model
        </label>
        <select
          value={value.model_name}
          onChange={(e) => handleModelNameChange(e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          {availableModels && availableModels[value.model_type] && 
            Object.entries(availableModels[value.model_type]).map(([modelName, modelInfo]) => (
              <option key={modelName} value={modelName}>
                {modelName} - {modelInfo.description}
              </option>
            ))
          }
        </select>
        
        {availableModels && availableModels[value.model_type] && availableModels[value.model_type][value.model_name] && (
          <div className="mt-2 p-2 bg-gray-50 rounded text-sm">
            <div><strong>Dimensions:</strong> {availableModels[value.model_type][value.model_name].dimensions}</div>
            <div><strong>Performance:</strong> {availableModels[value.model_type][value.model_name].performance}</div>
          </div>
        )}
      </div>

      {/* Configuration Options */}
      {value.model_type === 'openai' && (
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            OpenAI API Key
          </label>
          <input
            type="password"
            value={value.config.api_key || ''}
            onChange={(e) => handleConfigChange('api_key', e.target.value)}
            placeholder="sk-..."
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <p className="text-xs text-gray-500 mt-1">
            Your API key will be stored securely and only used for this character's embeddings.
          </p>
        </div>
      )}

      {value.model_type === 'sentence_transformers' && (
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Device
          </label>
          <select
            value={value.config.device || 'auto'}
            onChange={(e) => handleConfigChange('device', e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="auto">Auto (recommended)</option>
            <option value="cpu">CPU</option>
            <option value="cuda">NVIDIA GPU</option>
            <option value="mps">Apple Silicon GPU</option>
          </select>
          <p className="text-xs text-gray-500 mt-1">
            Auto-detection will choose the best available device for your system.
          </p>
        </div>
      )}

      {/* Test Result */}
      {testResult && (
        <div className={`p-3 rounded-md text-sm ${
          testResult.success 
            ? 'bg-green-50 text-green-800 border border-green-200' 
            : 'bg-red-50 text-red-800 border border-red-200'
        }`}>
          {testResult.message}
        </div>
      )}
    </div>
  );
};

export default VectorModelSelector; 