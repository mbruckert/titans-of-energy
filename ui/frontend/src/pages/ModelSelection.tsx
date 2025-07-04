import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import UploadProgress from "../components/UploadProgress";
import { Download, Check, Loader } from 'lucide-react';
import { API_BASE_URL, API_ENDPOINTS } from '../config/api';

interface ModelInfo {
    id: string;
    name: string;
    type: string;
    repo: string;
    requiresKey: boolean;
    available: boolean;
    downloaded: boolean;
}

const ModelSelection = () => {
    const navigate = useNavigate();
    const [selectedModel, setSelectedModel] = useState<string>('');
    const [apiKey, setApiKey] = useState<string>('');
    const [showApiKey, setShowApiKey] = useState<boolean>(false);
    const [systemPrompt, setSystemPrompt] = useState<string>('');
    const [models, setModels] = useState<ModelInfo[]>([]);
    const [loading, setLoading] = useState(true);
    const [downloading, setDownloading] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);

    // Custom model state
    const [showCustomModel, setShowCustomModel] = useState<boolean>(false);
    const [customModelUrl, setCustomModelUrl] = useState<string>('');
    const [customModelName, setCustomModelName] = useState<string>('');
    const [isCustomModelDownloading, setIsCustomModelDownloading] = useState<boolean>(false);

    // Prefill system prompt based on character name
    useEffect(() => {
        const characterData = JSON.parse(sessionStorage.getItem('newCharacterData') || '{}');
        const characterName = characterData.name || 'the character';
        
        const defaultSystemPrompt = `You are ${characterName} answering questions in their style, so answer in the first person. Output at MOST 30 words.`;
        setSystemPrompt(defaultSystemPrompt);
    }, []);

    // Load models from API
    useEffect(() => {
        loadModels();
    }, []);

    const loadModels = async () => {
        try {
            setLoading(true);
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
            setError(err instanceof Error ? err.message : 'Failed to load models');
            // Fallback to default models if API fails
            setModels([
                { id: 'google-gemma-3-4b-it-qat-q4_0-gguf', name: 'Gemma3 4B', type: 'gguf', repo: 'google/gemma-3-4b-it-qat-q4_0-gguf:gemma-3-4b-it-q4_0.gguf', requiresKey: false, available: false, downloaded: false },
                { id: 'llama-3.2-3b', name: 'Llama 3.2 3B', type: 'huggingface', repo: 'meta-llama/Llama-3.2-3B', requiresKey: false, available: false, downloaded: false },
                { id: 'gpt-4o', name: 'GPT-4o', type: 'openai_api', repo: 'gpt-4o', requiresKey: true, available: true, downloaded: true },
                { id: 'gpt-4o-mini', name: 'GPT-4o-mini', type: 'openai_api', repo: 'gpt-4o-mini', requiresKey: true, available: true, downloaded: true },
            ]);
        } finally {
            setLoading(false);
        }
    };

    const downloadModel = async (model: ModelInfo) => {
        try {
            setDownloading(model.id);
            setError(null);
            
            // Determine the correct model name and type for the backend
            let modelName = model.repo;
            let modelType = 'huggingface';
            
            if (model.type === 'openai_api') {
                modelType = 'openai';
            } else if (model.type === 'gguf') {
                // GGUF models are downloaded from Hugging Face
                modelType = 'huggingface';
            } else if (model.type === 'huggingface') {
                modelType = 'huggingface';
            }
            
            // Validate that we have a model name
            if (!modelName) {
                throw new Error('Model repository name is missing');
            }
            
            const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.DOWNLOAD_MODEL}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    model_name: modelName,
                    model_type: modelType
                }),
            });
            
            if (!response.ok) {
                throw new Error('Failed to download model');
            }
            
            const data = await response.json();
            
            if (data.status === 'success') {
                // Reload models to get updated status
                await loadModels();
            } else {
                throw new Error(data.error || 'Download failed');
            }
        } catch (err) {
            console.error('Error downloading model:', err);
            setError(err instanceof Error ? err.message : 'Failed to download model');
        } finally {
            setDownloading(null);
        }
    };

    const handleModelSelect = (model: ModelInfo) => {
        if (model.available) {
            setSelectedModel(model.id);
            setShowApiKey(model.requiresKey);
        }
    };

    const downloadCustomModel = async () => {
        if (!customModelUrl.trim() || !customModelName.trim()) {
            setError('Please provide both a model URL and a name');
            return;
        }

        try {
            setIsCustomModelDownloading(true);
            setError(null);
            
            // Determine model type based on URL
            let modelType = 'huggingface';
            if (customModelUrl.includes('.gguf')) {
                modelType = 'huggingface'; // GGUF files are still downloaded via HF
            }
            
            console.log('Downloading custom model:', {
                model_name: customModelUrl,
                model_type: modelType,
                custom_name: customModelName
            });
            
            const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.DOWNLOAD_MODEL}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    model_name: customModelUrl,
                    model_type: modelType,
                    custom_name: customModelName
                }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to download model');
            }

            const result = await response.json();
            console.log('Model downloaded successfully:', result);
            
            // Refresh the models list to include the newly downloaded model
            await loadModels();
            
            // Reset the custom model form
            setCustomModelUrl('');
            setCustomModelName('');
            setShowCustomModel(false);
            
            // Show success message
            alert(`Model "${customModelName}" downloaded successfully! You can now select it from the available models.`);
            
        } catch (err) {
            console.error('Download failed:', err);
            const errorMessage = err instanceof Error ? err.message : 'Failed to download model';
            setError(`Download failed: ${errorMessage}`);
        } finally {
            setIsCustomModelDownloading(false);
        }
    };

    const handleSubmit = () => {
        const selectedModelInfo = models.find(m => m.id === selectedModel);
        if (selectedModelInfo && selectedModelInfo.available && 
            (!selectedModelInfo.requiresKey || apiKey) && systemPrompt.trim()) {
            
            // Get existing character data from session storage
            const existingData = JSON.parse(sessionStorage.getItem('newCharacterData') || '{}');
            
            // Add model configuration
            const modelConfig = {
                model_path: selectedModel,
                system_prompt: systemPrompt.trim(),
                ...(apiKey && { api_key: apiKey })
            };

            const updatedData = {
                ...existingData,
                llm_model: selectedModel,
                llm_config: modelConfig
            };

            // Store updated data
            sessionStorage.setItem('newCharacterData', JSON.stringify(updatedData));
            
            // Navigate to next step
            navigate('/knowledge-base');
        }
    };

    if (loading) {
        return (
            <div className="container mx-auto p-4 max-w-2xl">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-black mx-auto mb-4"></div>
                    <p className="text-gray-600">Loading available models...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="container mx-auto p-4 max-w-2xl">
            <div className="text-center mb-8">
                <h1 className="text-2xl font-bold mb-2">Choose Your Model</h1>
                <p className="text-gray-600">Select the AI model that will power your character</p>
            </div>

            {error && (
                <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
                    <p>{error}</p>
                    <button
                        onClick={() => setError(null)}
                        className="mt-2 text-red-800 hover:text-red-900 underline text-sm"
                    >
                        Dismiss
                    </button>
                </div>
            )}

            <div className="bg-white rounded-lg shadow p-6 mb-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                    {models.map((model) => (
                        <div key={model.id} className="relative">
                            <div className={`border rounded-lg transition-all ${
                                selectedModel === model.id && model.available
                                    ? 'border-black bg-gray-50'
                                    : model.available
                                    ? 'border-gray-200'
                                    : 'border-gray-200 bg-gray-50'
                            }`}>
                                <button
                                    onClick={() => handleModelSelect(model)}
                                    disabled={!model.available}
                                    className={`w-full p-4 text-left transition-all rounded-lg ${
                                        model.available
                                            ? 'hover:bg-gray-50 cursor-pointer'
                                            : 'cursor-not-allowed opacity-60'
                                    }`}
                                >
                                    <div className="flex items-start justify-between">
                                        <div className="flex-1 pr-3">
                                            <h3 className="font-semibold mb-1">{model.name}</h3>
                                            <div className="flex flex-col gap-1">
                                                {model.requiresKey && (
                                                    <span className="text-sm text-gray-500">Requires API Key</span>
                                                )}
                                                <div className="flex items-center gap-2">
                                                    {model.available ? (
                                                        <div className="flex items-center gap-1 text-green-600">
                                                            <Check size={16} />
                                                            <span className="text-sm">Available</span>
                                                        </div>
                                                    ) : (
                                                        <span className="text-sm text-orange-600">Not Downloaded</span>
                                                    )}
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </button>
                                
                                {/* Download button - positioned outside the main button */}
                                {!model.available && model.type !== 'openai_api' && (
                                    <div className="absolute top-4 right-4">
                                        <button
                                            onClick={(e) => {
                                                e.preventDefault();
                                                e.stopPropagation();
                                                downloadModel(model);
                                            }}
                                            disabled={downloading === model.id}
                                            className="p-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors shadow-sm"
                                            title="Download model"
                                        >
                                            {downloading === model.id ? (
                                                <Loader size={16} className="animate-spin" />
                                            ) : (
                                                <Download size={16} />
                                            )}
                                        </button>
                                    </div>
                                )}
                            </div>
                            
                            {downloading === model.id && (
                                <div className="absolute inset-0 bg-blue-50 border border-blue-200 rounded-lg flex items-center justify-center">
                                    <div className="text-center">
                                        <Loader className="w-6 h-6 animate-spin text-blue-500 mx-auto mb-2" />
                                        <p className="text-sm text-blue-600">Downloading...</p>
                                    </div>
                                </div>
                            )}
                        </div>
                    ))}
                </div>

                {/* Custom Model Section */}
                <div className="border-t pt-6">
                    <div className="flex items-center justify-between mb-4">
                        <h3 className="text-lg font-semibold">Custom Model</h3>
                        <button
                            onClick={() => setShowCustomModel(!showCustomModel)}
                            className="text-blue-600 hover:text-blue-800 text-sm font-medium"
                        >
                            {showCustomModel ? 'Hide' : 'Add Custom Model'}
                        </button>
                    </div>
                    
                    {showCustomModel && (
                        <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 space-y-4">
                            <div className="text-sm text-gray-600 mb-4">
                                <p className="mb-2">💡 <strong>How to add a custom model:</strong></p>
                                <ul className="list-disc list-inside space-y-1 text-xs">
                                    <li><strong>For GGUF models:</strong> Paste the direct link to the .gguf file (e.g., https://huggingface.co/user/model/blob/main/model.gguf)</li>
                                    <li><strong>For regular models:</strong> Paste the Hugging Face repository URL (e.g., https://huggingface.co/microsoft/DialoGPT-medium)</li>
                                    <li><strong>Repository format:</strong> You can also use repo/model format (e.g., microsoft/DialoGPT-medium)</li>
                                </ul>
                            </div>
                            
                            <div>
                                <label htmlFor="customModelName" className="block text-sm font-medium text-gray-700 mb-1">
                                    Model Name
                                </label>
                                <input
                                    type="text"
                                    id="customModelName"
                                    value={customModelName}
                                    onChange={(e) => setCustomModelName(e.target.value)}
                                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-black text-sm"
                                    placeholder="Enter a name for your custom model"
                                />
                            </div>
                            
                            <div>
                                <label htmlFor="customModelUrl" className="block text-sm font-medium text-gray-700 mb-1">
                                    Model URL or Repository
                                </label>
                                <input
                                    type="text"
                                    id="customModelUrl"
                                    value={customModelUrl}
                                    onChange={(e) => setCustomModelUrl(e.target.value)}
                                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-black text-sm"
                                    placeholder="https://huggingface.co/user/model or user/model"
                                />
                            </div>
                            
                            <div className="flex gap-2">
                                <button
                                    onClick={downloadCustomModel}
                                    disabled={isCustomModelDownloading || !customModelUrl.trim() || !customModelName.trim()}
                                    className={`px-4 py-2 rounded-md text-sm font-medium ${
                                        !isCustomModelDownloading && customModelUrl.trim() && customModelName.trim()
                                            ? 'bg-blue-600 text-white hover:bg-blue-700'
                                            : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                                    }`}
                                >
                                    {isCustomModelDownloading ? (
                                        <div className="flex items-center">
                                            <Loader size={16} className="animate-spin mr-2" />
                                            Downloading...
                                        </div>
                                    ) : (
                                        'Download Model'
                                    )}
                                </button>
                                <button
                                    onClick={() => {
                                        setShowCustomModel(false);
                                        setCustomModelUrl('');
                                        setCustomModelName('');
                                    }}
                                    className="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-50"
                                >
                                    Cancel
                                </button>
                            </div>
                        </div>
                    )}
                </div>

                {showApiKey && (
                    <div className="mt-4">
                        <label htmlFor="apiKey" className="block text-sm font-medium text-gray-700 mb-1">
                            API Key
                        </label>
                        <input
                            type="password"
                            id="apiKey"
                            value={apiKey}
                            onChange={(e) => setApiKey(e.target.value)}
                            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-black"
                            placeholder="Enter your API key"
                        />
                    </div>
                )}

                {/* System Prompt Configuration */}
                <div className="mt-4">
                    <label htmlFor="systemPrompt" className="block text-sm font-medium text-gray-700 mb-1">
                        System Prompt
                    </label>
                    <textarea
                        id="systemPrompt"
                        value={systemPrompt}
                        onChange={(e) => setSystemPrompt(e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-black"
                        rows={3}
                        placeholder="Enter the system prompt that defines how your character should behave..."
                    />
                    <p className="text-xs text-gray-500 mt-1">
                        This prompt defines your character's personality and response style. Keep it concise for better results.
                    </p>
                </div>
            </div>

            <div className="flex justify-end space-x-2">
                <button
                    onClick={() => navigate('/character-selection')}
                    className="px-4 py-2 border border-gray-300 rounded-md hover:bg-gray-50"
                >
                    Cancel
                </button>
                <button
                    onClick={handleSubmit}
                    disabled={!selectedModel || (showApiKey && !apiKey) || !systemPrompt.trim() || !models.find(m => m.id === selectedModel)?.available}
                    className={`px-4 py-2 rounded-md ${
                        selectedModel && (!showApiKey || apiKey) && systemPrompt.trim() && models.find(m => m.id === selectedModel)?.available
                            ? 'bg-black text-white hover:bg-gray-800'
                            : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                    }`}
                >
                    Continue
                </button>
            </div>

            <UploadProgress currentStep={1} />
        </div>
    );
};

export default ModelSelection;
