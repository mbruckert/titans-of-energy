import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import UploadProgress from "../components/UploadProgress";

const ModelSelection = () => {
    const navigate = useNavigate();
    const [selectedModel, setSelectedModel] = useState<string>('');
    const [apiKey, setApiKey] = useState<string>('');
    const [showApiKey, setShowApiKey] = useState<boolean>(false);
    const [systemPrompt, setSystemPrompt] = useState<string>('');

    const models = [
        { id: 'gemma-3-4b', name: 'Gemma3 4B', requiresKey: false },
        { id: 'llama-3.2-3b', name: 'Llama 3.2 3B', requiresKey: false },
        { id: 'gpt-4o', name: 'GPT-4o', requiresKey: true },
        { id: 'gpt-4o-mini', name: 'GPT-4o-mini', requiresKey: true },
    ];

    // Prefill system prompt based on character name
    useEffect(() => {
        const characterData = JSON.parse(sessionStorage.getItem('newCharacterData') || '{}');
        const characterName = characterData.name || 'the character';
        
        const defaultSystemPrompt = `You are ${characterName} answering questions in their style, so answer in the first person. Output at MOST 30 words.`;
        setSystemPrompt(defaultSystemPrompt);
    }, []);

    const handleSubmit = () => {
        if (selectedModel && (!models.find(m => m.id === selectedModel)?.requiresKey || apiKey) && systemPrompt.trim()) {
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

    return (
        <div className="container mx-auto p-4 max-w-2xl">
            <div className="text-center mb-8">
                <h1 className="text-2xl font-bold mb-2">Choose Your Model</h1>
                <p className="text-gray-600">Select the AI model that will power your character</p>
            </div>

            <div className="bg-white rounded-lg shadow p-6 mb-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                    {models.map((model) => (
                        <button
                            key={model.id}
                            onClick={() => {
                                setSelectedModel(model.id);
                                setShowApiKey(model.requiresKey);
                            }}
                            className={`p-4 border rounded-lg text-left transition-all ${
                                selectedModel === model.id
                                    ? 'border-black bg-gray-50'
                                    : 'border-gray-200 hover:border-gray-300'
                            }`}
                        >
                            <h3 className="font-semibold mb-1">{model.name}</h3>
                            {model.requiresKey && (
                                <span className="text-sm text-gray-500">Requires API Key</span>
                            )}
                        </button>
                    ))}
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
                    disabled={!selectedModel || (showApiKey && !apiKey) || !systemPrompt.trim()}
                    className={`px-4 py-2 rounded-md ${
                        selectedModel && (!showApiKey || apiKey) && systemPrompt.trim()
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
