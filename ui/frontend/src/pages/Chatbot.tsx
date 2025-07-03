import React, { useEffect, useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { useVoiceToText } from "react-speakup";
import { Mic, MicOff, Volume2, VolumeX, Play, Pause, Trash2, ChevronDown, ChevronUp, BookOpen, Search } from "lucide-react";
import { API_BASE_URL, API_ENDPOINTS } from '../config/api';

interface KnowledgeReference {
    id: number;
    content: string;
    source: string;
    chunk_id?: number;
    keywords: string[];
    entities: string[];
    type: string;
    relevance_score?: number;
}

interface Message {
    text: string;
    isUser: boolean;
    timestamp: Date;
    audioBase64?: string;
    id?: number;
    knowledgeReferences?: KnowledgeReference[];
}

const Chatbot = () => {
    const navigate = useNavigate();
    const location = useLocation();
    const [messages, setMessages] = useState<Message[]>([]);
    const [inputMessage, setInputMessage] = useState("");
    const [isListening, setIsListening] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [isLoadingHistory, setIsLoadingHistory] = useState(true);
    const [isLoadingCharacter, setIsLoadingCharacter] = useState(true);
    const [characterLoadingStatus, setCharacterLoadingStatus] = useState("Preparing character...");
    const [error, setError] = useState<string | null>(null);
    const [currentAudio, setCurrentAudio] = useState<HTMLAudioElement | null>(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [playingMessageIndex, setPlayingMessageIndex] = useState<number | null>(null);
    const [autoPlayAudio, setAutoPlayAudio] = useState(true);
    const [expandedReferences, setExpandedReferences] = useState<Set<number>>(new Set());

    const [pendingAutoPlayAudio, setPendingAutoPlayAudio] = useState<string | null>(null);

    // Get character data from location state
    const characterId = location.state?.characterId;
    const characterName = location.state?.characterName || "Character";

    useEffect(() => {
        if (characterId) {
            loadCharacterModels();
            loadChatHistory();
        }
    }, [characterId]);

    // Cleanup audio when component unmounts
    useEffect(() => {
        return () => {
            if (currentAudio) {
                currentAudio.pause();
                currentAudio.currentTime = 0;
            }
        };
    }, [currentAudio]);

    // Handle auto-play when messages are updated
    useEffect(() => {
        if (pendingAutoPlayAudio && autoPlayAudio && messages.length > 0) {
            const lastMessage = messages[messages.length - 1];
            // Only auto-play if the last message is from the bot and has audio
            if (!lastMessage.isUser && lastMessage.audioBase64 && !isPlaying && !currentAudio) {
                const messageIndex = messages.length - 1;
                setTimeout(() => {
                    playAudioFromBase64(pendingAutoPlayAudio, messageIndex);
                    setPendingAutoPlayAudio(null);
                }, 100);
            } else {
                setPendingAutoPlayAudio(null);
            }
        }
    }, [messages, pendingAutoPlayAudio, autoPlayAudio, isPlaying, currentAudio]);

    const loadCharacterModels = async () => {
        try {
            setIsLoadingCharacter(true);
            setCharacterLoadingStatus("Loading character models...");
            
            const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.LOAD_CHARACTER}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ character_id: characterId }),
            });

            if (!response.ok) {
                console.warn('Failed to preload character models');
                setCharacterLoadingStatus("Models not preloaded - performance may be slower");
            } else {
                const data = await response.json();
                console.log('Character models loaded successfully:', data);
                setCharacterLoadingStatus("Character ready!");
            }
            
            // Give a moment to show the "ready" status
            setTimeout(() => {
                setIsLoadingCharacter(false);
            }, 1000);
            
        } catch (err) {
            console.warn('Error loading character models:', err);
            setCharacterLoadingStatus("Models not preloaded - performance may be slower");
            setTimeout(() => {
                setIsLoadingCharacter(false);
            }, 1500);
        }
    };

    const loadChatHistory = async () => {
        try {
            setIsLoadingHistory(true);
            // Add a small delay to ensure character loading is visible
            await new Promise(resolve => setTimeout(resolve, 500));
            
            const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.GET_CHAT_HISTORY}/${characterId}?limit=20`);
            
            if (response.ok) {
                const data = await response.json();
                if (data.status === 'success' && data.chat_history) {
                    // Convert chat history to message format
                    const historyMessages: Message[] = [];
                    data.chat_history.reverse().forEach((entry: any) => {
                        // Add user message
                        historyMessages.push({
                            text: entry.user_message,
                            isUser: true,
                            timestamp: new Date(entry.created_at),
                            id: entry.id
                        });
                        // Add bot response
                        historyMessages.push({
                            text: entry.bot_response,
                            isUser: false,
                            timestamp: new Date(entry.created_at),
                            audioBase64: entry.audio_base64,
                            id: entry.id,
                            knowledgeReferences: entry.knowledge_references || []
                        });
                    });
                    setMessages(historyMessages);
                }
            }
        } catch (err) {
            console.warn('Failed to load chat history:', err);
        } finally {
            setIsLoadingHistory(false);
        }
    };

    const clearChatHistory = async () => {
        if (!confirm('Are you sure you want to clear all chat history?')) {
            return;
        }

        try {
            const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.CLEAR_CHAT_HISTORY}/${characterId}`, {
                method: 'DELETE',
            });

            if (response.ok) {
                setMessages([]);
                console.log('Chat history cleared');
            } else {
                setError('Failed to clear chat history');
            }
        } catch (err) {
            setError('Error clearing chat history');
        }
    };

    const handleSendMessage = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!inputMessage.trim() || isLoading) return;

        // Stop any currently playing audio
        stopCurrentAudio();

        // Add user message
        const userMessage: Message = {
            text: inputMessage,
            isUser: true,
            timestamp: new Date(),
        };
        setMessages(prev => [...prev, userMessage]);
        const currentInput = inputMessage;
        setInputMessage("");
        setIsLoading(true);
        setError(null);

        try {
            // Call the API with optimized performance
            const requestBody = {
                character_id: characterId,
                question: currentInput
            };

            console.log('🚀 Sending request:', requestBody);
            
            const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.ASK_QUESTION_TEXT}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestBody),
            });

            if (!response.ok) {
                throw new Error('Failed to get response from character');
            }

            const data = await response.json();

            if (data.status === 'success') {
                // Add bot message
                const botMessage: Message = {
                    text: data.text_response,
                    isUser: false,
                    timestamp: new Date(),
                    audioBase64: data.audio_base64,
                    knowledgeReferences: data.knowledge_references || [],
                };
                
                // Debug logging for audio issues
                if (!data.audio_base64) {
                    console.warn('No audio base64 received from API. Character may not have voice cloning configured.');
                    console.log('API Response:', data);
                } else {
                    console.log(`✅ Audio generated in ~${data.audio_generation_time || 'unknown'}ms`);
                }
                
                setMessages(prev => {
                    const newMessages = [...prev, botMessage];
                    return newMessages;
                });

                // Set pending auto-play audio if enabled and audio is available
                if (autoPlayAudio && data.audio_base64) {
                    setPendingAutoPlayAudio(data.audio_base64);
                }
            } else {
                throw new Error(data.error || 'Failed to get response');
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An error occurred');
            // Add error message
            const errorMessage: Message = {
                text: "Sorry, I'm having trouble responding right now. Please try again.",
                isUser: false,
                timestamp: new Date(),
            };
            setMessages(prev => [...prev, errorMessage]);
        } finally {
            setIsLoading(false);
        }
    };

    const stopCurrentAudio = () => {
        if (currentAudio) {
            currentAudio.pause();
            currentAudio.currentTime = 0;
            setCurrentAudio(null);
            setIsPlaying(false);
            setPlayingMessageIndex(null);
        }
    };

    const base64ToAudioData = (base64String: string): Uint8Array | null => {
        try {
            // Remove data URL prefix if present
            if (base64String.includes(',')) {
                base64String = base64String.split(',')[1];
            }

            // Decode base64 to bytes
            const binaryString = atob(base64String);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }
            return bytes;
        } catch (error) {
            console.error('Error converting base64 to audio:', error);
            return null;
        }
    };

    const playAudioFromBase64 = (base64Audio: string, messageIndex?: number) => {
        try {
            // Stop current audio if playing to prevent overlap
            if (isPlaying || currentAudio) {
                console.log('Audio already playing, stopping current audio first');
                stopCurrentAudio();
            }

            // Convert base64 to audio data
            const audioData = base64ToAudioData(base64Audio);
            if (!audioData) {
                setError('Failed to decode audio data');
                return;
            }

            // Create blob and object URL
            const audioBlob = new Blob([audioData], { type: 'audio/wav' });
            const audioUrl = URL.createObjectURL(audioBlob);

            const audio = new Audio(audioUrl);
            
            // Set state before attempting to play
            setCurrentAudio(audio);
            setIsPlaying(true);
            if (messageIndex !== undefined) {
                setPlayingMessageIndex(messageIndex);
            }
            
            // Set up event handlers before playing
            audio.onended = () => {
                setCurrentAudio(null);
                setIsPlaying(false);
                setPlayingMessageIndex(null);
                // Clean up object URL
                URL.revokeObjectURL(audioUrl);
            };

            audio.onerror = () => {
                console.warn('Audio playback error');
                setError('Audio playback failed. The audio data may be corrupted.');
                setIsPlaying(false);
                setPlayingMessageIndex(null);
                setCurrentAudio(null);
                // Clean up object URL
                URL.revokeObjectURL(audioUrl);
            };

            // Attempt to play
            audio.play().catch(err => {
                console.warn('Failed to play audio:', err);
                setError('Failed to play audio. Please check your audio settings.');
                setIsPlaying(false);
                setPlayingMessageIndex(null);
                setCurrentAudio(null);
                // Clean up object URL
                URL.revokeObjectURL(audioUrl);
            });

        } catch (err) {
            console.warn('Error playing audio:', err);
            setError('Error playing audio.');
            // Reset state on error
            setIsPlaying(false);
            setPlayingMessageIndex(null);
            setCurrentAudio(null);
        }
    };

    const toggleAudioPlayback = (audioBase64: string, messageIndex: number) => {
        if (isPlaying && playingMessageIndex === messageIndex) {
            // Pause current audio
            stopCurrentAudio();
        } else {
            // Stop any currently playing audio first, then play the new one
            stopCurrentAudio();
            // Small delay to ensure the previous audio is fully stopped
            setTimeout(() => {
                playAudioFromBase64(audioBase64, messageIndex);
            }, 50);
        }
    };

    const { startListening, stopListening, transcript } = useVoiceToText({
        continuous: true,
        lang: "en-US",
    });

    useEffect(() => {
        if (transcript) {
            setInputMessage(transcript);
        }
    }, [transcript]);

    const handleVoiceToggle = async () => {
        try {
            if (isListening) {
                stopListening();
                setIsListening(false);
            } else {
                // Request microphone permission
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                stream.getTracks().forEach(track => track.stop()); // Stop the stream after getting permission
                
                startListening();
                setIsListening(true);
                setError(null);
            }
        } catch (err) {
            setError("Microphone access denied. Please allow microphone access to use voice input.");
            setIsListening(false);
        }
    };

    // Component for displaying knowledge references
    const KnowledgeReferences: React.FC<{ references: KnowledgeReference[], messageIndex: number }> = ({ references, messageIndex }) => {
        if (!references || references.length === 0) return null;

        const isExpanded = expandedReferences.has(messageIndex);

        const toggleExpanded = () => {
            const newExpanded = new Set(expandedReferences);
            if (isExpanded) {
                newExpanded.delete(messageIndex);
            } else {
                newExpanded.add(messageIndex);
            }
            setExpandedReferences(newExpanded);
        };

        return (
            <div className="mt-3 border-t border-gray-200 pt-2">
                <button
                    onClick={toggleExpanded}
                    className="flex items-center gap-2 text-sm text-gray-600 hover:text-gray-800 transition-colors"
                >
                    <BookOpen size={14} />
                    <span>{references.length} source{references.length > 1 ? 's' : ''} referenced</span>
                    {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                </button>
                
                {isExpanded && (
                    <div className="mt-2 space-y-2 max-h-60 overflow-y-auto">
                        {references.map((ref, index) => (
                            <div key={ref.id} className="bg-gray-50 rounded-lg p-3 text-sm border border-gray-200">
                                <div className="flex items-start justify-between mb-2">
                                    <div className="flex items-center gap-2">
                                        <Search size={12} className="text-gray-500 mt-0.5" />
                                        <span className="font-medium text-gray-700">
                                            {ref.source || `Source ${ref.id}`}
                                        </span>
                                        {ref.relevance_score && (
                                            <span className="px-2 py-1 bg-blue-100 text-blue-700 rounded-full text-xs">
                                                {ref.relevance_score}% match
                                            </span>
                                        )}
                                    </div>
                                    <span className="text-xs text-gray-500 capitalize">
                                        {ref.type}
                                    </span>
                                </div>
                                
                                <p className="text-gray-700 mb-2 line-clamp-3">
                                    {ref.content}
                                </p>
                                
                                {(ref.keywords.length > 0 || ref.entities.length > 0) && (
                                    <div className="flex flex-wrap gap-1">
                                        {ref.keywords.slice(0, 3).map((keyword, kidx) => keyword && (
                                            <span key={kidx} className="px-2 py-1 bg-green-100 text-green-700 rounded-full text-xs">
                                                {keyword}
                                            </span>
                                        ))}
                                        {ref.entities.slice(0, 2).map((entity, eidx) => entity && (
                                            <span key={eidx} className="px-2 py-1 bg-purple-100 text-purple-700 rounded-full text-xs">
                                                {entity}
                                            </span>
                                        ))}
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                )}
            </div>
        );
    };

    // Redirect if no character selected
    if (!characterId) {
        return (
            <div className="flex flex-col h-screen bg-gray-50">
                <div className="flex-1 flex items-center justify-center">
                    <div className="text-center">
                        <p className="text-xl text-gray-600 mb-4">No character selected</p>
                        <button
                            onClick={() => navigate("/character-selection")}
                            className="bg-black text-white px-6 py-2 rounded-lg hover:bg-gray-800"
                        >
                            Select Character
                        </button>
                    </div>
                </div>
            </div>
        );
    }

    // Show loading screen while character is being prepared
    if (isLoadingCharacter) {
        return (
            <div className="flex flex-col h-screen bg-gray-50">
                <div className="flex-1 flex items-center justify-center">
                    <div className="text-center max-w-md">
                        <div className="mb-6">
                            <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-black mx-auto mb-4"></div>
                            <h2 className="text-2xl font-semibold text-gray-800 mb-2">Loading {characterName}</h2>
                            <p className="text-gray-600">{characterLoadingStatus}</p>
                        </div>
                        
                        <div className="bg-white rounded-lg shadow-sm p-4 border border-gray-200">
                            <div className="flex items-center gap-3 mb-3">
                                <div className="w-3 h-3 bg-blue-500 rounded-full animate-pulse"></div>
                                <span className="text-sm text-gray-600">Preparing AI models</span>
                            </div>
                            <div className="flex items-center gap-3 mb-3">
                                <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse" style={{animationDelay: '0.2s'}}></div>
                                <span className="text-sm text-gray-600">Loading voice synthesis</span>
                            </div>
                            <div className="flex items-center gap-3">
                                <div className="w-3 h-3 bg-purple-500 rounded-full animate-pulse" style={{animationDelay: '0.4s'}}></div>
                                <span className="text-sm text-gray-600">Initializing chat history</span>
                            </div>
                        </div>
                        
                        <p className="text-xs text-gray-500 mt-4">
                            This may take a moment for the first interaction...
                        </p>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="flex flex-col h-screen bg-gray-50">
            {/* Header */}
            <div className="bg-white shadow-sm p-4 flex items-center">
                <button
                    onClick={() => navigate("/character-selection")}
                    className="text-gray-600 hover:text-gray-800 mr-4"
                >
                    ← Back
                </button>
                <h1 className="text-xl font-semibold flex-1">
                    Talk to {characterName}
                </h1>
                
                {/* Audio Controls */}
                <div className="flex items-center space-x-2">
                    
                    <button
                        onClick={clearChatHistory}
                        className="p-2 rounded-full bg-gray-100 text-gray-600 hover:bg-red-100 hover:text-red-600"
                        title="Clear chat history"
                    >
                        <Trash2 size={20} />
                    </button>
                    
                    <button
                        onClick={() => setAutoPlayAudio(!autoPlayAudio)}
                        className={`p-2 rounded-full ${
                            autoPlayAudio 
                                ? "bg-green-100 text-green-600" 
                                : "bg-gray-100 text-gray-600"
                        }`}
                        title={autoPlayAudio ? "Auto-play enabled" : "Auto-play disabled"}
                    >
                        {autoPlayAudio ? <Volume2 size={20} /> : <VolumeX size={20} />}
                    </button>
                    
                    {(isLoading || isLoadingHistory) && (
                        <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-black"></div>
                    )}
                </div>
            </div>

            {/* Error message */}
            {error && (
                <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-2 text-sm">
                    {error}
                    <button 
                        onClick={() => setError(null)}
                        className="ml-2 text-red-800 hover:text-red-900"
                    >
                        ×
                    </button>
                </div>
            )}

            {/* Chat messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {isLoadingHistory ? (
                    <div className="text-center text-gray-500 mt-8">
                        <p>Loading chat history...</p>
                    </div>
                ) : messages.length === 0 ? (
                    <div className="text-center text-gray-500 mt-8">
                        <p>Start a conversation with {characterName}!</p>
                        <div className="text-sm mt-2 space-y-1">
                            <p>
                                {autoPlayAudio ? "🔊 Audio responses will play automatically" : "🔇 Click the play button to hear responses"}
                            </p>
                        </div>
                    </div>
                ) : (
                    messages.map((message, index) => (
                        <div
                            key={index}
                            className={`flex ${
                                message.isUser ? "justify-end" : "justify-start"
                            }`}
                        >
                            <div
                                className={`max-w-[70%] rounded-lg p-3 ${
                                    message.isUser
                                        ? "bg-black text-white"
                                        : "bg-white shadow-sm"
                                }`}
                            >
                                <p>{message.text}</p>
                                <div className="flex items-center justify-between mt-2">
                                    <span className="text-xs opacity-70">
                                        {message.timestamp.toLocaleTimeString([], {
                                            hour: "2-digit",
                                            minute: "2-digit",
                                        })}
                                    </span>
                                    {message.audioBase64 && !message.isUser && (
                                        <div className="flex items-center space-x-2">
                                            {isPlaying && playingMessageIndex === index && (
                                                <div className="flex items-center space-x-1">
                                                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                                                    <span className="text-xs text-green-600">Playing</span>
                                                </div>
                                            )}
                                            <button
                                                onClick={() => toggleAudioPlayback(message.audioBase64!, index)}
                                                className={`p-1 rounded-full transition-colors ${
                                                    isPlaying && playingMessageIndex === index
                                                        ? "bg-red-100 hover:bg-red-200 text-red-600"
                                                        : "bg-blue-100 hover:bg-blue-200 text-blue-600"
                                                }`}
                                                title={
                                                    isPlaying && playingMessageIndex === index 
                                                        ? "Pause audio" 
                                                        : "Play audio"
                                                }
                                            >
                                                {isPlaying && playingMessageIndex === index ? (
                                                    <Pause size={16} />
                                                ) : (
                                                    <Play size={16} />
                                                )}
                                            </button>
                                        </div>
                                    )}
                                </div>
                                {message.knowledgeReferences && (
                                    <KnowledgeReferences references={message.knowledgeReferences} messageIndex={index} />
                                )}
                            </div>
                        </div>
                    ))
                )}
            </div>

            {/* Input form */}
            <form
                onSubmit={handleSendMessage}
                className="bg-white border-t p-4"
            >
                <div className="flex space-x-4">
                    <input
                        type="text"
                        value={inputMessage}
                        onChange={(e) => setInputMessage(e.target.value)}
                        placeholder="Type your message..."
                        className="flex-1 border rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-black"
                        disabled={isLoading}
                    />
                    <button
                        type="button"
                        onClick={handleVoiceToggle}
                        className={`p-2 rounded-full ${
                            isListening 
                                ? "bg-red-500 text-white" 
                                : "bg-gray-100 text-gray-600 hover:bg-gray-200"
                        }`}
                        title={isListening ? "Stop listening" : "Start listening"}
                        disabled={isLoading}
                    >
                        {isListening ? <MicOff size={20} /> : <Mic size={20} />}
                    </button>
                    <button
                        type="submit"
                        className="bg-black text-white px-6 py-2 rounded-lg hover:bg-gray-800 transition-colors disabled:bg-gray-400"
                        disabled={isLoading || !inputMessage.trim()}
                    >
                        {isLoading ? "..." : "Send"}
                    </button>
                </div>
            </form>
        </div>
    );
};

export default Chatbot;
