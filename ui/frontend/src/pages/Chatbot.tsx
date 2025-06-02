import React, { useEffect, useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { useVoiceToText } from "react-speakup";
import { Mic, MicOff } from "lucide-react";

interface Message {
    text: string;
    isUser: boolean;
    timestamp: Date;
}

const Chatbot = () => {
    const navigate = useNavigate();
    const location = useLocation();
    const [messages, setMessages] = useState<Message[]>([]);
    const [inputMessage, setInputMessage] = useState("");
    const [isListening, setIsListening] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Get character name from location state
    const characterName = location.state?.characterName || "Character";

    const handleSendMessage = (e: React.FormEvent) => {
        e.preventDefault();
        if (!inputMessage.trim()) return;

        // Add user message
        const userMessage: Message = {
            text: inputMessage,
            isUser: true,
            timestamp: new Date(),
        };
        setMessages([...messages, userMessage]);
        setInputMessage("");

        // Simulate bot response (replace with actual API call later)
        setTimeout(() => {
            const botMessage: Message = {
                text: `Placeholder response from ${characterName}.`,
                isUser: false,
                timestamp: new Date(),
            };
            setMessages((prev) => [...prev, botMessage]);
        }, 1000);
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
                <h1 className="text-xl font-semibold">
                    Talk to {characterName}
                </h1>
            </div>

            {/* Error message */}
            {error && (
                <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-2 text-sm">
                    {error}
                </div>
            )}

            {/* Chat messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {messages.map((message, index) => (
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
                            <span className="text-xs opacity-70 mt-1 block">
                                {message.timestamp.toLocaleTimeString([], {
                                    hour: "2-digit",
                                    minute: "2-digit",
                                })}
                            </span>
                        </div>
                    </div>
                ))}
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
                    >
                        {isListening ? <MicOff size={20} /> : <Mic size={20} />}
                    </button>
                    <button
                        type="submit"
                        className="bg-black text-white px-6 py-2 rounded-lg hover:bg-gray-800 transition-colors"
                    >
                        Send
                    </button>
                </div>
            </form>
        </div>
    );
};

export default Chatbot;
