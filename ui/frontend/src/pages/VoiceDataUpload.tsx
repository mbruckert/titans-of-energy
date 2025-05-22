import { useState } from "react";
import { Settings, ArrowLeft, ArrowRight } from "lucide-react";

function VoiceDataUpload() {
    const [selectedFile, setSelectedFile] = useState(null);
    const [selectedModel, setSelectedModel] = useState("GPT-4");

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (file) {
            setSelectedFile(file);
        }
    };
    return (
        <div className="flex flex-col items-center justify-center w-full max-w-4xl mx-auto bg-white p-8 rounded-lg shadow">
            <div className="grid grid-cols-3 gap-8 w-full mb-12">
                {/* Step 1: Upload audio */}
                <div className="flex flex-col items-center">
                    <h1 className="text-xl font-bold mb-6">Upload audio</h1>
                    <div className="w-full">
                        <p className="text-sm font-semibold mb-2">
                            Select Audio File
                        </p>
                        <div className="border border-gray-300 rounded p-2 text-sm text-gray-500 mb-2">
                            <input
                                type="file"
                                accept=".mp3,.wav"
                                className="block w-full text-sm"
                                id="audio-upload"
                                onChange={handleFileChange}
                            />
                        </div>
                        <p className="text-xs text-gray-400">
                            Allowed formats: MP3, WAV
                        </p>
                    </div>
                    <div className="bg-black text-white w-8 h-8 rounded-full flex items-center justify-center mt-20">
                        <span>1</span>
                    </div>
                </div>

                {/* Step 2: Choose Model */}
                <div className="flex flex-col items-center">
                    <h1 className="text-xl font-bold mb-6">Choose Model</h1>
                    <div className="w-full">
                        <p className="text-sm font-semibold mb-2">
                            Choose a Model
                        </p>
                        <div className="flex gap-2 mb-2">
                            <button
                                className={`border border-gray-300 rounded p-2 text-sm text-gray-500 mb-2 cursor-pointer ${
                                    selectedModel === "GPT-4"
                                        ? "bg-gray-200"
                                        : "bg-white"
                                }`}
                                onClick={() => setSelectedModel("GPT-4")}
                            >
                                GPT-4
                            </button>
                            <button
                                className={`border border-gray-300 rounded p-2 text-sm text-gray-500 mb-2 cursor-pointer ${
                                    selectedModel === "Claude"
                                        ? "bg-gray-200"
                                        : "bg-white"
                                }`}
                                onClick={() => setSelectedModel("Claude")}
                            >
                                Claude
                            </button>
                            <button
                                className={`border border-gray-300 rounded p-2 text-sm text-gray-500 mb-2 cursor-pointer ${
                                    selectedModel === "LLAMA"
                                        ? "bg-gray-200"
                                        : "bg-white"
                                }`}
                                onClick={() => setSelectedModel("LLAMA")}
                            >
                                LLAMA
                            </button>
                            <button
                                className={`border border-gray-300 rounded p-2 text-sm text-gray-500 mb-2 cursor-pointer ${
                                    selectedModel === "Custom"
                                        ? "bg-gray-200"
                                        : "bg-white"
                                }`}
                                onClick={() => setSelectedModel("Custom")}
                            >
                                +
                            </button>
                        </div>
                        <p className="text-xs text-gray-400">
                            Select a processing model to analyze your data
                        </p>
                    </div>
                    <div className="bg-black text-white w-8 h-8 rounded-full flex items-center justify-center mt-auto">
                        <span>2</span>
                    </div>
                </div>

                {/* Step 3: Adjust Settings */}
                <div className="flex flex-col items-center">
                    <h1 className="text-xl font-bold mb-6">Adjust Settings</h1>
                    <div className="w-full flex justify-center">
                        <button className="w-12 h-12 rounded-full mt-5 border border-gray-300 flex items-center justify-center hover:bg-gray-200 cursor-pointer">
                            <Settings className="w-6 h-6 text-gray-700" />
                        </button>
                    </div>
                    <div className="bg-black text-white w-8 h-8 rounded-full flex items-center justify-center mt-auto">
                        <span>3</span>
                    </div>
                </div>
            </div>

            {/* Bottom navigation */}
            <div className="flex items-center justify-center gap-4">
                <button className="w-8 h-8 rounded-full border flex items-center justify-center">
                    <ArrowLeft className="w-4 h-4" />
                </button>
                <button className="bg-blue-400 text-white px-6 py-2 rounded">
                    Create
                </button>
                <button className="w-8 h-8 rounded-full border flex items-center justify-center">
                    <ArrowRight className="w-4 h-4" />
                </button>
            </div>
        </div>
    );
}

export default VoiceDataUpload;
