import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import ModelSelection from './pages/ModelSelection';
import CharacterSelection from './pages/CharacterSelection';
import KnowledgeBaseUpload from './pages/KnowledgeBaseUpload';
import VoiceCloningUpload from './pages/VoiceCloningUpload';
import StyleTuningConfig from './pages/StyleTuningConfig';
import TTSConfig from './pages/TTSConfig';
import Chatbot from './pages/Chatbot';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<CharacterSelection />} />
        <Route path="/character-selection" element={<CharacterSelection />} />
        <Route path="/model-selection" element={<ModelSelection />} />
        <Route path="/knowledge-base" element={<KnowledgeBaseUpload />} />
        <Route path="/voice-cloning" element={<VoiceCloningUpload />} />
        <Route path="/style-tuning" element={<StyleTuningConfig />} />
        <Route path="/tts-config" element={<TTSConfig />} />
        <Route path="/chatbot" element={<Chatbot />} />
      </Routes>
    </Router>
  );
}

export default App;
