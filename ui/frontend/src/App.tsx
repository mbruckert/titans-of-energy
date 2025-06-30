import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import ModelSelection from './pages/ModelSelection';
import CharacterSelection from './pages/CharacterSelection';
import KnowledgeBaseUpload from './pages/KnowledgeBaseUpload';
import VoiceCloningUpload from './pages/VoiceCloningUpload';
import VoiceDataUpload from './pages/VoiceDataUpload';
import StyleTuningConfig from './pages/StyleTuningConfig';
import Chatbot from './pages/Chatbot';
import VoiceInteraction from './pages/VoiceInteraction';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<CharacterSelection />} />
        <Route path="/character-selection" element={<CharacterSelection />} />
        <Route path="/model-selection" element={<ModelSelection />} />
        <Route path="/knowledge-base" element={<KnowledgeBaseUpload />} />
        <Route path="/voice-cloning" element={<VoiceCloningUpload />} />
        <Route path="/voice-data-upload" element={<VoiceDataUpload />} />
        <Route path="/style-tuning" element={<StyleTuningConfig />} />
        <Route path="/chatbot" element={<Chatbot />} />
        <Route path="/voice-interaction" element={<VoiceInteraction />} />
      </Routes>
    </Router>
  );
}

export default App;
