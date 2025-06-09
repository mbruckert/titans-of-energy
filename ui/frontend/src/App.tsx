import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import CharacterSelection from './pages/CharacterSelection';
import TrainingDataUpload from './pages/TrainingDataUpload';
import TTSDataUpload from './pages/TTSDataUpload';
import SpeechDataUpload from './pages/SpeechDataUpload';
import StyleDataUpload from './pages/StyleDataUpload';
import VoiceDataUpload from './pages/VoiceDataUpload';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<CharacterSelection />} />
        <Route path="/character-selection" element={<CharacterSelection />} />
        <Route path="/training-upload" element={<TrainingDataUpload />} />
        <Route path="/tts-upload" element={<TTSDataUpload />} />
        <Route path="/speech-upload" element={<SpeechDataUpload />} />
        <Route path="/style-upload" element={<StyleDataUpload />} />
        <Route path="/voice-upload" element={<VoiceDataUpload />} />
      </Routes>
    </Router>
  );
}

export default App;
