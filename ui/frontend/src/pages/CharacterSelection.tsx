import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import CharacterCard from '../components/CharacterCard';
import AddCharacterModal from '../components/AddCharacterModal';
import EditCharacterModal from '../components/EditCharacterModal';
import { API_BASE_URL, API_ENDPOINTS } from '../config/api';
import { 
  initializeCharacterCreation, 
  isCharacterCreationInProgress, 
  getCharacterCreationProgress,
  saveCharacterCreationData,
  saveCharacterCreationFiles
} from '../utils/characterCreation';

interface Character {
  id: number;
  name: string;
  image_base64?: string;
  llm_model?: string;
  created_at?: string;
}

function CharacterSelection() {
  const navigate = useNavigate();
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [selectedCharacter, setSelectedCharacter] = useState<Character | null>(null);
  const [characters, setCharacters] = useState<Character[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showProgressWarning, setShowProgressWarning] = useState(false);

  useEffect(() => {
    fetchCharacters();
  }, []);

  const fetchCharacters = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.GET_CHARACTERS}`);
      if (!response.ok) {
        throw new Error('Failed to fetch characters');
      }
      const data = await response.json();
      
      // Handle the API response structure
      if (data.status === 'success' && data.characters) {
        setCharacters(data.characters);
      } else {
        throw new Error(data.error || 'Failed to fetch characters');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const handleAddCharacter = () => {
    // Check if there's existing progress
    if (isCharacterCreationInProgress()) {
      setShowProgressWarning(true);
    } else {
      startNewCharacterCreation();
    }
  };

  const startNewCharacterCreation = () => {
    initializeCharacterCreation();
    setIsModalOpen(true);
    setShowProgressWarning(false);
  };

  const continueExistingCreation = () => {
    const progress = getCharacterCreationProgress();
    setShowProgressWarning(false);
    
    // Navigate to the appropriate step based on progress
    if (!progress.hasBasicInfo) {
      setIsModalOpen(true);
    } else if (!progress.hasModel) {
      navigate('/model-selection');
    } else if (!progress.hasKnowledgeBase) {
      navigate('/knowledge-base');
    } else if (!progress.hasVoiceCloning) {
      navigate('/voice-cloning');
    } else {
      navigate('/style-tuning');
    }
  };

  const handleSubmitCharacter = (name: string, imageFile: File, wakeword: string) => {
    // Save character data and files using utility functions
    saveCharacterCreationData({ name, wakeword });
    saveCharacterCreationFiles({ imageFile });
    
    console.log('Character image stored for creation:', imageFile.name);
    console.log('Character wakeword:', wakeword);
    
    setIsModalOpen(false);
    navigate('/model-selection');
  };

  const handleCharacterClick = (character: Character) => {
    navigate('/chatbot', { 
      state: { 
        characterId: character.id,
        characterName: character.name 
      } 
    });
  };

  const handleVoiceChat = (character: Character) => {
    navigate('/voice-interaction', { 
      state: { 
        characterId: character.id,
        characterName: character.name,
        characterImage: character.image_base64 
      } 
    });
  };

  const handleEditCharacter = (character: Character) => {
    setSelectedCharacter(character);
    setIsEditModalOpen(true);
  };

  const handleDeleteCharacter = async (character: Character) => {
    if (!confirm(`Are you sure you want to delete "${character.name}"? This action cannot be undone and will delete all associated data including chat history, knowledge base, and voice cloning data.`)) {
      return;
    }

    setError(null);
    
    try {
      const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.DELETE_CHARACTER}/${character.id}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to delete character');
      }

      // Show success message and refresh the character list
      console.log(`Character "${character.name}" deleted successfully`);
      fetchCharacters();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred while deleting the character');
    }
  };

  const handleEditSuccess = () => {
    setIsEditModalOpen(false);
    setSelectedCharacter(null);
    fetchCharacters(); // Refresh the character list
  };

  if (isLoading) {
    return (
      <div className="container mx-auto p-4">
        <div className="flex justify-center items-center h-64">
          <p className="text-xl text-gray-600">Loading characters...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="container mx-auto p-4">
        <div className="flex justify-center items-center h-64">
          <p className="text-xl text-red-600">Error: {error}</p>
          <button 
            onClick={fetchCharacters}
            className="ml-4 px-4 py-2 bg-black text-white rounded hover:bg-gray-800"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold text-center mb-8 text-black">Character Selection</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {characters.map((character) => (
          <CharacterCard
            key={character.id}
            name={character.name}
            image={character.image_base64 || '/public/images/default-character.jpg'}
            onClick={() => handleCharacterClick(character)}
            onEdit={() => handleEditCharacter(character)}
            onDelete={() => handleDeleteCharacter(character)}
            onVoiceChat={() => handleVoiceChat(character)}
          />
        ))}
        <CharacterCard
          isAddButton
          name=""
          image=""
          onClick={handleAddCharacter}
        />
      </div>

      <AddCharacterModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onSubmit={handleSubmitCharacter}
      />

      <EditCharacterModal
        isOpen={isEditModalOpen}
        character={selectedCharacter}
        onClose={() => {
          setIsEditModalOpen(false);
          setSelectedCharacter(null);
        }}
        onSuccess={handleEditSuccess}
      />

      {/* Progress Warning Modal */}
      {showProgressWarning && (
        <div className="fixed inset-0 bg-gray-800 bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full">
            <h2 className="text-xl font-bold mb-4">Character Creation In Progress</h2>
            <p className="text-gray-600 mb-6">
              You have an unfinished character creation. Would you like to continue where you left off or start a new character?
            </p>
            <div className="flex justify-end gap-3">
              <button
                onClick={() => setShowProgressWarning(false)}
                className="px-4 py-2 text-gray-600 hover:text-gray-800"
              >
                Cancel
              </button>
              <button
                onClick={startNewCharacterCreation}
                className="px-4 py-2 border border-gray-300 rounded-md hover:bg-gray-50"
              >
                Start New
              </button>
              <button
                onClick={continueExistingCreation}
                className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600"
              >
                Continue Existing
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default CharacterSelection;
