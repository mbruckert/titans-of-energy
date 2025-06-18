import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import CharacterCard from '../components/CharacterCard';
import AddCharacterModal from '../components/AddCharacterModal';
import { API_BASE_URL, API_ENDPOINTS } from '../config/api';

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
  const [characters, setCharacters] = useState<Character[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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
    setIsModalOpen(true);
  };

  const handleSubmitCharacter = (name: string, imageFile: File) => {
    // Initialize global file storage if it doesn't exist
    if (!window.characterCreationFiles) {
      window.characterCreationFiles = {};
    }
    
    // Store the image file in global storage
    window.characterCreationFiles.imageFile = imageFile;
    
    // Store character data in session storage (without File objects)
    sessionStorage.setItem('newCharacterData', JSON.stringify({
      name
    }));
    
    console.log('Character image stored for creation:', imageFile.name);
    
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
    </div>
  );
}

export default CharacterSelection;
