import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import CharacterCard from '../components/CharacterCard';
import AddCharacterModal from '../components/AddCharacterModal';

interface Character {
  id: number;
  name: string;
  image_path?: string;
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
      const response = await fetch('http://localhost:5000/characters');
      if (!response.ok) {
        throw new Error('Failed to fetch characters');
      }
      const data = await response.json();
      setCharacters(data);
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
    const imageUrl = URL.createObjectURL(imageFile);
    
    setCharacters([...characters, {
      id: characters.length + 1, // Temporary ID until backend integration
      name,
      image_path: imageUrl
    }]);
    
    setIsModalOpen(false);
    navigate('/model-selection');
  };

  const handleCharacterClick = (character: Character) => {
    navigate('/chatbot', { state: { characterName: character.name } });
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
            image={character.image_path || '/public/images/default-character.jpg'}
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
