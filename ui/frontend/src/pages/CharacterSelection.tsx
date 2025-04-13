import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import CharacterCard from '../components/CharacterCard';
import AddCharacterModal from '../components/AddCharacterModal';

interface Character {
  name: string;
  image: string;
}

function CharacterSelection() {
  const navigate = useNavigate();
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [characters, setCharacters] = useState<Character[]>([
    {
      name: "J. Robert Oppenheimer",
      image: "/public/images/oppenheimer.jpg"
    },
  ]);

  const handleAddCharacter = () => {
    setIsModalOpen(true);
  };

  const handleSubmitCharacter = (name: string, imageFile: File) => {
    const imageUrl = URL.createObjectURL(imageFile);
    
    setCharacters([...characters, {
      name,
      image: imageUrl
    }]);
    
    setIsModalOpen(false);
    // Navigate to training data upload after character creation
    navigate('/training-upload');
  };

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold text-center mb-8 text-black">Character Selection</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {characters.map((character, index) => (
          <CharacterCard
            key={index}
            name={character.name}
            image={character.image}
            onClick={() => console.log(`${character.name} selected`)}
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
