import React from 'react';

interface CharacterCardProps {
  name: string;
  image: string;
  isAddButton?: boolean;
  onClick?: () => void;
}

const CharacterCard: React.FC<CharacterCardProps> = ({ name, image, isAddButton = false, onClick }) => {
  if (isAddButton) {
    return (
      <div 
        onClick={onClick}
        className="relative aspect-[2/3] rounded-md bg-gray-800 hover:bg-gray-700 transition-all cursor-pointer flex flex-col items-center justify-center"
      >
        <div className="text-6xl text-gray-400">+</div>
        <p className="text-gray-400 mt-2">Train New Character</p>
      </div>
    );
  }

  return (
    <div 
      onClick={onClick}
      className="relative aspect-[2/3] rounded-md overflow-hidden group cursor-pointer"
    >
      <img 
        src={image} 
        alt={name}
        className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
      />
      <div className="absolute inset-0 bg-gradient-to-t from-black/70 to-transparent">
        <h3 className="absolute bottom-4 left-4 text-white text-xl font-semibold">{name}</h3>
      </div>
    </div>
  );
};

export default CharacterCard; 