import React from 'react';
import { Edit, Trash2, Mic } from 'lucide-react';

interface CharacterCardProps {
  name: string;
  image: string;
  id?: number | string;
  isAddButton?: boolean;
  onClick?: () => void;
  onEdit?: () => void;
  onDelete?: () => void;
  onVoiceChat?: () => void;
}

const CharacterCard: React.FC<CharacterCardProps> = ({ 
  name, 
  image, 
  id,
  isAddButton = false, 
  onClick, 
  onEdit, 
  onDelete,
  onVoiceChat
}) => {
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
    <div className="relative aspect-[2/3] rounded-md overflow-hidden group">
      <img 
        src={image} 
        alt={name}
        className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
      />
      <div className="absolute inset-0 bg-gradient-to-t from-black/70 to-transparent">
        <h3 className="absolute bottom-4 left-4 text-white text-xl font-semibold">{name}</h3>
      
      </div>
      
      {/* Character ID badge */}
      {id !== undefined && (
        <div className="absolute top-2 left-2 bg-black/60 backdrop-blur-sm text-white text-xs px-2 py-1 rounded-full font-medium z-10">
          ID: {id}
        </div>
      )}
      
      {/* Click area for main action - positioned behind buttons */}
      <div 
        onClick={onClick}
        className="absolute inset-0 cursor-pointer"
      />
      
      {/* Hover overlay for better button visibility */}
      <div className="absolute inset-0 bg-black/20 opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none" />
      
      {/* Action buttons overlay - positioned above click area */}
      <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex gap-2 z-10">
        {onVoiceChat && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              e.preventDefault();
              onVoiceChat();
            }}
            className="p-2 bg-purple-500 text-white rounded-full hover:bg-purple-600 transition-colors shadow-lg relative z-20"
            title="Voice chat"
          >
            <Mic size={16} />
          </button>
        )}
        {onEdit && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              e.preventDefault();
              onEdit();
            }}
            className="p-2 bg-blue-500 text-white rounded-full hover:bg-blue-600 transition-colors shadow-lg relative z-20"
            title="Edit character"
          >
            <Edit size={16} />
          </button>
        )}
        {onDelete && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              e.preventDefault();
              onDelete();
            }}
            className="p-2 bg-red-500 text-white rounded-full hover:bg-red-600 transition-colors shadow-lg relative z-20"
            title="Delete character"
          >
            <Trash2 size={16} />
          </button>
        )}
      </div>
    </div>
  );
};

export default CharacterCard; 