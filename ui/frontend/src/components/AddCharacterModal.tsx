import React, { useState, ChangeEvent, useEffect } from 'react';
import { API_BASE_URL } from '../config/api';

interface AddCharacterModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (name: string, image: File, wakeword: string) => void;
}

const AddCharacterModal: React.FC<AddCharacterModalProps> = ({ isOpen, onClose, onSubmit }) => {
  const [name, setName] = useState('');
  const [image, setImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [wakeword, setWakeword] = useState('');
  const [nameError, setNameError] = useState('');
  const [isCheckingName, setIsCheckingName] = useState(false);

  const checkCharacterName = async (nameToCheck: string) => {
    if (!nameToCheck.trim()) {
      setNameError('');
      return;
    }

    setIsCheckingName(true);
    try {
      const response = await fetch(`${API_BASE_URL}/check-character-name`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ name: nameToCheck.trim() }),
      });

      const data = await response.json();
      
      if (response.ok) {
        if (data.exists) {
          setNameError('A character with this name already exists. Please choose a different name.');
        } else {
          setNameError('');
        }
      } else {
        console.error('Error checking character name:', data.error);
        setNameError('Unable to verify name availability. Please try again.');
      }
    } catch (error) {
      console.error('Error checking character name:', error);
      setNameError('Unable to verify name availability. Please try again.');
    } finally {
      setIsCheckingName(false);
    }
  };

  // Debounce name checking with useEffect
  useEffect(() => {
    if (!name.trim()) {
      setNameError('');
      return;
    }

    const timeoutId = setTimeout(() => {
      checkCharacterName(name);
    }, 500);

    return () => clearTimeout(timeoutId);
  }, [name]);

  const handleNameChange = (e: ChangeEvent<HTMLInputElement>) => {
    const newName = e.target.value;
    setName(newName);
    
    // Clear previous error when typing
    setNameError('');
  };

  const handleImageChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && (file.type === 'image/jpeg' || file.type === 'image/png')) {
      setImage(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    // Prevent submission if there's a name error or if we're still checking
    if (nameError || isCheckingName) {
      return;
    }
    
    if (name && image) {
      const finalWakeword = wakeword.trim() || `hey ${name.toLowerCase()}`;
      onSubmit(name, image, finalWakeword);
      handleClose();
    }
  };

  const handleClose = () => {
    setName('');
    setImage(null);
    setPreviewUrl(null);
    setWakeword('');
    setNameError('');
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 flex items-center justify-center p-4 z-50"
    style={{ backgroundColor: 'rgba(0, 0, 0, 0.5)' }}
    >
      <div className="bg-white rounded-lg p-6 max-w-md w-full">
        <h2 className="text-2xl font-bold mb-4">Train New Character</h2>
        <form onSubmit={handleSubmit}>
          <div className="mb-4">
            <label htmlFor="name" className="block text-sm font-medium text-gray-700 mb-1">
              Full Name
            </label>
            <div className="relative">
              <input
                type="text"
                id="name"
                value={name}
                onChange={handleNameChange}
                className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                  nameError ? 'border-red-500' : 'border-gray-300'
                }`}
                required
              />
              {isCheckingName && (
                <div className="absolute right-3 top-2.5">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
                </div>
              )}
            </div>
            {nameError && (
              <p className="text-red-500 text-sm mt-1">{nameError}</p>
            )}
          </div>
          
          <div className="mb-4">
            <label htmlFor="wakeword" className="block text-sm font-medium text-gray-700 mb-1">
              Wake Word (Optional)
            </label>
            <input
              type="text"
              id="wakeword"
              value={wakeword}
              onChange={(e) => setWakeword(e.target.value)}
              placeholder={`hey ${name.toLowerCase() || 'character'}`}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <p className="text-xs text-gray-500 mt-1">
              The phrase to activate voice interaction. Leave empty to use "hey {name.toLowerCase() || 'character'}"
            </p>
          </div>
          
          <div className="mb-4">
            <label htmlFor="image" className="block text-sm font-medium text-gray-700 mb-1">
              Character Image (PNG or JPG)
            </label>
            <input
              type="file"
              id="image"
              accept="image/jpeg,image/png"
              onChange={handleImageChange}
              className="w-full"
              required
            />
            {previewUrl && (
              <div className="mt-2">
                <img
                  src={previewUrl}
                  alt="Preview"
                  className="w-32 h-32 object-cover rounded-md"
                />
              </div>
            )}
          </div>

          <div className="flex justify-end gap-2">
            <button
              type="button"
              onClick={handleClose}
              className="px-4 py-2 text-gray-600 hover:text-gray-800"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!!nameError || isCheckingName || !name || !image}
              className={`px-4 py-2 rounded-md ${
                nameError || isCheckingName || !name || !image
                  ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                  : 'bg-blue-500 text-white hover:bg-blue-600'
              }`}
            >
              {isCheckingName ? 'Checking...' : 'Add Character'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default AddCharacterModal; 