import React from 'react';

interface Step {
  label: string;
  isActive: boolean;
}

interface UploadProgressProps {
  currentStep: number;
}

const UploadProgress: React.FC<UploadProgressProps> = ({ currentStep }) => {
  const steps: Step[] = [
    { label: 'Training Data', isActive: currentStep === 1 },
    { label: 'TTS Data', isActive: currentStep === 2 },
    { label: 'Speech Data', isActive: currentStep === 3 },
    { label: 'Style Data', isActive: currentStep === 4 }
  ];

  return (
    <div className="fixed bottom-8 left-0 right-0">
      <div className="flex justify-center items-center">
        <div className="flex items-center">
          {steps.map((step, index) => (
            <React.Fragment key={index}>
              {index > 0 && (
                <div className="w-16 h-0.5 bg-gray-300" />
              )}
              <div className="flex flex-col items-center">
                <div
                  className={`w-8 h-8 rounded-full ${
                    step.isActive ? 'bg-black' : 'bg-gray-300'
                  }`}
                />
                <span className="mt-2 text-sm text-gray-600">{step.label}</span>
              </div>
            </React.Fragment>
          ))}
        </div>
      </div>
    </div>
  );
};

export default UploadProgress; 