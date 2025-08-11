import React from 'react';

const Eye = ({ gaze }) => {
        const getPupilTransform = () => {
    switch (gaze) {
      case 'typing':
        return 'translate-x-6 translate-y-2'; // 오른쪽 아래
      case 'responding':
        return '-translate-x-6 translate-y-2'; // 왼쪽 아래
      case 'top-left':
        return '-translate-x-5 -translate-y-3'; // 왼쪽 위
      case 'top-right':
        return 'translate-x-5 -translate-y-3'; // 오른쪽 위
      default:
        return 'translate-x-0 translate-y-0'; // 정면
    }
  };

  const getPupilShape = () => {
    if (gaze === 'sleeping') {
      return 'w-5 h-1 bg-gray-800'; // '-' 모양
    } 
    return 'w-5 h-5 bg-gray-800 rounded-full'; // 'o' 모양
  };

  return (
    <div className="w-20 h-10 bg-white rounded-full flex justify-center items-center shadow-inner">
      <div className={`${getPupilShape()} transition-all duration-300 ease-in-out ${getPupilTransform()}`}></div>
    </div>
  );
};

const Eyes = ({ gaze }) => {
  return (
    <div className="flex justify-center items-center p-4 bg-gray-100 space-x-4">
      <Eye gaze={gaze} />
      <Eye gaze={gaze} />
    </div>
  );
};

export default Eyes;
