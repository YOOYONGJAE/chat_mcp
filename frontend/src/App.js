import React, { useState, useRef, useEffect } from 'react';
import Eyes from './Eyes';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isBotTyping, setIsBotTyping] = useState(false);
  const [isUserTyping, setIsUserTyping] = useState(false);
  const [eyeGaze, setEyeGaze] = useState('default');
  const messagesEndRef = useRef(null);
  const idleTimerRef = useRef(null);
  const typingTimerRef = useRef(null);
  const isSleepingRef = useRef(false);
  const inputRef = useRef(null);
  const gazeIntervalRef = useRef(null);

  // Function to start the timer that makes the eyes sleep
  const startIdleTimer = () => {
    clearTimeout(idleTimerRef.current);
    idleTimerRef.current = setTimeout(() => {
      setEyeGaze('sleeping');
      isSleepingRef.current = true;
    }, 15000); // 15 seconds
  };

  // Welcome message
  useEffect(() => {
    startIdleTimer(); // Start idle timer on initial load
    const welcomeMessage = {
      text: '안녕하세요. 엔큐브 챗봇 다큐브입니다.\n다큐브는 지금 공부중이라 내용이 조금 틀릴 수 있어요.\n제공되는 정보가 궁금하시면 "정보" 라고 입력해주세요.',
      sender: 'bot'
    };
    setMessages([welcomeMessage]);
  }, []);

  // Scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isBotTyping, isUserTyping]);

  // Cleanup timers
  useEffect(() => {
    return () => {
      clearTimeout(idleTimerRef.current);
      clearTimeout(typingTimerRef.current);
      if (gazeIntervalRef.current) {
        clearInterval(gazeIntervalRef.current);
      }
    };
  }, []);

  // Eye animation effect when bot is typing
  useEffect(() => {
    if (isBotTyping) {
      clearTimeout(idleTimerRef.current); // Don't sleep while bot is typing
      const gazeSequence = ['top-left', 'top-right'];
      let currentIndex = 0;
      setEyeGaze(gazeSequence[currentIndex]);
      gazeIntervalRef.current = setInterval(() => {
        currentIndex = (currentIndex + 1) % gazeSequence.length;
        setEyeGaze(gazeSequence[currentIndex]);
      }, 700);
    } else {
      if (gazeIntervalRef.current) {
        clearInterval(gazeIntervalRef.current);
        gazeIntervalRef.current = null;
      }
    }
    return () => {
      if (gazeIntervalRef.current) {
        clearInterval(gazeIntervalRef.current);
      }
    };
  }, [isBotTyping]);

  const handleFocus = () => {
    clearTimeout(idleTimerRef.current);
    if (isSleepingRef.current) {
      isSleepingRef.current = false;
      const wakeUpSequence = async () => {
        for (let i = 0; i < 2; i++) {
          setEyeGaze('sleeping');
          await new Promise(resolve => setTimeout(resolve, 200));
          setEyeGaze('default');
          await new Promise(resolve => setTimeout(resolve, 200));
        }
        setEyeGaze('typing');
      };
      wakeUpSequence();
    } else {
      setEyeGaze('typing');
    }
  };

  const handleBlur = () => {
    setEyeGaze('default');
    startIdleTimer(); // Start sleep timer on blur
  };

  const handleInputChange = (e) => {
    const value = e.target.value;
    setInput(value);

    clearTimeout(idleTimerRef.current); // Clear sleep timer on activity
    if (typingTimerRef.current) {
      clearTimeout(typingTimerRef.current);
    }

    if (value.trim() !== '') {
      setIsUserTyping(true);
      setEyeGaze('typing');
    } else {
      setIsUserTyping(false);
      setEyeGaze('default');
      startIdleTimer(); // Start sleep timer if input is cleared
      return;
    }

    typingTimerRef.current = setTimeout(() => {
      setIsUserTyping(false);
      setEyeGaze('default');
      startIdleTimer(); // Start sleep timer when user stops typing
    }, 1500);
  };

  const sendMessage = async (e) => {
    e.preventDefault();
    const trimmedInput = input.trim();
    if (trimmedInput === '') return;

    clearTimeout(idleTimerRef.current);
    setIsUserTyping(false);
    clearTimeout(typingTimerRef.current);
    const userMessage = { text: trimmedInput, sender: 'user' };
    setMessages((prevMessages) => [...prevMessages, userMessage]);
    setInput('');
    
    setIsBotTyping(true);

    if (trimmedInput === "정보") {
      await new Promise(resolve => setTimeout(resolve, 500));
      const infoMessage = {
        text: '현재 회사 소개와 회사 주소, 이메일, 전화번호, 창립일, 그리고 회사에서 하는 일 대해 설명해드릴 수 있어요.\n무엇이 궁금하신가요?',
        sender: 'bot'
      };
      
      setIsBotTyping(false);
      setEyeGaze('responding');
      setMessages((prevMessages) => [...prevMessages, infoMessage]);
      
      setTimeout(() => {
        setEyeGaze('default');
        startIdleTimer();
      }, 500);
      return;
    }

    try {
      const response = await fetch(`http://devstudio.ddns.net:4000/chat_test?question=${encodeURIComponent(trimmedInput)}`);
      const data = await response.json();
      const botMessage = { text: data.answer, sender: 'bot' };
      
      setIsBotTyping(false);
      setEyeGaze('responding');
      setMessages((prevMessages) => [...prevMessages, botMessage]);

      setTimeout(() => {
        setEyeGaze('default');
        startIdleTimer();
      }, 1000);

    } catch (error) {
      console.error("Error sending message:", error);
      const errorMessage = { text: "죄송합니다. 메시지를 보내는 데 실패했습니다.", sender: 'bot' };
      setMessages((prevMessages) => [...prevMessages, errorMessage]);
      setIsBotTyping(false);
      setEyeGaze('default');
      startIdleTimer();
    }
  };

  return (
    <div className="bg-gray-100 flex justify-center items-center min-h-screen">
      <div className="w-[450px] h-screen flex flex-col shadow-xl border-r border-l border-gray-200 bg-white">
        <div className="p-4 text-center bg-white">
          <h1 className="text-2xl font-bold text-blue-800">챗봇 다큐브에게 질문해보세요!</h1>
          <p className="text-sm text-gray-500 italic mt-1">*다큐브는 아직 공부중이므로 대답이 틀릴 수 있어요.</p>
        </div>
        <Eyes gaze={eyeGaze} />
        <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50">
          {messages.map((msg, index) => (
            <div
              key={index}
              className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`text-sm max-w-xs px-4 py-2 rounded-lg shadow-md whitespace-pre-wrap ${msg.sender === 'user'
                  ? 'bg-blue-500 text-white rounded-br-none'
                  : 'bg-white text-blue-800 rounded-bl-none'
                }`}
              >
                {msg.text}
              </div>
            </div>
          ))}

          {isUserTyping && (
            <div className="flex justify-end">
              <div className="max-w-xs px-4 py-2 rounded-lg shadow-md bg-blue-500 text-white rounded-br-none flex items-center">
                <span className="animate-pulse">사용자 작성중...</span>
              </div>
            </div>
          )}

          {isBotTyping && (
            <div className="flex justify-start">
              <div className="max-w-xs px-4 py-2 rounded-lg shadow-md bg-white text-blue-800 rounded-bl-none flex items-center">
                <div className="loader mr-2"></div>
                <span>응답 대기중...</span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
        <form onSubmit={sendMessage} className="p-4 bg-white border-t border-gray-200 flex">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={handleInputChange}
            onFocus={handleFocus}
            onBlur={handleBlur}
            className="flex-1 p-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="메시지를 입력하세요..."
          />
          <button
            type="submit"
            className="ml-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            보내기
          </button>
        </form>
      </div>
    </div>
  );
}

export default App;
