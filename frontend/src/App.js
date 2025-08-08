import React, { useState, useRef, useEffect } from 'react';
import Eyes from './Eyes';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isBotTyping, setIsBotTyping] = useState(false); // 명확성을 위해 isTyping -> isBotTyping으로 변경
  const [isUserTyping, setIsUserTyping] = useState(false); // ✅ 사용자 타이핑 상태
  const [eyeGaze, setEyeGaze] = useState('default');
  const messagesEndRef = useRef(null);
  const idleTimerRef = useRef(null);
  const typingTimerRef = useRef(null); // ✅ 사용자 타이핑 타이머 Ref
  const isSleepingRef = useRef(false);
  const inputRef = useRef(null); // ✅ input 엘리먼트 참조

  // ✅ 컴포넌트가 처음 마운트될 때 환영 메시지 설정
  useEffect(() => {
    const welcomeMessage = {
      text: '안녕하세요. 엔큐브 챗봇 다큐브입니다.\n다큐브는 지금 공부중이라 내용이 조금 틀릴 수 있어요.\n제공되는 정보가 궁금하시면 "정보" 라고 입력해주세요.',
      sender: 'bot'
    };
    setMessages([welcomeMessage]);
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  // 메시지나 타이핑 인디케이터가 추가될 때 스크롤
  useEffect(() => {
    scrollToBottom();
  }, [messages, isBotTyping, isUserTyping]);

  // 컴포넌트 언마운트 시 모든 타이머 정리
  useEffect(() => {
    return () => {
      clearTimeout(idleTimerRef.current);
      clearTimeout(typingTimerRef.current);
    };
  }, []);

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
    idleTimerRef.current = setTimeout(() => {
      setEyeGaze('sleeping');
      isSleepingRef.current = true;
    }, 15000);
  };

  // ✅ 사용자가 입력할 때 호출되는 함수
  const handleInputChange = (e) => {
    const value = e.target.value;
    setInput(value);

    if (typingTimerRef.current) {
      clearTimeout(typingTimerRef.current);
    }

    if (value.trim() !== '') {
      setIsUserTyping(true);
    } else {
      setIsUserTyping(false);
      return;
    }

    // 1.5초 동안 추가 입력이 없으면 타이핑 멈춤으로 간주
    typingTimerRef.current = setTimeout(() => {
      setIsUserTyping(false);
    }, 1500);
  };

  const sendMessage = async (e) => {
    e.preventDefault();
    const trimmedInput = input.trim();
    if (trimmedInput === '') return;

    // 메시지 전송 시 사용자 타이핑 중단 및 UI 업데이트
    setIsUserTyping(false);
    clearTimeout(typingTimerRef.current);
    const userMessage = { text: trimmedInput, sender: 'user' };
    setMessages((prevMessages) => [...prevMessages, userMessage]);
    setInput('');

    // "정보" 키워드 프론트엔드에서 처리
    if (trimmedInput === "정보") {
      setIsBotTyping(true);
      setEyeGaze('responding');

      // 자연스러운 응답을 위해 잠시 대기
      await new Promise(resolve => setTimeout(resolve, 500));

      const infoMessage = {
        text: '현재 회사 소개와 회사 주소, 이메일, 전화번호, 창립일, 그리고 회사에서 하는 일 대해 설명해드릴 수 있어요.\n무엇이 궁금하신가요?',
        sender: 'bot'
      };
      setMessages((prevMessages) => [...prevMessages, infoMessage]);
      
      setIsBotTyping(false);
      setTimeout(() => {
        if (document.activeElement === inputRef.current) {
          setEyeGaze('typing');
        } else {
          setEyeGaze('default');
        }
      }, 500);
      return; // 백엔드 호출 방지
    }

    // 그 외의 경우, 기존 백엔드 호출 로직 실행
    setIsBotTyping(true);
    setEyeGaze('default');

    try {
      const response = await fetch(`http://devstudio.ddns.net:4000/chat_test?question=${encodeURIComponent(trimmedInput)}`);
      const data = await response.json();
      const botMessage = { text: data.answer, sender: 'bot' };
      
      setEyeGaze('responding');
      setMessages((prevMessages) => [...prevMessages, botMessage]);

      // 응답 애니메이션 후 포커스 상태에 따라 눈동자 변경
      setTimeout(() => {
        if (document.activeElement === inputRef.current) {
          setEyeGaze('typing');
        } else {
          setEyeGaze('default');
        }
      }, 2000);

    } catch (error) {
      console.error("Error sending message:", error);
      const errorMessage = { text: "죄송합니다. 메시지를 보내는 데 실패했습니다.", sender: 'bot' };
      setMessages((prevMessages) => [...prevMessages, errorMessage]);
    } finally {
      setIsBotTyping(false);
    }
  };

  return (
    <div className="bg-gray-100 flex justify-center items-center min-h-screen">
      <div className="w-1/2 h-screen flex flex-col shadow-xl border-r border-l border-gray-200 bg-white">
        {/* ✅ 추가된 헤더 텍스트 */}
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
                className={`max-w-xs px-4 py-2 rounded-lg shadow-md whitespace-pre-wrap ${msg.sender === 'user'
                  ? 'bg-blue-500 text-white rounded-br-none'
                  : 'bg-white text-blue-800 rounded-bl-none'
                }`}
              >
                {msg.text}
              </div>
            </div>
          ))}

          {/* ✅ 사용자 타이핑 말풍선 */}
          {isUserTyping && (
            <div className="flex justify-end">
              <div className="max-w-xs px-4 py-2 rounded-lg shadow-md bg-blue-500 text-white rounded-br-none flex items-center">
                <span className="animate-pulse">사용자 작성중...</span>
              </div>
            </div>
          )}

          {/* 봇 응답 대기중 말풍선 */}
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
            ref={inputRef} // ✅ ref 할당
            type="text"
            value={input}
            onChange={handleInputChange} // ✅ 수정된 핸들러 사용
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
