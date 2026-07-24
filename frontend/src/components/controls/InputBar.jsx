import { useState, useRef, useEffect } from 'react';
import { IconSend } from '../Icons';

export default function InputBar({ onSend, disabled }) {
  const [text, setText] = useState('');
  const textareaRef = useRef(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = '24px';
      const scrollHeight = textareaRef.current.scrollHeight;
      textareaRef.current.style.height = `${Math.min(scrollHeight, 200)}px`;
    }
  }, [text]);

  const handleSend = () => {
    if (!text.trim() || disabled) return;
    onSend(text);
    setText('');
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="input-area-wrapper">
      <div className="input-area-inner">
        <div className="input-bar">
          <textarea
            ref={textareaRef}
            placeholder="Ask Aria about HR policies..."
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={disabled}
            rows={1}
          />
          <button 
            className="btn-send" 
            onClick={handleSend} 
            disabled={disabled || !text.trim()}
            title="Send message"
          >
            <IconSend />
          </button>
        </div>
        <div className="input-hint">
          Press Enter to send, Shift+Enter for new line
        </div>
      </div>
    </div>
  );
}
