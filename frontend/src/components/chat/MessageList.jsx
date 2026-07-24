import { useEffect, useRef } from 'react';
import MessageBubble from './MessageBubble';
import { IconSparkles } from '../Icons';

export default function MessageList({ messages }) {
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon">
          <IconSparkles />
        </div>
        <h2>Hi, I'm Aria.</h2>
        <p>Your AI assistant for TechAhead HR policies.</p>
      </div>
    );
  }

  return (
    <div className="messages-area">
      <div className="messages-inner">
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}
        <div ref={endRef} style={{ height: 1 }} />
      </div>
    </div>
  );
}
