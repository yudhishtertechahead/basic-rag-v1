import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import TypingIndicator from './TypingIndicator';
import SourceChips from './SourceChips';
import ContextPanel from './ContextPanel';
import { useState } from 'react';
import { IconCopy, IconUser, IconSparkles } from '../Icons';

export default function MessageBubble({ message }) {
  const [copied, setCopied] = useState(false);
  const isUser = message.role === 'user';

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className={`message-row ${isUser ? 'user' : ''}`}>
      <div className={`avatar ${isUser ? 'user' : 'bot'}`}>
        {isUser ? <IconUser /> : <IconSparkles />}
      </div>
      
      <div className="bubble-wrap">
        <div className={`bubble ${isUser ? 'user' : 'bot'}`}>
          {isUser ? (
            <div style={{ whiteSpace: 'pre-wrap' }}>{message.content}</div>
          ) : message.content ? (
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {message.content}
            </ReactMarkdown>
          ) : message.isStreaming ? (
            <TypingIndicator />
          ) : (
            <div style={{ color: 'var(--error)' }}>Empty response</div>
          )}
        </div>

        {!isUser && (
          <>
            {message.content && !message.isStreaming && (
              <div className="bubble-actions">
                <button className="btn-action" onClick={handleCopy} title="Copy answer">
                  <IconCopy />
                </button>
              </div>
            )}
            {message.context?.length > 0 && (
              <>
                <SourceChips context={message.context} />
                <ContextPanel chunks={message.context} />
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}
