import { useEffect } from 'react';
import { useChat } from '../../hooks/useChat';
import TopBar from '../layout/TopBar';
import MessageList from './MessageList';
import InputBar from '../controls/InputBar';
import QuickReplyBar from '../controls/QuickReplyBar';

export default function ChatPage({ 
  session, 
  onSessionUpdate 
}) {
  const { messages, setMessages, isStreaming, sendMessage } = useChat({
    sessionId: session.id,
    model: session.model,
    promptId: session.promptId,
    multiDoc: session.multiDoc !== false,  // default true
    onSessionUpdate
  });

  // Load messages when session changes
  useEffect(() => {
    setMessages(session.messages || []);
  }, [session.id, session.messages, setMessages]);

  const handleModelChange = (newModel) => {
    onSessionUpdate({ model: newModel });
  };

  const handlePromptChange = (newPromptId) => {
    onSessionUpdate({ promptId: newPromptId });
  };

  const handleMultiDocChange = (enabled) => {
    onSessionUpdate({ multiDoc: enabled });
  };

  return (
    <div className="main-area">
      <TopBar 
        model={session.model} 
        onModelChange={handleModelChange}
        promptId={session.promptId}
        onPromptChange={handlePromptChange}
        multiDoc={session.multiDoc !== false}
        onMultiDocChange={handleMultiDocChange}
        disabled={isStreaming}
      />
      
      <MessageList messages={messages} />
      
      <div className="input-area-inner" style={{ padding: '0 20px', width: '100%', maxWidth: 800, margin: '0 auto' }}>
        {messages.length === 0 && (
          <QuickReplyBar onSelect={sendMessage} disabled={isStreaming} />
        )}
      </div>
      
      <InputBar onSend={sendMessage} disabled={isStreaming} />
    </div>
  );
}
