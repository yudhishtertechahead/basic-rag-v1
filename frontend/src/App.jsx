import { useEffect, useState } from 'react';
import Sidebar from './components/layout/Sidebar';
import ChatPage from './components/chat/ChatPage';
import { useSessions } from './hooks/useSessions';

export default function App() {
  const {
    sessions,
    activeId,
    activeSession,
    createSession,
    switchSession,
    updateSession,
    deleteSession,
    clearAll,
  } = useSessions();

  // Theme Management
  const [theme, setTheme] = useState(() => localStorage.getItem('theme') || 'dark');
  
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggleTheme = () => setTheme(prev => (prev === 'dark' ? 'light' : 'dark'));

  // If no sessions exist, auto-create one
  useEffect(() => {
    if (sessions.length === 0) {
      createSession();
    }
  }, [sessions.length, createSession]);

  const handleSessionUpdate = (patch) => {
    if (activeId) updateSession(activeId, patch);
  };

  return (
    <div className="app-layout">
      <Sidebar 
        sessions={sessions}
        activeId={activeId}
        onNew={() => createSession()}
        onSwitch={switchSession}
        onDelete={deleteSession}
        onClear={clearAll}
        theme={theme}
        toggleTheme={toggleTheme}
      />
      
      {activeSession ? (
        <ChatPage 
          key={activeSession.id} // Force remount on session switch
          session={activeSession}
          onSessionUpdate={handleSessionUpdate}
        />
      ) : (
        <div className="main-area" style={{ alignItems: 'center', justifyContent: 'center' }}>
          <p style={{ color: 'var(--text-muted)' }}>No active chat</p>
        </div>
      )}
    </div>
  );
}
