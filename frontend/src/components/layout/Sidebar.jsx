import { useEffect, useState } from 'react';
import { fetchHealth } from '../../api/ariaApi';
import { IconSparkles, IconPlus, IconMessage, IconTrash, IconSun, IconMoon } from '../Icons';

export default function Sidebar({ sessions, activeId, onNew, onSwitch, onDelete, onClear, theme, toggleTheme }) {
  const [status, setStatus] = useState('connecting');

  useEffect(() => {
    let mounted = true;
    const check = async () => {
      try {
        await fetchHealth();
        if (mounted) setStatus('online');
      } catch {
        if (mounted) setStatus('offline');
      }
    };
    check();
    const interval = setInterval(check, 30000);
    return () => { mounted = false; clearInterval(interval); };
  }, []);

  const now = Date.now();
  const DAY = 24 * 60 * 60 * 1000;
  const today = sessions.filter(s => now - s.createdAt < DAY);
  const older = sessions.filter(s => now - s.createdAt >= DAY);

  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-logo">
          <div className="sidebar-logo-icon">
            <IconSparkles />
          </div>
          <div className="sidebar-logo-text">
            <span className="sidebar-logo-name">Aria</span>
            <span className="sidebar-logo-sub">TechAhead HR</span>
          </div>
          <button 
            className="btn-action" 
            onClick={toggleTheme} 
            title="Toggle theme"
            style={{ marginLeft: 'auto' }}
          >
            {theme === 'dark' ? <IconSun /> : <IconMoon />}
          </button>
        </div>
        <button className="btn-new-chat" onClick={onNew}>
          New Chat
          <IconPlus />
        </button>
      </div>

      <div className="sidebar-sessions">
        {today.length > 0 && <div className="sessions-group-label">Today</div>}
        {today.map(s => (
          <SessionItem 
            key={s.id} session={s} isActive={s.id === activeId} 
            onClick={() => onSwitch(s.id)} 
            onDelete={(e) => { e.stopPropagation(); onDelete(s.id); }} 
          />
        ))}

        {older.length > 0 && <div className="sessions-group-label">Older</div>}
        {older.map(s => (
          <SessionItem 
            key={s.id} session={s} isActive={s.id === activeId} 
            onClick={() => onSwitch(s.id)} 
            onDelete={(e) => { e.stopPropagation(); onDelete(s.id); }} 
          />
        ))}
      </div>

      <div className="sidebar-footer">
        <div className="status-badge" title={`Backend is ${status}`}>
          <div className={`status-dot ${status}`} />
          {status === 'connecting' ? 'Connecting...' : status === 'online' ? 'System Online' : 'System Offline'}
        </div>
        {sessions.length > 0 && (
          <button className="btn-clear-all" onClick={onClear}>
            Clear all chats
          </button>
        )}
      </div>
    </div>
  );
}

function SessionItem({ session, isActive, onClick, onDelete }) {
  return (
    <div className={`session-item ${isActive ? 'active' : ''}`} onClick={onClick}>
      <IconMessage className="icon-msg" />
      <div className="session-item-title">{session.title || 'New Chat'}</div>
      <button className="session-delete-btn" onClick={onDelete} title="Delete chat">
        <IconTrash />
      </button>
    </div>
  );
}
