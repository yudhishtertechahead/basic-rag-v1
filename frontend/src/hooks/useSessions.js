/**
 * useSessions.js
 * Manages multiple named chat sessions stored in localStorage.
 * Each session: { id, title, messages, model, promptId, createdAt }
 */

import { useState, useCallback } from 'react';

const STORAGE_KEY = 'aria_sessions';

function loadSessions() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
  } catch {
    return [];
  }
}

function saveSessions(sessions) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
}

function newSessionId() {
  return crypto.randomUUID();
}

export function useSessions() {
  const [sessions, setSessions] = useState(() => loadSessions());
  const [activeId, setActiveId] = useState(() => {
    const s = loadSessions();
    return s.length > 0 ? s[0].id : null;
  });

  const activeSession = sessions.find((s) => s.id === activeId) || null;

  // Create a fresh session and switch to it
  const createSession = useCallback((model = 'groq', promptId = 'default') => {
    const id = newSessionId();
    const session = {
      id,
      title: 'New Chat',
      messages: [],
      model,
      promptId,
      createdAt: Date.now(),
    };
    setSessions((prev) => {
      const updated = [session, ...prev];
      saveSessions(updated);
      return updated;
    });
    setActiveId(id);
    return id;
  }, []);

  // Switch active session
  const switchSession = useCallback((id) => {
    setActiveId(id);
  }, []);

  // Update messages + title for a session
  const updateSession = useCallback((id, patch) => {
    setSessions((prev) => {
      const updated = prev.map((s) => (s.id === id ? { ...s, ...patch } : s));
      saveSessions(updated);
      return updated;
    });
  }, []);

  // Delete a single session
  const deleteSession = useCallback(
    (id) => {
      setSessions((prev) => {
        const updated = prev.filter((s) => s.id !== id);
        saveSessions(updated);
        return updated;
      });
      if (activeId === id) {
        const remaining = sessions.filter((s) => s.id !== id);
        setActiveId(remaining.length > 0 ? remaining[0].id : null);
      }
    },
    [activeId, sessions]
  );

  // Clear all sessions
  const clearAll = useCallback(() => {
    setSessions([]);
    setActiveId(null);
    localStorage.removeItem(STORAGE_KEY);
  }, []);

  return {
    sessions,
    activeId,
    activeSession,
    createSession,
    switchSession,
    updateSession,
    deleteSession,
    clearAll,
  };
}
