/**
 * useChat.js
 * Core chat logic hook: manages messages for the active session,
 * fires the two parallel API calls (context + stream), and drives
 * token-by-token UI updates via ReadableStream.
 */

import { useState, useCallback, useRef } from 'react';
import { streamChat, fetchContext } from '../api/ariaApi';

export function useChat({ sessionId, model, promptId, multiDoc = true, onSessionUpdate }) {
  const [messages, setMessages] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const abortRef = useRef(null);

  // Sync messages from session when switching sessions
  const loadMessages = useCallback((msgs) => {
    setMessages(msgs || []);
  }, []);

  const sendMessage = useCallback(
    async (question) => {
      if (!question.trim() || isStreaming) return;

      // 1. Add user message
      const userMsg = { role: 'user', content: question, id: crypto.randomUUID() };
      const botMsg = {
        role: 'bot',
        content: '',
        id: crypto.randomUUID(),
        context: null,   // will be filled by /context call
        isStreaming: true,
      };

      setMessages((prev) => [...prev, userMsg, botMsg]);
      setIsStreaming(true);

      // Update session title from first message
      if (onSessionUpdate) {
        onSessionUpdate({ title: question.slice(0, 45) + (question.length > 45 ? '…' : '') });
      }

      // 2. Fire BOTH calls in parallel
      const contextPromise = fetchContext({ question, sessionId }).catch(() => ({ chunks: [] }));

      let streamError = null;
      try {
        const { reader, decoder } = await streamChat({
          question,
          sessionId,
          llmProvider: model,
          promptId,
          multiDoc,
        });

        // 3. Read stream tokens and update the bot bubble live
        let fullText = '';
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value, { stream: true });
          fullText += chunk;

          setMessages((prev) =>
            prev.map((m) =>
              m.id === botMsg.id ? { ...m, content: fullText } : m
            )
          );
        }
      } catch (err) {
        streamError = err.message;
        setMessages((prev) =>
          prev.map((m) =>
            m.id === botMsg.id
              ? { ...m, content: `⚠️ Error: ${err.message}`, isStreaming: false }
              : m
          )
        );
      }

      // 4. Attach retrieved context once it arrives (non-blocking)
      contextPromise.then((contextData) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === botMsg.id
              ? { ...m, context: contextData.chunks || [], isStreaming: false }
              : m
          )
        );

        // Persist final messages to session
        if (onSessionUpdate) {
          setMessages((current) => {
            onSessionUpdate({ messages: current });
            return current;
          });
        }
      });

      setIsStreaming(false);
    },
    [isStreaming, sessionId, model, promptId, multiDoc, onSessionUpdate]
  );

  const clearMessages = useCallback(() => {
    setMessages([]);
  }, []);

  return { messages, setMessages: loadMessages, isStreaming, sendMessage, clearMessages };
}
