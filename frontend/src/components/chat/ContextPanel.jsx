import { useState } from 'react';
import { IconFile, IconChevronRight } from '../Icons';

export default function ContextPanel({ chunks }) {
  const [open, setOpen] = useState(false);

  if (!chunks || chunks.length === 0) return null;

  return (
    <div className="context-panel">
      <button className="context-toggle" onClick={() => setOpen((o) => !o)}>
        <IconFile />
        <span>Retrieved Context ({chunks.length})</span>
        <IconChevronRight className={open ? 'open' : ''} />
      </button>

      {open && (
        <div className="context-chunks">
          {chunks.map((chunk) => (
            <div key={chunk.index} className="context-chunk">
              <div className="context-chunk-header">
                <span className="context-chunk-index">{chunk.index}</span>
                <span className="context-chunk-source">
                  {(chunk.source || 'Unknown').split(/[/\\]/).pop()}
                </span>
                {chunk.page != null && (
                  <span className="context-chunk-page">Page {chunk.page + 1}</span>
                )}
              </div>
              <div className="context-chunk-text">{chunk.content}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
