import { useEffect, useState } from 'react';
import { fetchPrompts } from '../../api/ariaApi';
import { IconSliders, IconChevronDown } from '../Icons';

export default function PromptSelector({ value, onChange, disabled }) {
  const [prompts, setPrompts] = useState([]);

  useEffect(() => {
    fetchPrompts().then(data => {
      if (data && data.prompts) setPrompts(data.prompts);
    }).catch(err => console.error("Failed to load prompts", err));
  }, []);

  return (
    <div className="select-wrapper">
      <IconSliders className="icon-left" />
      <select value={value} onChange={(e) => onChange(e.target.value)} disabled={disabled || prompts.length === 0}>
        {prompts.length === 0 ? (
          <option value="default">Loading prompts...</option>
        ) : (
          prompts.map(p => (
            <option key={p.id} value={p.id} title={p.description}>{p.name}</option>
          ))
        )}
      </select>
      <IconChevronDown className="icon-right" />
    </div>
  );
}
