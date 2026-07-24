import { IconSettings, IconChevronDown } from '../Icons';

export default function ModelSelector({ value, onChange, disabled }) {
  return (
    <div className="select-wrapper">
      <IconSettings className="icon-left" />
      <select value={value} onChange={(e) => onChange(e.target.value)} disabled={disabled}>
        <option value="groq">Groq (Fast)</option>
        <option value="google">Gemini</option>
        <option value="ollama">Ollama (Local)</option>
      </select>
      <IconChevronDown className="icon-right" />
    </div>
  );
}
