import ModelSelector from '../controls/ModelSelector';
import PromptSelector from '../controls/PromptSelector';
import MultiDocToggle from '../controls/MultiDocToggle';

export default function TopBar({ model, onModelChange, promptId, onPromptChange, multiDoc, onMultiDocChange, disabled }) {
  return (
    <div className="topbar">
      <div className="topbar-title">Chat</div>
      <MultiDocToggle enabled={multiDoc} onChange={onMultiDocChange} disabled={disabled} />
      <PromptSelector value={promptId} onChange={onPromptChange} disabled={disabled} />
      <ModelSelector value={model} onChange={onModelChange} disabled={disabled} />
    </div>
  );
}
