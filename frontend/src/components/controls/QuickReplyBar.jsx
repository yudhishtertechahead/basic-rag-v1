export default function QuickReplyBar({ onSelect, disabled }) {
  const SUGGESTIONS = [
    'Leave Policy',
    'Dress Code',
    'POSH Policy',
    'Probation',
    'Referral Incentive'
  ];

  if (disabled) return null;

  return (
    <div className="quick-replies">
      {SUGGESTIONS.map((topic) => (
        <button
          key={topic}
          className="quick-reply-chip"
          onClick={() => onSelect(`What is the ${topic}?`)}
        >
          {topic}
        </button>
      ))}
    </div>
  );
}
