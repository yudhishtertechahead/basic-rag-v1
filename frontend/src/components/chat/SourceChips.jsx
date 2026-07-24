import { IconFile } from '../Icons';

export default function SourceChips({ context }) {
  if (!context || context.length === 0) return null;

  const seen = new Set();
  const unique = context.filter((chunk) => {
    const name = (chunk.source || 'Unknown').split(/[/\\]/).pop();
    if (seen.has(name)) return false;
    seen.add(name);
    return true;
  });

  return (
    <div className="source-chips">
      {unique.map((chunk, i) => {
        const name = (chunk.source || 'Unknown').split(/[/\\]/).pop();
        const page = chunk.page != null ? ` · p.${chunk.page + 1}` : '';
        return (
          <span key={i} className="source-chip" title={chunk.source}>
            <IconFile />
            {name}{page}
          </span>
        );
      })}
    </div>
  );
}
