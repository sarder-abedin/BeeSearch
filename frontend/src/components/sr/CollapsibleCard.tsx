import { useState, type ReactNode } from "react";
import "./CollapsibleCard.css";

interface CollapsibleCardProps {
  header: ReactNode;
  children: ReactNode;
}

function CollapsibleCard({ header, children }: CollapsibleCardProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="collapsible-card">
      <button
        type="button"
        className="collapsible-card__toggle"
        onClick={() => setExpanded((prev) => !prev)}
        aria-expanded={expanded}
      >
        <span className="collapsible-card__chevron" aria-hidden="true">
          {expanded ? "▾" : "▸"}
        </span>
        {header}
      </button>
      {expanded && <div className="collapsible-card__body">{children}</div>}
    </div>
  );
}

export default CollapsibleCard;
