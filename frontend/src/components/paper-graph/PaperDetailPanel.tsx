import type { ExpandRelationship, PaperNode } from "../../api/paperGraphTypes";

interface PaperDetailPanelProps {
  node: PaperNode;
  /** For Feature 1: clicking this sets the node as the new similarity-graph origin. */
  onSetAsOrigin?: (node: PaperNode) => void;
  /** For Feature 2: expands this node's neighborhood by the chosen relationship. */
  onExpand?: (nodeId: string, relationship: ExpandRelationship) => void;
  expandRelationship?: ExpandRelationship;
  onExpandRelationshipChange?: (r: ExpandRelationship) => void;
  expanding?: boolean;
}

const RELATIONSHIP_LABELS: Record<ExpandRelationship, string> = {
  earlier: "Earlier work (references)",
  later: "Later work (citations)",
  similar: "Similar papers (recommended)",
  authors: "Author network",
};

export default function PaperDetailPanel({
  node,
  onSetAsOrigin,
  onExpand,
  expandRelationship = "later",
  onExpandRelationshipChange,
  expanding = false,
}: PaperDetailPanelProps) {
  const meta = [
    node.authors.slice(0, 3).join(", ") + (node.authors.length > 3 ? " et al." : ""),
    node.year ?? "n.d.",
    node.venue,
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <div className="pg-detail">
      <h4>{node.title}</h4>
      <p className="pg-detail__meta">{meta}</p>
      {node.citation_count != null && (
        <p className="pg-detail__meta">Citations: {node.citation_count.toLocaleString()}</p>
      )}

      {node.abstract ? (
        <p className="pg-detail__abstract">{node.abstract.slice(0, 400)}{node.abstract.length > 400 ? "…" : ""}</p>
      ) : (
        <p className="pg-detail__abstract pg-detail__abstract--unavailable">Abstract unavailable</p>
      )}

      {node.url && (
        <p className="pg-detail__link">
          <a href={node.url} target="_blank" rel="noopener noreferrer">Open on Semantic Scholar ↗</a>
        </p>
      )}

      <div className="pg-detail__actions">
        {onSetAsOrigin && (
          <button type="button" className="sr-button" onClick={() => onSetAsOrigin(node)}>
            Set as new origin
          </button>
        )}
      </div>

      {onExpand && (
        <div className="pg-expand-row">
          <label htmlFor="pg-rel-select">Explore:</label>
          <select
            id="pg-rel-select"
            value={expandRelationship}
            onChange={(e) => onExpandRelationshipChange?.(e.target.value as ExpandRelationship)}
          >
            {(Object.keys(RELATIONSHIP_LABELS) as ExpandRelationship[]).map((r) => (
              <option key={r} value={r}>{RELATIONSHIP_LABELS[r]}</option>
            ))}
          </select>
          <button
            type="button"
            className="sr-button"
            disabled={expanding}
            onClick={() => onExpand(node.id, expandRelationship)}
          >
            {expanding ? "Expanding…" : "Expand"}
          </button>
        </div>
      )}
    </div>
  );
}
