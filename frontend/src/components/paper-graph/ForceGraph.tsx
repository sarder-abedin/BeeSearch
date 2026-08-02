import { useCallback } from "react";
import ForceGraph2D from "react-force-graph-2d";
import type { GraphData, PaperNode } from "../../api/paperGraphTypes";

interface FGNode {
  id: string;
  title: string;
  year: number | null;
  citation_count: number | null;
  _paper: PaperNode;
  // Populated by react-force-graph-2d during layout
  x?: number;
  y?: number;
}

interface FGLink {
  source: string;
  target: string;
  weight: number;
  edge_type: string;
}

interface ForceGraphProps {
  graph: GraphData;
  onNodeClick: (paper: PaperNode) => void;
  selectedNodeId?: string | null;
  width?: number;
  height?: number;
}

// Year → colour gradient: older = muted blue, newer = warm amber (accent colour)
function yearToColor(year: number | null): string {
  if (year == null) return "#999999";
  const minY = 2000;
  const maxY = new Date().getFullYear();
  const t = Math.max(0, Math.min(1, (year - minY) / (maxY - minY)));
  // Interpolate from #6b6f6a (muted) to #b8860b (accent)
  const r = Math.round(107 + t * (184 - 107));
  const g = Math.round(111 + t * (134 - 111));
  const b = Math.round(106 + t * (11 - 106));
  return `rgb(${r},${g},${b})`;
}

// Node radius scaled by log(citation_count) so highly-cited papers are larger
function nodeRadius(citationCount: number | null): number {
  if (citationCount == null || citationCount <= 0) return 5;
  return Math.min(20, 4 + Math.log10(citationCount + 1) * 4);
}

export default function ForceGraph({
  graph,
  onNodeClick,
  selectedNodeId,
  width = 680,
  height = 480,
}: ForceGraphProps) {
  const fgNodes: FGNode[] = graph.nodes.map((p) => ({
    id: p.id,
    title: p.title,
    year: p.year,
    citation_count: p.citation_count,
    _paper: p,
  }));

  const fgLinks: FGLink[] = graph.edges.map((e) => ({
    source: e.source,
    target: e.target,
    weight: e.weight,
    edge_type: e.edge_type,
  }));

  const paintNode = useCallback(
    (node: FGNode, ctx: CanvasRenderingContext2D) => {
      const r = nodeRadius(node.citation_count);
      const isSelected = node.id === selectedNodeId;
      ctx.beginPath();
      ctx.arc(node.x ?? 0, node.y ?? 0, isSelected ? r + 3 : r, 0, 2 * Math.PI);
      ctx.fillStyle = yearToColor(node.year);
      ctx.fill();
      if (isSelected) {
        ctx.strokeStyle = "#b8860b";
        ctx.lineWidth = 2;
        ctx.stroke();
      }
      // Label for selected node or large nodes
      if (isSelected || r >= 10) {
        ctx.font = `${isSelected ? "bold " : ""}${Math.max(9, r * 0.8)}px sans-serif`;
        ctx.fillStyle = "#1f2320";
        ctx.textAlign = "center";
        ctx.fillText(
          node.title.length > 30 ? node.title.slice(0, 28) + "…" : node.title,
          node.x ?? 0,
          (node.y ?? 0) + r + 9,
        );
      }
    },
    [selectedNodeId],
  );

  const linkColor = useCallback((link: FGLink) => {
    switch (link.edge_type) {
      case "reference": return "rgba(100,130,200,0.5)";
      case "citation":  return "rgba(80,160,100,0.5)";
      case "co_author": return "rgba(180,80,180,0.5)";
      case "recommendation": return "rgba(200,140,50,0.5)";
      default:          return `rgba(184,134,11,${Math.max(0.1, link.weight * 0.7)})`;
    }
  }, []);

  const linkWidth = useCallback(
    (link: FGLink) => (link.edge_type === "similarity" ? Math.max(0.5, link.weight * 3) : 1),
    [],
  );

  return (
    <ForceGraph2D
      graphData={{ nodes: fgNodes, links: fgLinks }}
      width={width}
      height={height}
      nodeId="id"
      nodeCanvasObject={paintNode}
      nodeCanvasObjectMode={() => "replace"}
      linkColor={linkColor}
      linkWidth={linkWidth}
      linkDirectionalArrowLength={(link: FGLink) =>
        link.edge_type === "reference" || link.edge_type === "citation" ? 4 : 0
      }
      linkDirectionalArrowRelPos={1}
      onNodeClick={(node: FGNode) => onNodeClick(node._paper)}
      nodeLabel={(node: FGNode) => node.title}
      d3VelocityDecay={0.3}
      cooldownTicks={100}
    />
  );
}
