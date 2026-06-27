import type { PrismaFlow } from "../../api/systematicReviewTypes";
import "./PrismaFlowSummary.css";

const METRICS: { label: string; key: keyof PrismaFlow }[] = [
  { label: "Identified", key: "identified" },
  { label: "Screened", key: "screened" },
  { label: "Eligibility", key: "eligibility" },
  { label: "Included", key: "included" },
  { label: "Excluded", key: "excluded" },
];

interface PrismaFlowSummaryProps {
  flow: PrismaFlow;
}

function PrismaFlowSummary({ flow }: PrismaFlowSummaryProps) {
  return (
    <div className="prisma-flow">
      {METRICS.map(({ label, key }) => (
        <div className="prisma-flow__metric" key={key}>
          <span className="prisma-flow__label">{label}</span>
          <span className="prisma-flow__value">{flow[key] ?? 0}</span>
        </div>
      ))}
    </div>
  );
}

export default PrismaFlowSummary;
