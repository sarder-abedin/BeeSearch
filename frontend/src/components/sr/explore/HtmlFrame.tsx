import "../sr-common.css";

interface HtmlFrameProps {
  html: string;
  height?: number;
  title: string;
}

/** Renders a server-generated HTML+script blob (pyvis / plotly) in a sandboxed
 * iframe -- mirrors Streamlit's `st.components.v1.html()`. Deliberately NOT
 * `dangerouslySetInnerHTML`: browsers strip <script> tags from innerHTML, which
 * would silently break every chart this renders. No `sandbox` attribute, so
 * scripts execute, matching Streamlit's own default. */
function HtmlFrame({ html, height = 520, title }: HtmlFrameProps) {
  return <iframe srcDoc={html} className="sr-frame" style={{ height }} title={title} />;
}

export default HtmlFrame;
