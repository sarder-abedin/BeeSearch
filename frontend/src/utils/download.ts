/** Trigger a browser file download for in-memory content (Blob or text). */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  URL.revokeObjectURL(url);
}

export function downloadText(text: string, filename: string, mimeType = "text/plain"): void {
  downloadBlob(new Blob([text], { type: mimeType }), filename);
}
