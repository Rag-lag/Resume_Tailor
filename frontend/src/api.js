// POSTs the form and reads the NDJSON stream: one JSON object per line, progress
// events first, then a single result or error. Resolves with the result event.
export async function tailorResume(formData, onProgress) {
  let response;
  try {
    response = await fetch("/api/tailor", { method: "POST", body: formData });
  } catch {
    throw new Error("Couldn't reach the server. Check your connection and try again.");
  }
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === "string" ? body.detail
      : "The server couldn't start this run. Try again in a minute.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop();
    for (const line of lines.filter(Boolean)) {
      const event = JSON.parse(line);
      if (event.type === "progress") onProgress(event);
      else if (event.type === "result") return event;
      else if (event.type === "error") throw new Error(event.message);
    }
    if (done) break;
  }
  throw new Error("The run ended before finishing, probably a time limit. Try again.");
}
