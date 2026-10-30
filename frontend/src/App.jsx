import { useState } from "react";
const API = `${import.meta.env.VITE_API_URL || "http://localhost:8000"}/api`;
export default function App() {
  const [msgs, setMsgs] = useState([]);
  const [q, setQ] = useState("");
  const [status, setStatus] = useState("No PDF uploaded");
  const [busy, setBusy] = useState(false);

  async function upload(e) {
    const f = e.target.files[0];
    if (!f) return;
    const fd = new FormData();
    fd.append("file", f);
    setStatus("Indexing...");
    try {
      const j = await (await fetch(`${API}/upload`, { method: "POST", body: fd })).json();
      setStatus(`${j.file}: ${j.chunks} chunks indexed`);
    } catch { setStatus("Upload failed"); }
  }

  async function send() {
    if (!q.trim() || busy) return;
    const question = q;
    setQ(""); setBusy(true);
    setMsgs((m) => [...m, { role: "user", answer: question }]);
    try {
      const r = await fetch(`${API}/chat`, { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: "web", question }) });
      const j = await r.json();
      setMsgs((m) => [...m, { role: "bot", ...j }]);
    } catch {
      setMsgs((m) => [...m, { role: "bot", answer: "Request failed.", trace: [], sources: [] }]);
    }
    setBusy(false);
  }

  return (
    <div style={{ maxWidth: 760, margin: "24px auto", fontFamily: "system-ui", padding: 12 }}>
      <h2>InsightBot</h2>
      <input type="file" accept="application/pdf" onChange={upload} /> <small>{status}</small>
      <div style={{ margin: "16px 0" }}>
        {msgs.map((m, i) => (
          <div key={i} style={{ margin: "12px 0", textAlign: m.role === "user" ? "right" : "left" }}>
            <div style={{ display: "inline-block", background: m.role === "user" ? "#dbeafe" : "#f3f4f6",
              padding: 10, borderRadius: 8, maxWidth: "85%", textAlign: "left" }}>
              {m.answer}
              {m.role === "bot" && m.trace && (
                <>
                  <div style={{ marginTop: 6, fontSize: 12, color: m.grounded ? "green" : "crimson" }}>
                    {m.grounded ? "✔ grounded in document" : "⚠ may not be fully supported by the document"}
                  </div>
                  <details style={{ fontSize: 12, marginTop: 4 }}>
                    <summary>Trace ({m.trace.length} steps)</summary>
                    {m.trace.map((t, k) => <div key={k}>{t.step} · {t.ms} ms · {t.info}</div>)}
                  </details>
                  <details style={{ fontSize: 12 }}>
                    <summary>Sources ({m.sources.length})</summary>
                    {m.sources.map((s, k) => <div key={k}>p.{s.page}: {s.text}…</div>)}
                  </details>
                </>
              )}
            </div>
          </div>
        ))}
        {busy && <small>Thinking…</small>}
      </div>
      <input value={q} onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === "Enter" && send()}
        placeholder="Ask about your PDF" style={{ width: "80%", padding: 8 }} />
      <button onClick={send} disabled={busy} style={{ padding: 8, marginLeft: 8 }}>Send</button>
    </div>
  );
}
