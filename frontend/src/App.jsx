import { useRef, useState } from "react";
import { tailorResume } from "./api.js";
import Result from "./Result.jsx";

const MAX_BYTES = 4 * 1024 * 1024; // the API rejects bodies over 4.5 MB
const STEPS = [
  ["parse_resume", "Read your resume"],
  ["analyze_jd", "Read the job description"],
  ["retrieve", "Find matching experience"],
  ["tailor", "Tailor your resume"],
  ["score", "Check the match"],
];

function validate(file, jd) {
  if (!file) return "Choose your resume file first.";
  if (!/\.(pdf|docx)$/i.test(file.name)) return "Upload your resume as a PDF or DOCX file.";
  if (file.size > MAX_BYTES) return "That file is over 4 MB. Upload a smaller PDF or DOCX.";
  if (!jd.trim()) return "Paste the job description you're applying to.";
  return null;
}

export default function App() {
  const [file, setFile] = useState(null);
  const [jd, setJd] = useState("");
  const [dragging, setDragging] = useState(false);
  const [running, setRunning] = useState(false);
  const [doneStep, setDoneStep] = useState(-1); // index of the last finished step
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [outcome, setOutcome] = useState(null);
  const progressRef = useRef(null);
  const jdRef = useRef(null);

  async function handleSubmit(e) {
    e.preventDefault();
    const problem = validate(file, jd);
    setError(problem ?? "");
    if (problem) return;

    setRunning(true);
    setOutcome(null);
    setDoneStep(-1);
    setStatus("Reading your resume");
    requestAnimationFrame(() => progressRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }));

    const body = new FormData();
    body.append("resume", file);
    body.append("job_description", jd);
    try {
      setOutcome(await tailorResume(body, (event) => {
        setDoneStep(STEPS.findIndex(([node]) => node === event.node));
        setStatus(event.message);
      }));
    } catch (err) {
      setError(err.message);
    } finally {
      setRunning(false);
    }
  }

  function handleDrop(e) {
    e.preventDefault();
    setDragging(false);
    if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
  }

  function startOver() {
    setOutcome(null);
    setJd("");
    jdRef.current?.focus();
  }

  const dropClass = ["drop", dragging && "is-over", file && "has-file"].filter(Boolean).join(" ");

  return (
    <>
      <header className="site">
        <span className="wordmark">Resume Tailor</span>
        <a className="source" href="https://github.com/Rag-lag/Resume_Tailor">Source on GitHub</a>
      </header>

      <main>
        <section className="intro">
          <h1>Match your resume to a job without inventing anything.</h1>
          <p>Upload your resume and paste a job description. You get a tailored version built only
            from what's already on your resume, and a plain list of what it doesn't cover.</p>
        </section>

        <form onSubmit={handleSubmit} noValidate>
          <label className={dropClass}
            onDragEnter={(e) => { e.preventDefault(); setDragging(true); }}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}>
            <input type="file" accept=".pdf,.docx" onChange={(e) => setFile(e.target.files[0] ?? null)} />
            <span className="drop-title">{file ? file.name : "Drop your resume here, or choose a file"}</span>
            <span className="drop-hint">PDF or DOCX, up to 4 MB</span>
          </label>

          <label className="field-label" htmlFor="jd">Job description</label>
          <textarea id="jd" ref={jdRef} rows={10} value={jd} onChange={(e) => setJd(e.target.value)}
            placeholder="Paste the full posting, including requirements and responsibilities." />

          <div className="form-foot">
            <button type="submit" disabled={running}>{running ? "Tailoring…" : "Tailor resume"}</button>
            <p className="note">Takes 2 to 4 minutes. Your resume and the job description are sent to free
              AI models through OpenRouter, which may log them.</p>
          </div>
          {error && <p className="error" role="alert">{error}</p>}
        </form>

        {/* Always mounted: screen readers only announce changes inside a live region that already exists. */}
        <section className={running ? "progress" : undefined} ref={progressRef} aria-live="polite">
          {running && (<>
            <ol className="steps">
              {STEPS.map(([node, label], i) => (
                <li key={node} className={i <= doneStep ? "is-done" : i === doneStep + 1 ? "is-current" : ""}>
                  {label}
                </li>
              ))}
            </ol>
            <p className="status">{status}</p>
          </>)}
        </section>

        {outcome && <Result tailored={outcome.tailored} score={outcome.score} onStartOver={startOver} />}
      </main>
    </>
  );
}
