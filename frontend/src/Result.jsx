import { useEffect, useRef, useState } from "react";

function asPlainText(t) {
  const lines = [];
  if (t.tailored_summary) lines.push("SUMMARY", t.tailored_summary, "");
  for (const [title, items, heading] of [
    ["EXPERIENCE", t.tailored_experience, (e) => [e.role, e.company].filter(Boolean).join(", ")],
    ["PROJECTS", t.tailored_projects, (p) => p.name],
  ]) {
    if (!items?.length) continue;
    lines.push(title);
    for (const item of items) lines.push(heading(item), ...(item.bullets ?? []).map((b) => `- ${b}`));
    lines.push("");
  }
  if (t.tailored_skills?.length) lines.push("SKILLS", t.tailored_skills.join(", "));
  return lines.join("\n").trim();
}

function Entries({ title, items, heading }) {
  return (
    <>
      <h4>{title}</h4>
      {items?.length ? items.map((item, i) => {
        const [main, org] = heading(item);
        return (
          <div className="entry" key={i}>
            <p className="entry-title">{main}</p>
            {org && <p className="entry-org">{org}</p>}
            <ul>{(item.bullets ?? []).map((b, j) => <li key={j}>{b}</li>)}</ul>
          </div>
        );
      }) : <p className="empty">No {title.toLowerCase()} matched this job.</p>}
    </>
  );
}

export default function Result({ tailored, score, onStartOver }) {
  const [revealed, setRevealed] = useState(false);
  const [copyLabel, setCopyLabel] = useState("Copy resume text");
  const ref = useRef(null);
  const covered = tailored.covered_requirements ?? [];
  const gaps = score.genuine_gaps ?? [];
  const skills = (tailored.tailored_skills ?? []).join(", ");

  useEffect(() => {
    ref.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    const frame = requestAnimationFrame(() => setRevealed(true)); // starts the highlighter sweep
    return () => cancelAnimationFrame(frame);
  }, []);

  async function copy() {
    try {
      await navigator.clipboard.writeText(asPlainText(tailored));
      setCopyLabel("Copied");
    } catch {
      setCopyLabel("Couldn't copy. Select the text instead.");
    }
    setTimeout(() => setCopyLabel("Copy resume text"), 2000);
  }

  return (
    <section className={revealed ? "result is-revealed" : "result"} ref={ref}>
      <h2 className="verdict">
        {covered.length
          ? `Your resume backs up ${covered.length} requirements for this job. ${gaps.length} aren't on it yet.`
          : "Your resume doesn't back up any of this job's requirements yet."}
      </h2>
      <p className="scoreline">
        Match score {score.match_score} out of 100. {Math.round((score.keyword_coverage ?? 0) * 100)}% of
        the named skills appear in the tailored resume.
      </p>

      <div className="ledger">
        <div>
          <h3>Backed by your resume</h3>
          <ul className="marks">
            {covered.length ? covered.map((item, i) => (
              // Stagger the sweep, capped so long lists don't drag.
              <li key={i}><span style={{ transitionDelay: `${Math.min(i, 8) * 70}ms` }}>{item}</span></li>
            )) : <li className="empty">Nothing matched yet.</li>}
          </ul>
        </div>
        <div>
          <h3>Not on your resume</h3>
          <ul className="gaps">
            {gaps.length ? gaps.map((item, i) => <li key={i}><span>{item}</span></li>)
              : <li className="empty">No gaps found.</li>}
          </ul>
        </div>
      </div>

      <article className="sheet" aria-label="Tailored resume">
        {tailored.tailored_summary && (<><h4>Summary</h4><p className="summary">{tailored.tailored_summary}</p></>)}
        <Entries title="Experience" items={tailored.tailored_experience} heading={(e) => [e.role || "Role", e.company]} />
        <Entries title="Projects" items={tailored.tailored_projects} heading={(p) => [p.name || "Project", null]} />
        <h4>Skills</h4>
        <p className={skills ? "skills" : "empty"}>{skills || "No skills matched this job."}</p>
      </article>

      <div className="actions">
        <button type="button" onClick={copy}>{copyLabel}</button>
        <button type="button" className="quiet" onClick={onStartOver}>Tailor for another job</button>
      </div>
    </section>
  );
}
