import traceback

import gradio as gr

from graph.orchestrator import stream_pipeline, should_retry

SUPPORTED_EXTENSIONS = (".pdf", ".docx")

# The pipeline yields after a node FINISHES, so the status shows the step that
# is about to run next, not the one that just completed.
NEXT_STEP_LABEL = {
    "parse_resume": "Analyzing job description",
    "analyze_jd": "Retrieving relevant experience",
    "retrieve": "Tailoring resume",
    "tailor": "Scoring match quality",
}

IDLE_BUTTON = gr.update(interactive=True, value="Tailor resume")
BUSY_BUTTON = gr.update(interactive=False, value="Working...")


def _status_after(node_name: str, state: dict) -> str:
    if node_name == "score":
        if should_retry(state) == "retry":
            return f"⏳ Score below target, re-tailoring (attempt {state['iteration'] + 1})..."
        return "⏳ Finishing..."
    label = NEXT_STEP_LABEL.get(node_name, node_name)
    if node_name == "tailor" and state["iteration"] > 1:
        label += f" (attempt {state['iteration']})"
    return f"⏳ {label}..."


def _join(items) -> str:
    return ", ".join(str(i) for i in (items or []))


def _render_entries(entries, title_fn) -> str:
    blocks = []
    for entry in entries or []:
        bullets = "\n".join(f"- {b}" for b in entry.get("bullets", []) or [])
        blocks.append(f"**{title_fn(entry)}**\n{bullets}")
    return "\n\n".join(blocks)


def _render_results(tailored: dict, score: dict) -> tuple[str, str, str, str]:
    summary = f"""## Match score: {score.get('match_score', 0)}/100

**Keyword coverage:** {score.get('keyword_coverage', 0.0):.0%}

**Covered requirements:** {_join(tailored.get('covered_requirements')) or 'None'}

**Gaps (honest):** {_join(score.get('genuine_gaps')) or 'None identified'}
"""
    if tailored.get("tailored_summary"):
        summary += f"\n**Summary:** {tailored['tailored_summary']}\n"

    experience = _render_entries(
        tailored.get("tailored_experience"),
        lambda e: f"{e.get('role', 'Role')} — {e.get('company', 'Company')}",
    ) or "_No experience matched this job description._"
    projects = _render_entries(
        tailored.get("tailored_projects"),
        lambda p: p.get("name", "Project"),
    ) or "_No projects matched this job description._"
    skills = _join(tailored.get("tailored_skills")) or "_No matching skills found._"
    return summary, experience, projects, skills


def process(resume_file, jd_text):
    resume_path = getattr(resume_file, "name", resume_file)
    jd_text = (jd_text or "").strip()

    if not resume_path:
        yield IDLE_BUTTON, "⚠️ Please upload a resume (PDF or DOCX).", "", "", "", ""
        return
    if not str(resume_path).lower().endswith(SUPPORTED_EXTENSIONS):
        yield IDLE_BUTTON, "⚠️ Unsupported file type. Upload a PDF or DOCX.", "", "", "", ""
        return
    if not jd_text:
        yield IDLE_BUTTON, "⚠️ Please paste a job description.", "", "", "", ""
        return

    yield BUSY_BUTTON, "⏳ Parsing resume...", "", "", "", ""

    try:
        final_state = None
        for node_name, state in stream_pipeline(resume_path, jd_text):
            final_state = state
            yield BUSY_BUTTON, _status_after(node_name, state), "", "", "", ""

        tailored = (final_state or {}).get("best_tailored_resume")
        score = (final_state or {}).get("best_score_result")
        if not tailored or not score:
            raise RuntimeError("Pipeline finished without producing a tailored resume.")
        outputs = _render_results(tailored, score)
    except Exception:
        traceback.print_exc()  # full details stay in the server log, not on a public page
        yield IDLE_BUTTON, "❌ Something went wrong, please try again.", "", "", "", ""
        return

    yield IDLE_BUTTON, "✅ Done", *outputs


with gr.Blocks(title="Resume Tailor") as demo:
    gr.Markdown("# Multi-agent resume tailoring")

    with gr.Row():
        resume_input = gr.File(label="Upload resume (PDF or DOCX)", file_types=list(SUPPORTED_EXTENSIONS))
        jd_input = gr.Textbox(label="Job description", lines=10)

    submit_btn = gr.Button("Tailor resume", variant="primary")
    status_output = gr.Markdown()

    score_output = gr.Markdown(label="Score")
    gr.Markdown("### Experience")
    bullets_output = gr.Markdown()
    gr.Markdown("### Projects")
    projects_output = gr.Markdown()
    gr.Markdown("### Skills")
    skills_output = gr.Markdown()

    submit_btn.click(
        fn=process,
        inputs=[resume_input, jd_input],
        outputs=[submit_btn, status_output, score_output, bullets_output, projects_output, skills_output],
        concurrency_limit=4,  # runs are isolated per collection; Gradio's default of 1 queues everyone
    )

if __name__ == "__main__":
    demo.launch()
