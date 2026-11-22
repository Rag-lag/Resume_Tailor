"""FastAPI backend: one request does the whole run and streams progress as NDJSON
lines, so nothing has to live on the server between requests. The React app in
frontend/ calls it at /api/tailor.
Local dev: venv\\Scripts\\python.exe -m uvicorn app:app --reload --port 8000"""
import json
import os
import tempfile
import traceback

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from graph.orchestrator import stream_pipeline, should_retry

SUPPORTED_EXTENSIONS = (".pdf", ".docx")
MAX_UPLOAD_BYTES = 4 * 1024 * 1024  # Vercel rejects request bodies over 4.5 MB

# The pipeline reports a node once it FINISHES, so the message names the next step.
NEXT_STEP_MESSAGE = {
    "parse_resume": "Reading the job description",
    "analyze_jd": "Finding matching experience",
    "retrieve": "Tailoring your resume",
    "tailor": "Checking the match",
}

app = FastAPI(title="Resume Tailor")


def _progress_message(node_name: str, state: dict) -> str:
    if node_name == "score":
        if should_retry(state) == "retry":
            return f"Match is low, trying again (attempt {state['iteration'] + 1})"
        return "Finishing up"
    return NEXT_STEP_MESSAGE.get(node_name, node_name)


def _line(event: dict) -> str:
    return json.dumps(event) + "\n"


@app.post("/api/tailor")
async def tailor(resume: UploadFile = File(...), job_description: str = Form("")):
    ext = os.path.splitext(resume.filename or "")[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, "Upload your resume as a PDF or DOCX file.")
    data = await resume.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(400, "That file is empty. Upload your resume as a PDF or DOCX file.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "That file is over 4 MB. Upload a smaller PDF or DOCX.")
    jd_text = job_description.strip()
    if not jd_text:
        raise HTTPException(400, "Paste the job description you're applying to.")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(data)

    def events():
        try:
            final_state = None
            for node_name, state in stream_pipeline(tmp.name, jd_text):
                final_state = state
                yield _line({"type": "progress", "node": node_name,
                             "message": _progress_message(node_name, state)})
            tailored = (final_state or {}).get("best_tailored_resume")
            score = (final_state or {}).get("best_score_result")
            if not tailored or not score:
                raise RuntimeError("Pipeline finished without producing a tailored resume.")
            yield _line({"type": "result", "tailored": tailored, "score": score})
        except Exception:
            traceback.print_exc()  # details stay in the server log, not on a public page
            yield _line({"type": "error",
                         "message": "Tailoring stopped partway through. Try again in a minute."})
        finally:
            os.unlink(tmp.name)

    return StreamingResponse(events(), media_type="application/x-ndjson")
