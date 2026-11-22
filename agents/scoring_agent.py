import re

from utils.llm_client import call_llm_json

SCORE_SYSTEM_PROMPT = """Score how well a tailored resume matches a job description.
Return ONLY valid JSON:
{
  "match_score": int,          // 0-100
  "genuine_gaps": [str, ...],  // requirements with no honest supporting evidence
  "improvement_feedback": str  // specific, actionable note for the tailoring agent
}
Be honest — do not inflate the score. A candid low score with real gaps listed
is more useful than a flattering one.

genuine_gaps must list specific, concrete requirements (individual skills,
tools, or technical responsibilities) — the same granularity as the JD
requirements' must_have_skills/nice_to_have_skills/key_responsibilities
fields. Do not restate the job posting's general mission, culture, or
soft-skill framing as a gap; only list things a hiring manager would check
off a skills list.
"""


def _to_score(value) -> int:
    """LLM JSON may return the score as a string, a float, or not at all."""
    try:
        return max(0, min(100, int(float(value))))
    except (TypeError, ValueError):
        return 0


def _resume_text(tailored_resume: dict) -> str:
    parts = [tailored_resume.get("tailored_summary", "")]
    for section in ("tailored_experience", "tailored_projects"):
        for entry in tailored_resume.get(section, []) or []:
            parts.extend(entry.get("bullets", []) or [])
    parts.extend(tailored_resume.get("tailored_skills", []) or [])
    return " ".join(str(p) for p in parts).lower()


def keyword_coverage(tailored_resume: dict, jd_requirements: dict) -> float:
    """Fraction of the JD's named skills that literally appear in the tailored resume.
    Computed independently of both LLMs, so it can't just echo the tailor's self-report."""
    keywords = (jd_requirements.get("must_have_skills", []) or []) + \
               (jd_requirements.get("nice_to_have_skills", []) or [])
    keywords = {k.strip().lower() for k in keywords if k and k.strip()}
    if not keywords:
        return 0.0
    text = _resume_text(tailored_resume)
    # Whole-word match, so "Java" doesn't count "JavaScript" and "Go" doesn't count "Google".
    hits = sum(1 for k in keywords if re.search(rf"(?<!\w){re.escape(k)}(?!\w)", text))
    return round(hits / len(keywords), 3)


def score_tailored_resume(tailored_resume: dict, jd_requirements: dict) -> dict:
    prompt = f"Tailored resume: {tailored_resume}\n\nJD requirements: {jd_requirements}"
    result = call_llm_json(prompt, system=SCORE_SYSTEM_PROMPT, temperature=0.0)
    return {
        "match_score": _to_score(result.get("match_score")),
        "genuine_gaps": [str(g) for g in result.get("genuine_gaps", []) or []],
        "improvement_feedback": str(result.get("improvement_feedback", "") or ""),
        "keyword_coverage": keyword_coverage(tailored_resume, jd_requirements),
    }
