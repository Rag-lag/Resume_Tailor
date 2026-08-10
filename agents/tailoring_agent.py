from utils.llm_client import call_llm_json

TAILOR_SYSTEM_PROMPT = """You tailor a resume to a job description using ONLY
the provided evidence chunks — real bullets from the candidate's actual resume.

Rules:
- Never invent skills, tools, or achievements not present in the evidence.
- You may reorder, re-emphasize, and rephrase evidence bullets to mirror the
  JD's terminology, but the underlying claim must stay true to the original.
- If a JD requirement has no supporting evidence, do not fabricate a bullet —
  omit it and it will be reported as a gap.
- Prioritize bullets that match must-have skills over nice-to-have ones.
- The tailored_summary is held to the SAME grounding rule as bullets. Do not
  restate the JD's own requirements (years of experience, seniority level,
  , etc.) as if they describe the candidate. Only claim a seniority level, 
  years of experience, or scope of experience that is directly and honestly 
  supported by the evidence provided. If the evidence suggests an early-career
  or junior profile, the summary must reflect that honestly rather 
  than mirroring the JD's seniority language.
- tailored_projects: include any project evidence chunks that are relevant
  to the JD, in the same grounded, no-invention style as tailored_experience.
  Omit projects with no relevant evidence rather than forcing a weak match.
- tailored_skills: list ONLY skills that appear as skill-type evidence chunks
  AND are relevant to this JD's requirements. Do not add a skill just
  because the JD mentions it — it must be present in the candidate's actual
  evidence.
- For any role or project you include in tailored_experience or
  tailored_projects, include ALL of that role's retrieved evidence bullets
  that are reasonably relevant to the JD, not just the single strongest
  match — aim for 2-3 bullets per included role when that many relevant
  bullets exist in the evidence, so the output reads as a complete resume
  entry rather than a single fragment. Only include fewer than 2 if the
  evidence genuinely doesn't support more.  
- SELF-CONSISTENCY: every claim in tailored_summary and every entry in
  covered_requirements must be traceable to something actually included in
  THIS SAME OUTPUT's tailored_experience, tailored_projects, or
  tailored_skills. Do not summarize or claim coverage of a requirement using
  evidence you chose not to surface elsewhere in this output — if you
  reference a specific technology in the summary or mark a
  requirement covered, the corresponding bullet or skill must appear in
  this same response.

Return ONLY valid JSON:
{
  "tailored_experience": [
    {"company": str, "role": str, "bullets": [str, ...]}
  ],
  "tailored_projects": [
    {"name": str, "bullets": [str, ...]}
  ],
  "tailored_skills": [str, ...],
  "tailored_summary": str,
  "covered_requirements": [str, ...],
  "uncovered_requirements": [str, ...]
}
"""

def tailor_resume(structured_resume: dict, jd_requirements: dict,
                   evidence: dict, feedback: str = "") -> dict:
    prompt = f"""
Job requirements: {jd_requirements}

Evidence retrieved per requirement: {evidence}

Original resume structure (for company/role context): {structured_resume}

{"Feedback from previous attempt to address: " + feedback if feedback else ""}
"""
    return call_llm_json(prompt, system=TAILOR_SYSTEM_PROMPT, temperature=0.0)