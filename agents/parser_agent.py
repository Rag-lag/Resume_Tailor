from utils.llm_client import call_llm_json
from utils.pdf_docx_parser import extract_text

PARSE_SYSTEM_PROMPT = """You convert raw resume text into structured JSON.
Return ONLY valid JSON matching this schema, no other text:
{
  "experience": [
    {
      "company": str, "role": str, "dates": str,
      "bullets": [str, ...]
    }
  ],
  "projects": [
    {"name": str, "description": str, "bullets": [str, ...]}
  ],
  "skills": [str, ...],
  "education": [{"institution": str, "degree": str, "dates": str}]
}
Preserve the original wording of bullets exactly — do not paraphrase or invent content.
"""

def parse_resume(file_path: str) -> dict:
    raw_text = extract_text(file_path)
    structured = call_llm_json(raw_text, system=PARSE_SYSTEM_PROMPT)
    return structured


def resume_to_chunks(structured: dict) -> list[dict]:
    """Flatten structured resume into retrievable chunks with metadata."""
    chunks = []
    for exp in structured.get("experience", []):
        for bullet in exp["bullets"]:
            chunks.append({
                "text": bullet,
                "source_type": "experience",
                "company": exp["company"],
                "role": exp["role"],
            })
    for proj in structured.get("projects", []):
        for bullet in proj.get("bullets", []):
            chunks.append({
                "text": bullet,
                "source_type": "project",
                "project_name": proj["name"],
            })
    seen_skills = set()
    for skill in structured.get("skills", []):
        if skill.lower() not in seen_skills:
            seen_skills.add(skill.lower())
            chunks.append({"text": skill, "source_type": "skill"})
    return chunks