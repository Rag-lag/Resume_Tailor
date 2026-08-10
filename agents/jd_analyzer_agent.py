from utils.llm_client import call_llm_json

JD_SYSTEM_PROMPT = """Extract structured requirements from a job description.
Return ONLY valid JSON:
{
  "must_have_skills": [str, ...],
  "nice_to_have_skills": [str, ...],
  "seniority": str,
  "domain_keywords": [str, ...],
  "key_responsibilities": [str, ...]
}
"""

def analyze_jd(jd_text: str) -> dict:
  return call_llm_json(jd_text, system=JD_SYSTEM_PROMPT, temperature=0.0)