from typing import TypedDict, Optional


class ResumeState(TypedDict):
    resume_file_path: str
    jd_text: str
    collection_name: str
    structured_resume: Optional[dict]
    jd_requirements: Optional[dict]
    evidence: Optional[dict]
    tailored_resume: Optional[dict]      # latest attempt
    score_result: Optional[dict]         # latest attempt's score (feeds retry feedback)
    best_tailored_resume: Optional[dict] # highest-scoring attempt so far
    best_score_result: Optional[dict]
    iteration: int
    max_iterations: int
