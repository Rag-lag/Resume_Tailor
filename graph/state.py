from typing import TypedDict, Optional


class ResumeState(TypedDict):
    resume_file_path: str
    jd_text: str
    started_at: float                    # time.time() at start, for the retry time budget
    structured_resume: Optional[dict]
    index: Optional[dict]                # this run's resume chunks + vectors
    jd_requirements: Optional[dict]
    evidence: Optional[dict]
    tailored_resume: Optional[dict]      # latest attempt
    score_result: Optional[dict]         # latest attempt's score (feeds retry feedback)
    best_tailored_resume: Optional[dict] # highest-scoring attempt so far
    best_score_result: Optional[dict]
    iteration: int
    max_iterations: int
