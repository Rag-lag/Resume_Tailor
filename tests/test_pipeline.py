"""Run from the repo root: venv\\Scripts\\python.exe -m unittest discover -s tests
Stubs out every LLM call; still needs OPENROUTER_API_KEY set (llm_client reads it on import)."""
import unittest
from unittest.mock import patch

import agents.scoring_agent as scoring
import graph.orchestrator as orch
from agents.scoring_agent import _to_score, keyword_coverage

RESUME = {"experience": [{"company": "A", "role": "Dev", "bullets": ["Built a Flutter app for 1M users"]}],
          "projects": [], "skills": ["Flutter"]}
JD = {"must_have_skills": ["Flutter"], "nice_to_have_skills": [], "key_responsibilities": []}


class ScoringTest(unittest.TestCase):
    def test_score_is_coerced(self):
        self.assertEqual(_to_score("65"), 65)
        self.assertEqual(_to_score(None), 0)
        self.assertEqual(_to_score(140), 100)

    def test_coverage_matches_whole_words_only(self):
        jd = {"must_have_skills": ["Java", "Go", "C"]}
        self.assertEqual(keyword_coverage({"tailored_skills": ["JavaScript", "Google Cloud"]}, jd), 0.0)
        self.assertEqual(keyword_coverage({"tailored_skills": ["C#", "C++"]}, {"must_have_skills": ["C#", "C++"]}), 1.0)


class PipelineTest(unittest.TestCase):
    def run_with_scores(self, scores):
        scores = iter(scores)
        attempts = []

        def fake_tailor(structured, jd, evidence, feedback=""):
            attempts.append(evidence)
            return {"tailored_summary": f"attempt {len(attempts)}", "tailored_skills": ["Flutter"]}

        with patch.object(orch, "parse_resume", lambda p: RESUME), \
             patch.object(orch, "analyze_jd", lambda t: JD), \
             patch.object(orch, "tailor_resume", fake_tailor), \
             patch.object(scoring, "call_llm_json", lambda *a, **k: {"match_score": next(scores)}):
            return orch.run_pipeline("resume.pdf", "jd"), attempts

    def test_keeps_best_attempt_when_retry_is_worse(self):
        state, _ = self.run_with_scores([65, 40])
        self.assertEqual(state["best_tailored_resume"]["tailored_summary"], "attempt 1")
        self.assertEqual(state["best_score_result"]["match_score"], 65)

    def test_retrieval_finds_the_real_bullet(self):
        _, attempts = self.run_with_scores([90])
        texts = [e["text"] for e in attempts[0]["Flutter"]]
        self.assertIn("Built a Flutter app for 1M users", texts)


if __name__ == "__main__":
    unittest.main()
