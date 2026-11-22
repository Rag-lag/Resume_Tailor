from langgraph.graph import StateGraph, END

from graph.state import ResumeState
from agents.parser_agent import parse_resume, resume_to_chunks
from agents.jd_analyzer_agent import analyze_jd
from agents.retrieval_agent import gather_evidence
from agents.tailoring_agent import tailor_resume
from agents.scoring_agent import score_tailored_resume
from vectorstore.chroma_client import index_resume_chunks, new_collection_name, delete_collection

SCORE_THRESHOLD = 70

# Nodes return only the keys they change; LangGraph merges them into the state.


def node_parse_resume(state: ResumeState) -> dict:
    structured = parse_resume(state["resume_file_path"])
    index_resume_chunks(resume_to_chunks(structured), state["collection_name"])
    return {"structured_resume": structured}


def node_analyze_jd(state: ResumeState) -> dict:
    return {"jd_requirements": analyze_jd(state["jd_text"])}


def node_retrieve(state: ResumeState) -> dict:
    return {"evidence": gather_evidence(state["jd_requirements"], state["collection_name"])}


def node_tailor(state: ResumeState) -> dict:
    feedback = (state.get("score_result") or {}).get("improvement_feedback", "")
    tailored = tailor_resume(
        state["structured_resume"], state["jd_requirements"],
        state["evidence"], feedback=feedback,
    )
    return {"tailored_resume": tailored, "iteration": state["iteration"] + 1}


def node_score(state: ResumeState) -> dict:
    result = score_tailored_resume(state["tailored_resume"], state["jd_requirements"])
    update = {"score_result": result}
    best = state.get("best_score_result")
    # A retry can score lower than the first draft; keep whichever attempt is best.
    if best is None or result["match_score"] > best["match_score"]:
        update["best_tailored_resume"] = state["tailored_resume"]
        update["best_score_result"] = result
    return update


def should_retry(state: ResumeState) -> str:
    score = state["score_result"]["match_score"]  # already coerced to int by the scorer
    if score >= SCORE_THRESHOLD or state["iteration"] >= state["max_iterations"]:
        return "end"
    return "retry"


def build_graph():
    graph = StateGraph(ResumeState)
    graph.add_node("parse_resume", node_parse_resume)
    graph.add_node("analyze_jd", node_analyze_jd)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("tailor", node_tailor)
    graph.add_node("score", node_score)

    graph.set_entry_point("parse_resume")
    graph.add_edge("parse_resume", "analyze_jd")
    graph.add_edge("analyze_jd", "retrieve")
    graph.add_edge("retrieve", "tailor")
    graph.add_edge("tailor", "score")
    graph.add_conditional_edges("score", should_retry, {"retry": "tailor", "end": END})
    return graph.compile()


_APP = build_graph()  # compiled once per process, reused by every request


def _initial_state(resume_file_path: str, jd_text: str, max_iterations: int) -> ResumeState:
    return {
        "resume_file_path": resume_file_path,
        "jd_text": jd_text,
        "collection_name": new_collection_name(),
        "structured_resume": None,
        "jd_requirements": None,
        "evidence": None,
        "tailored_resume": None,
        "score_result": None,
        "best_tailored_resume": None,
        "best_score_result": None,
        "iteration": 0,
        "max_iterations": max_iterations,
    }


def stream_pipeline(resume_file_path: str, jd_text: str, max_iterations: int = 2):
    """Yields (node_name, current_state) after each node completes — for driving
    a progress UI. Use run_pipeline() when you only want the final state."""
    state = _initial_state(resume_file_path, jd_text, max_iterations)
    try:
        # "updates" mode is used (rather than "values") because the UI needs the
        # node name; the partial update is merged into the running state here.
        for update in _APP.stream(state, stream_mode="updates"):
            for node_name, node_output in update.items():
                state = {**state, **(node_output or {})}
                yield node_name, state
    finally:
        delete_collection(state["collection_name"])


def run_pipeline(resume_file_path: str, jd_text: str, max_iterations: int = 2) -> ResumeState:
    state = None
    for _, state in stream_pipeline(resume_file_path, jd_text, max_iterations):
        pass
    return state
