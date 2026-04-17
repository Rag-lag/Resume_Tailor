from __future__ import annotations

from typing import Dict, List

from llm import ask_llm
from rag import retrieve

TYPE_PRIORITY = {
    "experience": 1,
    "project": 2,
    "achievement": 3,
    "skill": 4,
    "education": 5,
    "other": 6,
}

def rerank_results(results: List[Dict[str, object]]):
    def sort_key(item: Dict[str, object]):
        meta = item.get("metadata", {}) or {}
        item_type = meta.get("type", "other")
        priority = TYPE_PRIORITY.get(str(item_type), 6)
        distance = float(item.get("distance", 9999))
        return (priority, distance)
    return sorted(results, key=sort_key)

def format_context(results: List[Dict[str, object]]):
    parts: List[str] = []
    for i, item in enumerate(results, start=1):
        meta = item.get("metadata", {}) or {}
        source = meta.get("source", "unknown")
        item_type = meta.get("type", "unknown")
        tags = meta.get("tags", "")

        parts.append(
            f"[Evidence {i}]\n"
            f"Source: {source}\n"
            f"Type: {item_type}\n"
            f"Tags: {tags}\n"
            f"Distance: {item['distance']}\n"
            f"Text:\n{item['text']}"
        )

    return "\n\n".join(parts)

def analyze_jd(job_desc: str) -> str:
    prompt = f"""
        You are analyzing a job description for a candidate-fit assistant.

        Job Description:
        {job_desc}

        Return these sections:
        1. Role Type
        2. Core Required Skills
        3. Nice-to-Have Skills
        4. Main Responsibilities
        5. Seniority Hints
        6. Search Keywords

        Be concise and practical.
    """
    return ask_llm(prompt)

def generate_fit_analysis(job_desc: str, jd_analysis: str, context: str) -> str:
    prompt = f"""
        You are an AI assistant helping evaluate candidate fit.

        Job Description:
        {job_desc}

        JD Analysis:
        {jd_analysis}

        Candidate Evidence:
        {context}

        Task:
        Write a grounded fit analysis using only the candidate evidence.

        Strict Rules:
        - Use only the provided evidence.
        - Do not invent experience.
        - Clearly distinguish between direct match, adjacent match, and gaps.
        - Do not include commentary outside the requested format.

        Required Output Format:

        Overall Fit:
        <2-4 sentences>

        Direct Match:
        - <bullet>
        - <bullet>

        Adjacent Match:
        - <bullet>
        - <bullet>

        Gaps:
        - <bullet>
        - <bullet>

        Recruiter Talking Points:
        - <bullet>
        - <bullet>
        - <bullet>
    """
    return ask_llm(prompt)


def critique_fit_analysis(job_desc: str, draft: str, context: str) -> str:
    prompt = f"""
        You are reviewing a candidate fit analysis for accuracy, grounding, and clarity.

        Job Description:
        {job_desc}

        Candidate Evidence:
        {context}

        Draft Fit Analysis:
        {draft}

        Task:
        Identify only the problems that should be fixed in the draft.

        Strict Rules:
        - Do not rewrite the draft.
        - Do not praise the draft.
        - Do not include a score.
        - Do not include any meta commentary.
        - Focus only on issues that need correction.

        Check for:
        - unsupported or overstated claims
        - confusion between professional experience and project work
        - missing stronger evidence from the candidate context
        - vague or generic wording
        - incorrect gap statements
        - formatting problems
        - sections not requested in the intended final output

        Required Output Format:

        Issues to fix:
        - <issue>
        - <issue>
        - <issue>
    """
    return ask_llm(prompt)


def improve_fit_analysis(job_desc: str, draft: str, critique: str, context: str) -> str:
    prompt = f"""
        You are rewriting a candidate fit analysis into its final version.

        Job Description:
        {job_desc}

        Candidate Evidence:
        {context}

        Previous Draft:
        {draft}

        Issues to Fix:
        {critique}

        Task:
        Rewrite the fit analysis so it is clearer, more accurate, more grounded in evidence, and more recruiter-friendly.

        Strict Rules:
        - Output only the final fit analysis.
        - Do not include commentary, explanations, notes, or meta text.
        - Do not say things like "I made changes", "let me know", or "what's next".
        - Do not include a score.
        - Do not include sections such as "What's good", "What should be improved", or "What's next".
        - Use only information supported by the candidate evidence.
        - Clearly distinguish between professional experience and project work.
        - If something is not directly supported by evidence, label it as adjacent or a gap.
        - Be concise and specific.

        Required Output Format:

        Overall Fit:
        <2-4 sentences>

        Direct Match:
        - <bullet>
        - <bullet>

        Adjacent Match:
        - <bullet>
        - <bullet>

        Gaps:
        - <bullet>
        - <bullet>

        Recruiter Talking Points:
        - <bullet>
        - <bullet>
        - <bullet>
    """
    return ask_llm(prompt)

def run_agent(job_desc: str, top_k: int = 8):
    raw_results = retrieve(job_desc, top_k=top_k)     
    ranked_results = rerank_results(raw_results)
    # for i in ranked_results:
    #     print("Text:",i["text"])
    #     print("Distance:",i["distance"])
          
    context = format_context(ranked_results)
    print("Starting analysis")
    jd_analysis = analyze_jd(job_desc)
    draft = generate_fit_analysis(job_desc, jd_analysis, context)
    critique = critique_fit_analysis(job_desc, draft, context)
    final = improve_fit_analysis(job_desc, draft, critique, context)
   
    # print("jd analysis")
    # print(jd_analysis)
    # print("Context retreived")
    # print(context)
    # print("fir analysis")
    # print(draft)
    # print("critque analysis")
    # print(critique)
    # print("Improvements")
    # print(final)
    

    return {
        "jd_analysis": jd_analysis,
        "retrieved_context": context,
        "draft": draft,
        "critique": critique,
        "final": final,
    }

# run_agent("Looking for a software engineer who can build pipelins for a RAG based LLM Chatbot.")