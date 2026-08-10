from vectorstore.chroma_client import retrieve_relevant_chunks

def gather_evidence(jd_requirements: dict) -> dict:
    """Retrieve resume chunks relevant to each JD requirement category."""
    evidence = {}
    all_requirements = (
        jd_requirements["must_have_skills"]
        + jd_requirements["nice_to_have_skills"]
        + jd_requirements["key_responsibilities"]
    )
    for req in all_requirements:
        evidence[req] = retrieve_relevant_chunks(req, n_results=3)
    return evidence