from vectorstore.index import retrieve_relevant_chunks


def gather_evidence(jd_requirements: dict, index: dict) -> dict:
    """Retrieve resume chunks relevant to each JD requirement category."""
    all_requirements = (
        jd_requirements.get("must_have_skills", [])
        + jd_requirements.get("nice_to_have_skills", [])
        + jd_requirements.get("key_responsibilities", [])
    )
    return dict(zip(all_requirements, retrieve_relevant_chunks(all_requirements, index)))
