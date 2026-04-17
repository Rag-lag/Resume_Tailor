from __future__ import annotations

from agent import run_agent
from rag import rebuild_collection


def main() -> None:
    print("Job Agent with ChromaDB")
    print("-----------------------")
    print("1. Rebuild knowledge base")
    print("2. Run JD fit analysis")

    choice = input("Choose 1 or 2: ").strip()

    if choice == "1":
        rebuild_collection()
        print("Knowledge base rebuilt successfully.")
        return

    if choice == "2":
        job_desc = input("\nPaste the job description:\n\n").strip()
        if not job_desc:
            print("No job description provided.")
            return

        result = run_agent(job_desc)

        print("\n========== JD ANALYSIS ==========\n")
        print(result["jd_analysis"])

        print("\n========== RETRIEVED CONTEXT ==========\n")
        print(result["retrieved_context"])

        print("\n========== DRAFT ==========\n")
        print(result["draft"])

        print("\n========== CRITIQUE ==========\n")
        print(result["critique"])

        print("\n========== FINAL ==========\n")
        print(result["final"])
        return

    print("Invalid option.")


if __name__ == "__main__":
    main()