"""
Evaluate the RAG system with a set of questions and expected sources.
"""
import json
from statistics import mean
from dotenv import load_dotenv

from . import config, index

# Load environment variables
load_dotenv()


def _reciprocal_rank(results, expected_sources):
    """
    Calculate a score based on how early the expected sources appear in the results.
    """
    for i, hit in enumerate(results, start=1):
        src = (hit.get("metadata") or {}).get("source_file")
        if src in expected_sources:
            return 1.0 / i
    return 0.0


def _hit_at_k(results, expected_sources):
    """
    Calculate a score based on whether the expected sources appear in the results.
    """
    for hit in results:
        src = (hit.get("metadata") or {}).get("source_file")
        if src in expected_sources:
            return 1.0
    return 0.0


def main():
    questions_path = config.PROJECT_ROOT / "eval" / "questions.json"
    data = json.loads(questions_path.read_text(encoding="utf-8"))

    k = 5  # Number of results to return
    rows = []
    hits = []
    mrrs = []  # Mean Reciprocal Rank

    for item in data:
        qid = item["id"]
        question = item["question"]
        expected = set(item["expected_sources"])

        results = index.query_index(question, n_results=k)

        h = _hit_at_k(results, expected)
        r = _reciprocal_rank(results, expected)
        hits.append(h)
        mrrs.append(r)

        top_sources = [
            (hit.get("metadata") or {}).get("source_file", "?")
            for hit in results
        ]

        best_distance = None
        if results and results[0].get("distance") is not None:
            best_distance = float(results[0]["distance"])

        rows.append(
            {
                "id": qid,
                "question": question,
                "expected_sources": list(expected),
                "top_sources": top_sources,
                "hit_at_k": h,
                "reciprocal_rank": r,
                "best_distance": best_distance,
            }
        )

    summary = {
        "num_questions": len(data),
        "k": k,
        "hit_at_k": mean(hits) if hits else 0.0,
        "mrr": mean(mrrs) if mrrs else 0.0,
    }

    print("[EVAL] Summary")
    print(json.dumps(summary, indent=2))

    out_dir = config.PROJECT_ROOT / "eval" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "latest.json"
    out_file.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    print(f"[EVAL] Wrote report to {out_file}.")


if __name__ == "__main__":
    main()