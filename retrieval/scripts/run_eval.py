"""Run the retrieval ablation: bm25 vs dense vs hybrid vs hybrid+rerank.

Examples:
    python scripts/run_eval.py                                  # offline sample data
    python scripts/run_eval.py --corpus data/scifact/corpus.jsonl \
        --queries data/scifact/queries.jsonl --embedder st --reranker cross
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragv2.eval.harness import format_table, load_jsonl, run_eval  # noqa: E402
from ragv2.factory import build_retriever  # noqa: E402


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default=str(root / "data/sample/corpus.jsonl"))
    p.add_argument("--queries", default=str(root / "data/sample/queries.jsonl"))
    p.add_argument("--embedder", default="hashing", choices=["hashing", "st"])
    p.add_argument("--reranker", default="lexical", choices=["none", "lexical", "cross"])
    p.add_argument("--cross-model", help="Hugging Face cross-encoder for --reranker cross, e.g. BAAI/bge-reranker-base.")
    p.add_argument("--max-words", type=int, default=120)
    p.add_argument("--overlap", type=int, default=20)
    p.add_argument("--candidates", type=int, default=50, help="Candidates retrieved per index before fusion.")
    p.add_argument("--rrf-k", type=int, default=60, help="RRF rank constant for hybrid modes.")
    p.add_argument("--weights", type=float, nargs=2, default=[1.0, 1.0], metavar=("BM25", "DENSE"),
                   help="RRF weights for the bm25 and dense rankings.")
    p.add_argument("--fetch-k", type=int, default=50, help="Results fetched before document-level scoring.")
    p.add_argument("--recall-k", type=int, nargs="+", default=[5, 10], help="Recall cutoffs to report.")
    p.add_argument("--output", help="Optional path for the machine-readable JSON report.")
    args = p.parse_args()

    retriever = build_retriever(
        args.embedder,
        args.reranker,
        cross_model=args.cross_model,
        weights=tuple(args.weights),
        max_words=args.max_words,
        overlap=args.overlap,
        candidates=args.candidates,
        rrf_k=args.rrf_k,
    )
    n = retriever.add_documents(load_jsonl(args.corpus))
    queries = load_jsonl(args.queries)
    print(f"Indexed {n} chunks; {len(queries)} queries; embedder={args.embedder} reranker={args.reranker}\n")
    report = run_eval(
        retriever,
        queries,
        recall_ks=tuple(args.recall_k),
        fetch_k=args.fetch_k,
    )
    print(format_table(report))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote report to {output}")


if __name__ == "__main__":
    main()
