"""Download BEIR SciFact and convert it to this repo's JSONL format.

NOTE: needs normal internet access (run on your machine). Not exercised in the
build sandbox, so treat it as untested until you run it once.

Output: data/scifact/corpus.jsonl and data/scifact/queries.jsonl
"""
import csv
import io
import json
import urllib.request
import zipfile
from pathlib import Path

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "data" / "scifact"
    out.mkdir(parents=True, exist_ok=True)

    print("Downloading SciFact...")
    with urllib.request.urlopen(URL) as resp:
        zf = zipfile.ZipFile(io.BytesIO(resp.read()))

    names = {n.split("/", 1)[-1]: n for n in zf.namelist()}

    with zf.open(names["corpus.jsonl"]) as f, open(out / "corpus.jsonl", "w", encoding="utf-8") as w:
        for line in f:
            d = json.loads(line)
            text = (d.get("title", "") + ". " + d.get("text", "")).strip(". ")
            w.write(json.dumps({"id": str(d["_id"]), "text": text}) + "\n")

    queries = {}
    with zf.open(names["queries.jsonl"]) as f:
        for line in f:
            q = json.loads(line)
            queries[str(q["_id"])] = q["text"]

    relevant: dict[str, list[str]] = {}
    with zf.open(names["qrels/test.tsv"]) as f:
        reader = csv.reader(io.TextIOWrapper(f, encoding="utf-8"), delimiter="\t")
        next(reader)  # header
        for qid, did, score in reader:
            if int(score) > 0:
                relevant.setdefault(qid, []).append(did)

    with open(out / "queries.jsonl", "w", encoding="utf-8") as w:
        for qid, docs in relevant.items():
            w.write(json.dumps({"id": qid, "text": queries[qid], "relevant": docs}) + "\n")

    print(f"Wrote {len(relevant)} queries to {out}")


if __name__ == "__main__":
    main()
