"""Parse BioRED test as full-RE (positive + no_relation) JSONL.

Enumerates ALL valid type-pair candidates per doc (unordered). Pairs with gold
annotation get their relation label; pairs without get label='no_relation'.

Valid type pairs are inferred from the set of entity-type combinations
that appear in any gold relation in the corpus.
"""
import argparse, json
from pathlib import Path
from collections import Counter


def parse_pubtator(path):
    docs, cur = [], None
    with path.open() as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                if cur is not None:
                    docs.append(cur)
                    cur = None
                continue
            if "|t|" in line[:30]:
                pmid, _, title = line.partition("|t|")
                cur = {"pmid": pmid, "title": title, "abstract": "", "ents": {}, "rels": []}
            elif "|a|" in line[:30]:
                pmid, _, abstract = line.partition("|a|")
                cur["abstract"] = abstract
            else:
                parts = line.split("\t")
                if len(parts) == 6:
                    pmid, start, end, text, etype, ident = parts
                    cur["ents"].setdefault(ident, []).append({
                        "start": int(start), "end": int(end), "text": text, "type": etype,
                    })
                elif len(parts) >= 4:
                    cur["rels"].append({"label": parts[1], "a1": parts[2], "a2": parts[3]})
    if cur is not None:
        docs.append(cur)
    return docs


def sentence_window(text, e1_span, e2_span, max_chars=600):
    lo = max(0, min(e1_span[0], e2_span[0]) - 80)
    hi = min(len(text), max(e1_span[1], e2_span[1]) + 80)
    w = text[lo:hi]
    if len(w) > max_chars:
        w = w[:max_chars]
    return w.replace("\n", " ").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    docs = parse_pubtator(Path(args.src))
    print(f"parsed {len(docs)} documents")

    # Build valid unordered type-pair set from gold
    valid_tp = set()
    for d in docs:
        for r in d["rels"]:
            a1, a2 = r["a1"], r["a2"]
            if a1 in d["ents"] and a2 in d["ents"]:
                t1 = d["ents"][a1][0]["type"]
                t2 = d["ents"][a2][0]["type"]
                valid_tp.add(frozenset([t1, t2]))
    print(f"valid type pairs: {len(valid_tp)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    label_counts = Counter()
    n_records = 0
    with out_path.open("w") as f:
        for d in docs:
            pmid = d["pmid"]
            text = (d["title"] + " " + d["abstract"]).strip()
            # Build gold map (unordered)
            gold = {}
            for r in d["rels"]:
                a1, a2 = r["a1"], r["a2"]
                if a1 in d["ents"] and a2 in d["ents"]:
                    key = frozenset([a1, a2])
                    gold[key] = (r["label"], a1, a2)  # keep original direction for reference
            # Enumerate unordered pairs
            ent_ids = list(d["ents"].keys())
            for i, a1 in enumerate(ent_ids):
                for a2 in ent_ids[i+1:]:
                    t1 = d["ents"][a1][0]["type"]
                    t2 = d["ents"][a2][0]["type"]
                    if frozenset([t1, t2]) not in valid_tp:
                        continue
                    key = frozenset([a1, a2])
                    if key in gold:
                        label, ga1, ga2 = gold[key]
                        e1_id, e2_id = ga1, ga2  # preserve gold direction
                    else:
                        label = "no_relation"
                        e1_id, e2_id = a1, a2
                    e1 = d["ents"][e1_id][0]
                    e2 = d["ents"][e2_id][0]
                    window = sentence_window(text, (e1["start"], e1["end"]), (e2["start"], e2["end"]))
                    rec = {
                        "pmid": pmid,
                        "label": label,
                        "e1_id": e1_id,
                        "e2_id": e2_id,
                        "e1_text": e1["text"],
                        "e1_type": e1["type"],
                        "e2_text": e2["text"],
                        "e2_type": e2["type"],
                        "doc": text.replace("\n", " ").strip(),
                        "sentence_window": window,
                    }
                    f.write(json.dumps(rec) + "\n")
                    n_records += 1
                    label_counts[label] += 1

    print(f"WROTE {n_records} records to {args.out}")
    print("LABEL DISTRIBUTION:")
    for k, v in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {v:6d}  {k}")


if __name__ == "__main__":
    main()
