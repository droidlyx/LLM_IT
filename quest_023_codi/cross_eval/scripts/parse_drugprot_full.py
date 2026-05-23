"""Parse DrugProt (TSV format) as full-RE JSONL.

DrugProt format:
  *_abstracs.tsv:   pmid \t title \t abstract
  *_entities.tsv:   pmid \t T# \t TYPE \t start \t end \t text
  *_relations.tsv:  pmid \t LABEL \t Arg1:T# \t Arg2:T#

Enumerates all (CHEMICAL, GENE) pairs per doc. Gold relations get their label;
all other valid type-pair combinations get label='no_relation'.

Valid type pairs (inferred from gold): {CHEMICAL, GENE-N, GENE-Y} cross product
restricted to (CHEMICAL × GENE-*) — but we read from gold to be safe.
"""
import argparse, json
from pathlib import Path
from collections import defaultdict, Counter


def load_abstracts(path):
    out = {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                pmid, title, abstract = parts[0], parts[1], parts[2]
                out[pmid] = f"{title} {abstract}".strip()
    return out


def load_entities(path):
    """Return {pmid: {Tid: {type, start, end, text}}}"""
    out = defaultdict(dict)
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 6:
                pmid, tid, etype, start, end, text = parts[:6]
                out[pmid][tid] = {
                    "type": etype, "start": int(start), "end": int(end), "text": text,
                }
    return out


def load_relations(path):
    """Return {pmid: [(label, t1, t2)]}"""
    out = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                pmid, label, a1, a2 = parts[:4]
                # Strip "Arg1:" / "Arg2:" prefix
                t1 = a1.split(":")[1] if ":" in a1 else a1
                t2 = a2.split(":")[1] if ":" in a2 else a2
                out[pmid].append((label, t1, t2))
    return out


def sentence_window(text, e1_span, e2_span, max_chars=600):
    lo = max(0, min(e1_span[0], e2_span[0]) - 80)
    hi = min(len(text), max(e1_span[1], e2_span[1]) + 80)
    w = text[lo:hi]
    if len(w) > max_chars:
        w = w[:max_chars]
    return w.replace("\n", " ").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--abstracts", required=True)
    ap.add_argument("--entities", required=True)
    ap.add_argument("--relations", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    abstracts = load_abstracts(args.abstracts)
    entities = load_entities(args.entities)
    relations = load_relations(args.relations)
    print(f"abstracts: {len(abstracts)}, ent-pmids: {len(entities)}, rel-pmids: {len(relations)}")

    # Determine valid type pairs from gold (unordered)
    valid_tp = set()
    for pmid, rels in relations.items():
        for label, t1, t2 in rels:
            if t1 in entities.get(pmid, {}) and t2 in entities.get(pmid, {}):
                et1 = entities[pmid][t1]["type"]
                et2 = entities[pmid][t2]["type"]
                valid_tp.add(frozenset([et1, et2]))
    print(f"valid type pairs: {valid_tp}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    label_counts = Counter()
    n_records = 0
    with out_path.open("w") as f:
        for pmid, doc_ents in entities.items():
            if pmid not in abstracts:
                continue
            text = abstracts[pmid]
            # Build gold map (unordered by Tid)
            gold = {}
            for label, t1, t2 in relations.get(pmid, []):
                if t1 in doc_ents and t2 in doc_ents:
                    key = frozenset([t1, t2])
                    gold[key] = (label, t1, t2)
            # Enumerate unordered pairs
            tids = list(doc_ents.keys())
            for i, a1 in enumerate(tids):
                for a2 in tids[i+1:]:
                    et1 = doc_ents[a1]["type"]
                    et2 = doc_ents[a2]["type"]
                    if frozenset([et1, et2]) not in valid_tp:
                        continue
                    key = frozenset([a1, a2])
                    if key in gold:
                        label, ga1, ga2 = gold[key]
                        e1_id, e2_id = ga1, ga2
                    else:
                        label = "no_relation"
                        e1_id, e2_id = a1, a2
                    e1 = doc_ents[e1_id]
                    e2 = doc_ents[e2_id]
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
