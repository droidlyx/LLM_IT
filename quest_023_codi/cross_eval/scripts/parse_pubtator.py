"""Parse PubTator-format dataset to JSONL matching biored_test schema.

Output schema:
{"pmid": str, "label": str,
 "e1_id": str, "e2_id": str,
 "e1_text": str, "e1_type": str,
 "e2_text": str, "e2_type": str,
 "doc": str  (= title + ' ' + abstract)}

Pubtator format:
  PMID|t|title
  PMID|a|abstract
  PMID\tstart\tend\ttext\ttype\tentity_id     <- mention line (6 fields)
  PMID\tlabel\te1_id\te2_id                    <- relation line (4 fields, label is non-numeric)
  (blank line between abstracts)
"""
import json, sys, argparse
from pathlib import Path
from collections import defaultdict


def parse(path):
    abstracts = {}  # pmid -> dict(title, abstract, mentions: {ent_id -> {text, type}})
    relations = []  # list of {pmid, label, e1_id, e2_id}
    pmid = None
    with open(path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                continue
            if "|t|" in line:
                pmid, _, title = line.split("|", 2)
                abstracts.setdefault(pmid, {"title": "", "abstract": "", "mentions": {}})
                abstracts[pmid]["title"] = title
                continue
            if "|a|" in line:
                pmid, _, abstract = line.split("|", 2)
                abstracts.setdefault(pmid, {"title": "", "abstract": "", "mentions": {}})
                abstracts[pmid]["abstract"] = abstract
                continue
            parts = line.split("\t")
            if len(parts) == 6:
                pid, start, end, text, etype, eid = parts
                abstracts.setdefault(pid, {"title": "", "abstract": "", "mentions": {}})
                # Keep first occurrence per entity id (some datasets list multiple mentions of same ent_id)
                if eid not in abstracts[pid]["mentions"]:
                    abstracts[pid]["mentions"][eid] = {"text": text, "type": etype}
            elif len(parts) == 4 and not parts[1].isdigit():
                pid, label, e1, e2 = parts
                relations.append({"pmid": pid, "label": label, "e1_id": e1, "e2_id": e2})
            elif len(parts) == 4 and parts[1].isdigit():
                # Some datasets have 4-field MENTION lines (PMID, start, end, text/concept).
                # Skip — we use 6-field mentions.
                pass
            # else: silently skip malformed
    return abstracts, relations


def to_jsonl(abstracts, relations, out_path):
    skipped_no_entity = 0
    skipped_no_label = 0
    written = 0
    with open(out_path, "w") as out:
        for rel in relations:
            pid, lbl, e1, e2 = rel["pmid"], rel["label"], rel["e1_id"], rel["e2_id"]
            abs_data = abstracts.get(pid)
            if abs_data is None:
                skipped_no_entity += 1
                continue
            m1 = abs_data["mentions"].get(e1)
            m2 = abs_data["mentions"].get(e2)
            if m1 is None or m2 is None:
                skipped_no_entity += 1
                continue
            if not lbl or lbl.lower() == "none":
                skipped_no_label += 1
                continue
            doc = (abs_data["title"] + " " + abs_data["abstract"]).strip()
            rec = {
                "pmid": pid,
                "label": lbl,
                "e1_id": e1,
                "e2_id": e2,
                "e1_text": m1["text"],
                "e1_type": m1["type"],
                "e2_text": m2["text"],
                "e2_type": m2["type"],
                "doc": doc,
            }
            out.write(json.dumps(rec) + "\n")
            written += 1
    print(f"  wrote {written} records, skipped {skipped_no_entity} (no entity), {skipped_no_label} (no label)")
    return written


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("inp", help="Input pubtator path")
    p.add_argument("out", help="Output jsonl path")
    args = p.parse_args()
    abstracts, relations = parse(args.inp)
    print(f"Parsed {len(abstracts)} abstracts, {len(relations)} relations from {args.inp}")
    written = to_jsonl(abstracts, relations, args.out)
    print(f"-> {args.out}")
