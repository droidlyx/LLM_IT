"""CODI dataset: yields task_text + reasoning + label_idx per example."""
import json
from typing import List, Optional
import torch
from torch.utils.data import Dataset


def fmt_task(rec: dict) -> str:
    doc = rec.get("doc") or rec.get("sentence_window") or ""
    return (f"Document: {doc}\n\n"
            f"Entity 1: {rec['e1_text']} ({rec.get('e1_type', '')})\n"
            f"Entity 2: {rec['e2_text']} ({rec.get('e2_type', '')})\n\n"
            f"Question: What is the biological relationship between Entity 1 and Entity 2?")


class CODIDataset(Dataset):
    def __init__(self, path: str, label_list: List[str], sample_indices: Optional[List[int]] = None):
        recs = []
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r["label"] not in label_list:
                    continue
                recs.append(r)
        if sample_indices is not None:
            recs = [recs[i] for i in sample_indices]
        self.records = recs
        self.label2idx = {l: i for i, l in enumerate(label_list)}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        return {
            "task_text": fmt_task(r),
            "reasoning": r.get("reasoning", ""),
            "label_idx": self.label2idx[r["label"]],
        }


def make_codi_collator():
    def collate(batch):
        return {
            "task_text": [b["task_text"] for b in batch],
            "reasoning": [b["reasoning"] for b in batch],
            "label_idx": torch.tensor([b["label_idx"] for b in batch], dtype=torch.long),
        }
    return collate
