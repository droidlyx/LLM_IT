import json, tempfile, os
from src.codi_data import CODIDataset, make_codi_collator

def test_dataset_yields_task_reasoning_label():
    rows = [
        {"doc": "Doc A", "e1_text": "X", "e1_type": "T1",
         "e2_text": "Y", "e2_type": "T2",
         "reasoning": "Reason for A", "label": "A"},
        {"doc": "Doc B", "e1_text": "P", "e1_type": "T1",
         "e2_text": "Q", "e2_type": "T2",
         "reasoning": "Reason for B", "label": "B"},
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        path = f.name
    try:
        ds = CODIDataset(path, label_list=["A", "B"])
        assert len(ds) == 2
        item = ds[0]
        assert "task_text" in item and "reasoning" in item and "label_idx" in item
        assert "Doc A" in item["task_text"]
        assert item["label_idx"] == 0
    finally:
        os.unlink(path)

def test_collator_batches():
    items = [
        {"task_text": "T1", "reasoning": "R1", "label_idx": 0},
        {"task_text": "T2", "reasoning": "R2", "label_idx": 1},
    ]
    coll = make_codi_collator()
    batch = coll(items)
    assert batch["task_text"] == ["T1", "T2"]
    assert batch["reasoning"] == ["R1", "R2"]
    assert batch["label_idx"].tolist() == [0, 1]
