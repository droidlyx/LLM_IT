# Quest 023 CODI-Bi Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Silent CoT v1's single-token bottleneck with CODI-style multi-token latent block + joint teacher/student training. Validate via smoke v5 on the existing 11K synth dataset against full-RE eval (BioRED, BioTriplex, DrugProt).

**Architecture:** Add `<think>`/`<answer>` special tokens + k=4 learnable latent embeddings between them. Teacher branch sees reasoning text, student branch sees the k learnable latents. Both branches pool the last k+1 positions to produce a judgment vector → cosine to label embeddings. 4-term joint loss: `L_cls_S + α·L_cls_T + β·L_align + γ·L_distill` with linear warmup of β,γ over steps [200, 400].

**Tech Stack:** Qwen3-4B-Base, peft LoRA r=16, bf16 mixed precision, flash-attention-2, PyTorch 2.x.

**Reference base:** `/home/ds/DeepScientist/quests/022/.ds/worktrees/idea-idea-7ffb9515/src/` (read-only — copy patterns, don't modify).

**Implementation root:** `/home/ds/quest_023_codi/` (new directory; migratable to a DS quest worktree once smoke v5 passes).

**Data (frozen, do not regenerate):**
- `/tmp/synth_smoke/training_v4/synth_train.jsonl` (11,290 multi-schema synth samples with reasoning + label)
- `/tmp/synth_smoke/training_v4/label_dict.json` (27 labels × 5 templates)

**Eval data (full-RE only, positive-only is forbidden):**
- BioRED: `/home/ds/cross_eval/data/biored_full_test.jsonl` (6832 records)
- BioTriplex: `/home/ds/cross_eval/data/biotriplex_test.jsonl` (232 records)
- DrugProt: `/home/ds/cross_eval/data/drugprot_full_test.jsonl` (116K records, subsample 5K for smoke)

**Spec:** `/home/ds/distill_021_data/quest_023_codi_design.md`

---

### Task 1: Workspace setup

**Files:**
- Create: `/home/ds/quest_023_codi/src/__init__.py`
- Create: `/home/ds/quest_023_codi/tests/__init__.py`
- Create: `/home/ds/quest_023_codi/conftest.py`
- Create: `/home/ds/quest_023_codi/pyproject.toml`

- [ ] **Step 1: Create directory tree**

```bash
mkdir -p /home/ds/quest_023_codi/{src,tests,scripts,artifacts}
touch /home/ds/quest_023_codi/src/__init__.py
touch /home/ds/quest_023_codi/tests/__init__.py
```

- [ ] **Step 2: Add conftest.py for test fixtures**

```python
# /home/ds/quest_023_codi/conftest.py
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
```

- [ ] **Step 3: Add pyproject.toml so pytest finds the package**

```toml
# /home/ds/quest_023_codi/pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

- [ ] **Step 4: Verify pytest runs (no tests yet)**

```bash
cd /home/ds/quest_023_codi && pytest
```

Expected: `no tests ran` (exit code 5 is OK at this stage).

- [ ] **Step 5: Commit**

```bash
cd /home/ds/quest_023_codi && git init -q && git add . && \
git -c user.email=ds@local -c user.name=ds commit -q -m "scaffold: quest 023 codi workspace"
```

---

### Task 2: Special token + embedding resize

**Files:**
- Create: `/home/ds/quest_023_codi/src/codi_tokenizer.py`
- Test: `/home/ds/quest_023_codi/tests/test_codi_tokenizer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_codi_tokenizer.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.codi_tokenizer import add_codi_specials, init_special_embeddings

QWEN = "/root/shared-nvme/ds-workspace/Qwen3-4B-Base"

def test_add_specials_returns_ids():
    tok = AutoTokenizer.from_pretrained(QWEN, trust_remote_code=True)
    base_n = len(tok)
    think_id, answer_id = add_codi_specials(tok)
    assert len(tok) == base_n + 2
    assert tok.convert_ids_to_tokens(think_id) == "<think>"
    assert tok.convert_ids_to_tokens(answer_id) == "<answer>"

def test_init_special_embeddings_changes_rows():
    tok = AutoTokenizer.from_pretrained(QWEN, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(QWEN, dtype=torch.bfloat16, trust_remote_code=True)
    think_id, answer_id = add_codi_specials(tok)
    model.resize_token_embeddings(len(tok))
    emb = model.get_input_embeddings()
    before_t = emb.weight[think_id].clone()
    init_special_embeddings(model, tok, think_id, answer_id)
    after_t = emb.weight[think_id]
    assert not torch.allclose(before_t, after_t)
    assert after_t.norm().item() > 0
```

- [ ] **Step 2: Run to verify fail**

```bash
cd /home/ds/quest_023_codi && pytest tests/test_codi_tokenizer.py -v
```

Expected: `ImportError: cannot import name 'add_codi_specials'`.

- [ ] **Step 3: Implement**

```python
# src/codi_tokenizer.py
"""Add <think> and <answer> as special tokens; init their embeddings from related vocab."""
import torch

THINK = "<think>"
ANSWER = "<answer>"

def add_codi_specials(tokenizer):
    """Returns (think_id, answer_id). Caller is responsible for resize_token_embeddings on the model."""
    new = []
    for tok in [THINK, ANSWER]:
        if tok not in tokenizer.get_vocab():
            new.append(tok)
    if new:
        tokenizer.add_special_tokens({"additional_special_tokens": new})
    return tokenizer.convert_tokens_to_ids(THINK), tokenizer.convert_tokens_to_ids(ANSWER)


@torch.no_grad()
def init_special_embeddings(model, tokenizer, think_id: int, answer_id: int):
    """Set <think> ~ mean(['.', 'think', 'consider']); <answer> ~ mean(['the', 'answer', 'is'])."""
    emb = model.get_input_embeddings()
    def _mean_of(words):
        ids = []
        for w in words:
            sub_ids = tokenizer(w, add_special_tokens=False)["input_ids"]
            ids.extend(sub_ids)
        return emb.weight[ids].mean(dim=0)
    emb.weight[think_id] = _mean_of([".", "think", "consider"]).to(emb.weight.dtype)
    emb.weight[answer_id] = _mean_of(["the", "answer", "is"]).to(emb.weight.dtype)
```

- [ ] **Step 4: Run, verify pass**

```bash
cd /home/ds/quest_023_codi && pytest tests/test_codi_tokenizer.py -v
```

Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/ds/quest_023_codi && git add . && \
git -c user.email=ds@local -c user.name=ds commit -q -m "feat: codi special tokens + embedding init"
```

---

### Task 3: Learnable latent embeddings module

**Files:**
- Create: `/home/ds/quest_023_codi/src/codi_latents.py`
- Test: `/home/ds/quest_023_codi/tests/test_codi_latents.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_codi_latents.py
import torch
from src.codi_latents import LearnableLatents

def test_latents_shape():
    lat = LearnableLatents(k=4, d_model=2560)
    out = lat(batch_size=3)
    assert out.shape == (3, 4, 2560)

def test_latents_requires_grad():
    lat = LearnableLatents(k=4, d_model=2560)
    assert lat.latents.requires_grad

def test_latents_init_small():
    lat = LearnableLatents(k=4, d_model=2560)
    # std should be ~0.02
    assert 0.005 < lat.latents.std().item() < 0.05
```

- [ ] **Step 2: Run, verify fail**

```bash
cd /home/ds/quest_023_codi && pytest tests/test_codi_latents.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement**

```python
# src/codi_latents.py
import torch
import torch.nn as nn


class LearnableLatents(nn.Module):
    """k trainable d-dim embeddings expanded along the batch dim on call."""
    def __init__(self, k: int, d_model: int, init_std: float = 0.02):
        super().__init__()
        self.k = k
        self.d_model = d_model
        self.latents = nn.Parameter(torch.randn(k, d_model) * init_std)

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.latents.unsqueeze(0).expand(batch_size, -1, -1)
```

- [ ] **Step 4: Run, verify pass**

```bash
cd /home/ds/quest_023_codi && pytest tests/test_codi_latents.py -v
```

- [ ] **Step 5: Commit**

```bash
cd /home/ds/quest_023_codi && git add . && \
git -c user.email=ds@local -c user.name=ds commit -q -m "feat: learnable latent embeddings module"
```

---

### Task 4: CODI model loader

**Files:**
- Create: `/home/ds/quest_023_codi/src/codi_model.py`
- Test: `/home/ds/quest_023_codi/tests/test_codi_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_codi_model.py
import torch
from src.codi_model import load_codi_trainable

def test_load_returns_components():
    model, backbone, tok, latents, ids = load_codi_trainable(k=4, lora_r=8)
    assert "<think>" in tok.get_vocab()
    assert "<answer>" in tok.get_vocab()
    assert latents.k == 4
    assert ids["think"] >= 0 and ids["answer"] >= 0
    # LoRA params trainable + latents trainable
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable > 0
    assert latents.latents.requires_grad
```

- [ ] **Step 2: Run, verify fail**

```bash
cd /home/ds/quest_023_codi && pytest tests/test_codi_model.py -v
```

- [ ] **Step 3: Implement**

```python
# src/codi_model.py
"""LoRA-wrapped Qwen3-4B-Base with CODI specials + learnable latents."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

from .codi_tokenizer import add_codi_specials, init_special_embeddings
from .codi_latents import LearnableLatents

QWEN = "/root/shared-nvme/ds-workspace/Qwen3-4B-Base"


def load_tokenizer(path=QWEN):
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_codi_trainable(path: str = QWEN, k: int = 4, lora_r: int = 16,
                         lora_alpha: int = 32, lora_dropout: float = 0.05,
                         dtype=torch.bfloat16, device: str = "cuda"):
    """Returns (peft_model, backbone, tokenizer, latents, special_ids_dict)."""
    tok = load_tokenizer(path)
    think_id, answer_id = add_codi_specials(tok)
    base = AutoModelForCausalLM.from_pretrained(
        path, dtype=dtype, attn_implementation="flash_attention_2", trust_remote_code=True
    )
    base.resize_token_embeddings(len(tok))
    init_special_embeddings(base, tok, think_id, answer_id)

    lora = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["embed_tokens"],  # let resized embed train (covers <think>/<answer>)
        bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base, lora)
    backbone = model.base_model.model.model
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    backbone.enable_input_require_grads()
    model.to(device)

    d_model = base.config.hidden_size
    latents = LearnableLatents(k=k, d_model=d_model).to(device=device, dtype=dtype)

    return model, backbone, tok, latents, {"think": think_id, "answer": answer_id}
```

- [ ] **Step 4: Run, verify pass**

```bash
cd /home/ds/quest_023_codi && pytest tests/test_codi_model.py -v
```

- [ ] **Step 5: Commit**

```bash
cd /home/ds/quest_023_codi && git add . && \
git -c user.email=ds@local -c user.name=ds commit -q -m "feat: codi model loader with LoRA + latents"
```

---

### Task 5: Input builders (teacher + student branches)

**Files:**
- Create: `/home/ds/quest_023_codi/src/codi_inputs.py`
- Test: `/home/ds/quest_023_codi/tests/test_codi_inputs.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_codi_inputs.py
import torch
from src.codi_model import load_codi_trainable
from src.codi_inputs import build_student_inputs, build_teacher_inputs, pool_positions

def _setup():
    return load_codi_trainable(k=4, lora_r=8)

def test_student_input_shapes():
    model, backbone, tok, latents, ids = _setup()
    task_texts = ["Document: foo. Question: rel?", "Document: bar. Question: rel?"]
    inputs_embeds, attn, pool_idx = build_student_inputs(
        task_texts, tok, model, latents, ids, max_task_len=64
    )
    B = 2
    assert inputs_embeds.shape[0] == B
    # 1 think + 4 latents + 1 answer = 6 extra tokens past task
    assert inputs_embeds.shape[1] >= 6
    # pool_idx covers k+1 = 5 positions per row
    assert pool_idx.shape == (B, 5)

def test_teacher_input_pool_idx():
    model, backbone, tok, latents, ids = _setup()
    task_texts = ["Document: foo. Question: rel?"]
    reasonings = ["The entities show upregulation."]
    inputs_embeds, attn, pool_idx = build_teacher_inputs(
        task_texts, reasonings, tok, model, ids, k=4, max_task_len=64, max_reason_len=32
    )
    # 1 think + reasoning tokens + 1 answer; pool over last k=4 + answer = 5 positions
    assert pool_idx.shape == (1, 5)
```

- [ ] **Step 2: Run, verify fail**

```bash
cd /home/ds/quest_023_codi && pytest tests/test_codi_inputs.py -v
```

- [ ] **Step 3: Implement**

```python
# src/codi_inputs.py
"""Build inputs_embeds + attention_mask + pool position indices for teacher/student branches.

Both branches share the layout: [task_input] [<think>] [...] [<answer>].
Student: middle = k learnable latents (inserted as embeddings, not tokens).
Teacher: middle = reasoning text tokens.

Padding is LEFT (consistent with quest 022 convention).
Pool positions are the last k+1 token positions of each row (the latents/last-k-reason + <answer>).
"""
from typing import List, Tuple
import torch


def _left_pad_ids(rows: List[List[int]], pad_id: int):
    L = max(len(r) for r in rows)
    out_ids = torch.full((len(rows), L), pad_id, dtype=torch.long)
    out_mask = torch.zeros((len(rows), L), dtype=torch.long)
    for i, r in enumerate(rows):
        out_ids[i, L - len(r):] = torch.tensor(r, dtype=torch.long)
        out_mask[i, L - len(r):] = 1
    return out_ids, out_mask


def _truncate_left(ids: List[int], max_len: int):
    return ids[-max_len:] if len(ids) > max_len else ids


def build_student_inputs(task_texts: List[str], tok, model, latents, ids: dict,
                          max_task_len: int = 600):
    """
    Output:
      inputs_embeds: (B, L, D)  — task_emb || think_emb || k latent_emb || answer_emb
      attention_mask: (B, L)
      pool_idx: (B, k+1) — absolute positions of the k latents + <answer>
    """
    device = next(model.parameters()).device
    emb_layer = model.get_input_embeddings()
    k = latents.k

    # Tokenize tasks
    task_ids_list = []
    for t in task_texts:
        tids = tok(t, add_special_tokens=False, truncation=False)["input_ids"]
        task_ids_list.append(_truncate_left(tids, max_task_len))
    pad_id = tok.pad_token_id
    task_ids, task_mask = _left_pad_ids(task_ids_list, pad_id)
    task_ids = task_ids.to(device); task_mask = task_mask.to(device)

    # Embed task
    task_emb = emb_layer(task_ids)  # (B, L_task, D)
    B, L_task, D = task_emb.shape

    think_id = ids["think"]; answer_id = ids["answer"]
    think_emb = emb_layer(torch.tensor([think_id], device=device)).expand(B, 1, D)
    answer_emb = emb_layer(torch.tensor([answer_id], device=device)).expand(B, 1, D)
    latent_emb = latents(B).to(task_emb.dtype)  # (B, k, D)

    inputs_embeds = torch.cat([task_emb, think_emb, latent_emb, answer_emb], dim=1)  # (B, L_task+k+2, D)

    suffix_mask = torch.ones(B, k + 2, dtype=torch.long, device=device)
    attention_mask = torch.cat([task_mask, suffix_mask], dim=1)

    L = inputs_embeds.shape[1]
    # Pool positions: last k+1 absolute indices, but accounting for left-padding
    # Since think+latents+answer are appended (right side) and never padded,
    # they sit at the absolute tail: positions [L-(k+1)..L-1] are latents+answer.
    # That works for all rows because the suffix is identical length.
    pool_idx = torch.arange(L - (k + 1), L, device=device).unsqueeze(0).expand(B, -1)

    return inputs_embeds, attention_mask, pool_idx


def build_teacher_inputs(task_texts: List[str], reasonings: List[str], tok, model,
                          ids: dict, k: int = 4, max_task_len: int = 600,
                          max_reason_len: int = 400):
    """
    Output: same shape contract as student. Middle = reasoning tokens.
    Pool positions = last (k+1) positions = last k reasoning tokens + <answer>.
    """
    device = next(model.parameters()).device
    emb_layer = model.get_input_embeddings()
    pad_id = tok.pad_token_id
    think_id = ids["think"]; answer_id = ids["answer"]

    rows = []
    for t, r in zip(task_texts, reasonings):
        tids = _truncate_left(tok(t, add_special_tokens=False)["input_ids"], max_task_len)
        rids = tok(r, add_special_tokens=False)["input_ids"][:max_reason_len]
        # Enforce reasoning has at least k tokens (pad with answer_id if too short)
        if len(rids) < k:
            rids = rids + [answer_id] * (k - len(rids))
        rows.append(tids + [think_id] + rids + [answer_id])

    full_ids, full_mask = _left_pad_ids(rows, pad_id)
    full_ids = full_ids.to(device); full_mask = full_mask.to(device)
    inputs_embeds = emb_layer(full_ids)

    B, L = full_ids.shape
    # Last position is <answer>. The k tokens before it are the last k reasoning tokens.
    pool_idx = torch.arange(L - (k + 1), L, device=device).unsqueeze(0).expand(B, -1)

    return inputs_embeds, full_mask, pool_idx


def pool_positions(hidden_states: torch.Tensor, pool_idx: torch.Tensor) -> torch.Tensor:
    """hidden_states (B, L, D); pool_idx (B, P). Return (B, D) mean-pooled."""
    B, P = pool_idx.shape
    D = hidden_states.shape[-1]
    gather_idx = pool_idx.unsqueeze(-1).expand(-1, -1, D)
    picked = torch.gather(hidden_states, dim=1, index=gather_idx)  # (B, P, D)
    return picked.mean(dim=1)
```

- [ ] **Step 4: Run, verify pass**

```bash
cd /home/ds/quest_023_codi && pytest tests/test_codi_inputs.py -v
```

- [ ] **Step 5: Commit**

```bash
cd /home/ds/quest_023_codi && git add . && \
git -c user.email=ds@local -c user.name=ds commit -q -m "feat: codi teacher/student input builders + pool"
```

---

### Task 6: Label cache with CODI template

**Files:**
- Create: `/home/ds/quest_023_codi/src/codi_label_cache.py`
- Test: `/home/ds/quest_023_codi/tests/test_codi_label_cache.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_codi_label_cache.py
import torch
from src.codi_model import load_codi_trainable
from src.codi_label_cache import encode_label_templates_codi

def test_encode_shape():
    model, backbone, tok, latents, ids = load_codi_trainable(k=4, lora_r=8)
    label_dict = {
        "A": ["positive case template one", "positive case template two"],
        "B": ["negative case template one", "negative case template two"],
    }
    mean = torch.zeros(model.config.hidden_size, device="cuda")
    embs = encode_label_templates_codi(backbone, tok, model, latents, ids,
                                         label_dict, mean, device="cuda")
    # (n_labels, n_templates_per_label, D)
    assert embs.shape == (2, 2, model.config.hidden_size)
    # L2 normalized after centering
    norms = embs.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
```

- [ ] **Step 2: Run, verify fail**

```bash
cd /home/ds/quest_023_codi && pytest tests/test_codi_label_cache.py -v
```

- [ ] **Step 3: Implement**

```python
# src/codi_label_cache.py
"""Encode each label template via [text] [<think>] [k latents] [<answer>] and pool same way as judgments."""
import random
import torch
import torch.nn.functional as F

from .codi_inputs import build_student_inputs, pool_positions


@torch.no_grad()
def encode_label_templates_codi(backbone, tok, model, latents, ids: dict,
                                  label_dict: dict, mean: torch.Tensor,
                                  device: str = "cuda") -> torch.Tensor:
    """Returns tensor of shape (n_labels, max_templates_per_label, D), centered + L2 normalized."""
    labels = list(label_dict.keys())
    max_t = max(len(label_dict[l]) for l in labels)
    embs_per_label = []
    for lbl in labels:
        templates = label_dict[lbl]
        inputs_embeds, attn, pool_idx = build_student_inputs(
            templates, tok, model, latents, ids, max_task_len=64
        )
        out = backbone(inputs_embeds=inputs_embeds, attention_mask=attn, use_cache=False)
        h_pool = pool_positions(out.last_hidden_state.float(), pool_idx)  # (T, D)
        centered = h_pool - mean.to(h_pool.dtype).unsqueeze(0)
        normalized = F.normalize(centered, p=2, dim=-1)
        # Pad to max_t by repeating last template
        if normalized.shape[0] < max_t:
            pad = normalized[-1:].expand(max_t - normalized.shape[0], -1)
            normalized = torch.cat([normalized, pad], dim=0)
        embs_per_label.append(normalized)
    return torch.stack(embs_per_label, dim=0)  # (n_labels, max_t, D)


class CODILabelCache:
    """Per-step sampler: returns one template embedding per label."""
    def __init__(self, label_dict, embs_all):
        self.labels = list(label_dict.keys())
        self.embs_all = embs_all  # (n_labels, max_t, D)
        self.n_templates = {i: len(label_dict[l]) for i, l in enumerate(self.labels)}

    def sample_per_label(self, rng: random.Random) -> torch.Tensor:
        rows = []
        for i in range(len(self.labels)):
            t = rng.randrange(self.n_templates[i])
            rows.append(self.embs_all[i, t])
        return torch.stack(rows, dim=0)  # (n_labels, D)
```

- [ ] **Step 4: Run, verify pass**

```bash
cd /home/ds/quest_023_codi && pytest tests/test_codi_label_cache.py -v
```

- [ ] **Step 5: Commit**

```bash
cd /home/ds/quest_023_codi && git add . && \
git -c user.email=ds@local -c user.name=ds commit -q -m "feat: codi label cache (template + latents + answer)"
```

---

### Task 7: CODI training step with 4-loss + warmup

**Files:**
- Create: `/home/ds/quest_023_codi/src/codi_losses.py`
- Test: `/home/ds/quest_023_codi/tests/test_codi_losses.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_codi_losses.py
import torch
import random
from src.codi_model import load_codi_trainable
from src.codi_label_cache import encode_label_templates_codi, CODILabelCache
from src.codi_losses import codi_step, CODILossConfig

def test_one_step_runs_no_nan():
    model, backbone, tok, latents, ids = load_codi_trainable(k=4, lora_r=8)
    label_dict = {"A": ["template a one", "template a two"],
                  "B": ["template b one", "template b two"]}
    mean = torch.zeros(model.config.hidden_size, device="cuda")
    embs_all = encode_label_templates_codi(backbone, tok, model, latents, ids,
                                              label_dict, mean, device="cuda")
    cache = CODILabelCache(label_dict, embs_all)
    batch = {
        "task_text": ["Question 1?", "Question 2?"],
        "reasoning": ["This is reasoning one ok ok ok.", "Another reasoning two."],
        "label_idx": torch.tensor([0, 1]),
    }
    cfg = CODILossConfig(k=4, tau=0.07, alpha=0.5, beta=1.0, gamma=0.5,
                          warmup_steps=200, ramp_steps=200)
    rng = random.Random(0)
    label_embs = cache.sample_per_label(rng).to("cuda")
    loss, log = codi_step(model, backbone, latents, tok, ids, batch,
                            label_embs, mean, cfg, step=0)
    assert torch.isfinite(loss)
    assert log["L_cls_S"] > 0
    # At step 0, β and γ should be 0 (warmup)
    assert log["beta_eff"] == 0.0 and log["gamma_eff"] == 0.0

def test_warmup_ramp():
    cfg = CODILossConfig(k=4, warmup_steps=200, ramp_steps=200, beta=1.0, gamma=0.5)
    from src.codi_losses import _warmup_weights
    # Before warmup: 0
    assert _warmup_weights(0, cfg) == (0.0, 0.0)
    # End of warmup, ramp start: still 0
    assert _warmup_weights(200, cfg) == (0.0, 0.0)
    # Middle of ramp: 50% of target
    b, g = _warmup_weights(300, cfg)
    assert abs(b - 0.5) < 1e-4 and abs(g - 0.25) < 1e-4
    # End of ramp: full
    b, g = _warmup_weights(400, cfg)
    assert abs(b - 1.0) < 1e-4 and abs(g - 0.5) < 1e-4
```

- [ ] **Step 2: Run, verify fail**

```bash
cd /home/ds/quest_023_codi && pytest tests/test_codi_losses.py -v
```

- [ ] **Step 3: Implement**

```python
# src/codi_losses.py
"""CODI training step: teacher branch + student branch + 4-loss with warmup."""
from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn.functional as F

from .codi_inputs import build_student_inputs, build_teacher_inputs, pool_positions


@dataclass
class CODILossConfig:
    k: int = 4
    tau: float = 0.07
    alpha: float = 0.5
    beta: float = 1.0
    gamma: float = 0.5
    warmup_steps: int = 200
    ramp_steps: int = 200
    max_task_len: int = 600
    max_reason_len: int = 400


def _warmup_weights(step: int, cfg: CODILossConfig) -> Tuple[float, float]:
    """Linear ramp of β and γ over [warmup_steps, warmup_steps + ramp_steps]."""
    if step < cfg.warmup_steps:
        return 0.0, 0.0
    if step >= cfg.warmup_steps + cfg.ramp_steps:
        return cfg.beta, cfg.gamma
    frac = (step - cfg.warmup_steps) / cfg.ramp_steps
    return cfg.beta * frac, cfg.gamma * frac


def _cosine_logits(judgment: torch.Tensor, label_embs: torch.Tensor, mean: torch.Tensor,
                    tau: float) -> torch.Tensor:
    centered = judgment - mean.to(judgment.dtype).unsqueeze(0)
    rel_norm = F.normalize(centered, p=2, dim=1)
    return (rel_norm @ label_embs.t()) / tau


def codi_step(model, backbone, latents, tok, ids, batch, label_embs: torch.Tensor,
               mean: torch.Tensor, cfg: CODILossConfig, step: int):
    """One training step with teacher + student forwards and 4-loss."""
    device = next(model.parameters()).device
    label_embs = label_embs.to(device=device, dtype=torch.float32)
    mean = mean.to(device=device, dtype=torch.float32)
    gold = batch["label_idx"].to(device)

    # ---- Student forward ----
    s_embeds, s_mask, s_pool = build_student_inputs(
        batch["task_text"], tok, model, latents, ids, max_task_len=cfg.max_task_len
    )
    s_out = backbone(inputs_embeds=s_embeds, attention_mask=s_mask, use_cache=False)
    s_judgment = pool_positions(s_out.last_hidden_state.float(), s_pool)  # (B, D)

    # ---- Teacher forward ----
    t_embeds, t_mask, t_pool = build_teacher_inputs(
        batch["task_text"], batch["reasoning"], tok, model, ids,
        k=cfg.k, max_task_len=cfg.max_task_len, max_reason_len=cfg.max_reason_len
    )
    t_out = backbone(inputs_embeds=t_embeds, attention_mask=t_mask, use_cache=False)
    t_judgment = pool_positions(t_out.last_hidden_state.float(), t_pool)

    # ---- Logits ----
    logits_S = _cosine_logits(s_judgment, label_embs, mean, cfg.tau)
    logits_T = _cosine_logits(t_judgment, label_embs, mean, cfg.tau)

    # ---- Losses ----
    L_cls_S = F.cross_entropy(logits_S, gold)
    L_cls_T = F.cross_entropy(logits_T, gold)

    # Align in centered+normalized vec space (after both have been centered)
    s_centered_norm = F.normalize(s_judgment - mean.unsqueeze(0), p=2, dim=1)
    with torch.no_grad():
        t_centered_norm = F.normalize(t_judgment - mean.unsqueeze(0), p=2, dim=1)
    L_align = F.mse_loss(s_centered_norm, t_centered_norm)

    with torch.no_grad():
        p_T = F.softmax(logits_T.detach(), dim=-1)
    log_p_S = F.log_softmax(logits_S, dim=-1)
    L_distill = F.kl_div(log_p_S, p_T, reduction="batchmean")

    beta_eff, gamma_eff = _warmup_weights(step, cfg)
    total = L_cls_S + cfg.alpha * L_cls_T + beta_eff * L_align + gamma_eff * L_distill

    with torch.no_grad():
        acc_S = (logits_S.argmax(-1) == gold).float().mean().item()
        acc_T = (logits_T.argmax(-1) == gold).float().mean().item()
        align_gap = (s_centered_norm - t_centered_norm).norm(dim=-1).mean().item()

    log = {
        "L_cls_S": float(L_cls_S.detach()),
        "L_cls_T": float(L_cls_T.detach()),
        "L_align": float(L_align.detach()),
        "L_distill": float(L_distill.detach()),
        "total": float(total.detach()),
        "acc_S": acc_S, "acc_T": acc_T,
        "align_gap": align_gap,
        "beta_eff": beta_eff, "gamma_eff": gamma_eff,
    }
    return total, log
```

- [ ] **Step 4: Run, verify pass**

```bash
cd /home/ds/quest_023_codi && pytest tests/test_codi_losses.py -v
```

- [ ] **Step 5: Commit**

```bash
cd /home/ds/quest_023_codi && git add . && \
git -c user.email=ds@local -c user.name=ds commit -q -m "feat: codi step with 4-loss + warmup ramp"
```

---

### Task 8: Dataset + collator

**Files:**
- Create: `/home/ds/quest_023_codi/src/codi_data.py`
- Test: `/home/ds/quest_023_codi/tests/test_codi_data.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_codi_data.py
import json, tempfile, os
from src.codi_data import CODIDataset, make_codi_collator

def test_dataset_yields_task_reasoning_label():
    # Create a tiny JSONL
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
```

- [ ] **Step 2: Run, verify fail**

```bash
cd /home/ds/quest_023_codi && pytest tests/test_codi_data.py -v
```

- [ ] **Step 3: Implement**

```python
# src/codi_data.py
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
```

- [ ] **Step 4: Run, verify pass**

```bash
cd /home/ds/quest_023_codi && pytest tests/test_codi_data.py -v
```

- [ ] **Step 5: Commit**

```bash
cd /home/ds/quest_023_codi && git add . && \
git -c user.email=ds@local -c user.name=ds commit -q -m "feat: codi dataset + collator"
```

---

### Task 9: Mean computation script

**Files:**
- Create: `/home/ds/quest_023_codi/scripts/compute_mean.py`

- [ ] **Step 1: Implement script**

```python
# scripts/compute_mean.py
"""Compute global mean over all label templates via base model + CODI student template."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from src.codi_model import load_codi_trainable
from src.codi_inputs import build_student_inputs, pool_positions

LABEL_DICT = "/tmp/synth_smoke/training_v4/label_dict.json"
OUT_MEAN = "/home/ds/quest_023_codi/artifacts/mean.pt"

os.makedirs(os.path.dirname(OUT_MEAN), exist_ok=True)

print("Loading model + latents (fresh init)...", flush=True)
model, backbone, tok, latents, ids = load_codi_trainable(k=4, lora_r=16)
model.eval(); backbone.eval()
for p in model.parameters(): p.requires_grad_(False)
for p in latents.parameters(): p.requires_grad_(False)

label_dict = json.load(open(LABEL_DICT))
labels = list(label_dict.keys())
all_templates = [t for l in labels for t in label_dict[l]]
print(f"  {len(all_templates)} templates", flush=True)

embs = []
with torch.inference_mode():
    for i in range(0, len(all_templates), 16):
        chunk = all_templates[i:i+16]
        s_embeds, s_mask, s_pool = build_student_inputs(chunk, tok, model, latents, ids, max_task_len=64)
        out = backbone(inputs_embeds=s_embeds, attention_mask=s_mask, use_cache=False)
        h = pool_positions(out.last_hidden_state.float(), s_pool)
        embs.append(h)
embs = torch.cat(embs, dim=0)
mean = embs.mean(0)
print(f"  mean norm: {mean.norm().item():.2f}", flush=True)
torch.save(mean.cpu(), OUT_MEAN)
print(f"Saved {OUT_MEAN}", flush=True)
```

- [ ] **Step 2: Run and verify output**

```bash
cd /home/ds/quest_023_codi && python scripts/compute_mean.py
```

Expected: prints "mean norm: <number>" and saves `artifacts/mean.pt`.

- [ ] **Step 3: Commit**

```bash
cd /home/ds/quest_023_codi && git add . && \
git -c user.email=ds@local -c user.name=ds commit -q -m "script: compute global mean for centering"
```

---

### Task 10: Tiny-fixture smoke (50 steps, sanity)

**Files:**
- Create: `/home/ds/quest_023_codi/scripts/smoke_tiny.py`

- [ ] **Step 1: Implement**

```python
# scripts/smoke_tiny.py
"""50-step smoke run: verify no NaN, memory OK, loss curves sane."""
import os, sys, json, random, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.utils.data import DataLoader
from src.codi_model import load_codi_trainable
from src.codi_data import CODIDataset, make_codi_collator
from src.codi_label_cache import encode_label_templates_codi, CODILabelCache
from src.codi_losses import codi_step, CODILossConfig

DATA = "/tmp/synth_smoke/training_v4/synth_train.jsonl"
LABEL_DICT = "/tmp/synth_smoke/training_v4/label_dict.json"
MEAN_PT = "/home/ds/quest_023_codi/artifacts/mean.pt"
N_STEPS = 50

print("Loading model...", flush=True)
model, backbone, tok, latents, ids = load_codi_trainable(k=4, lora_r=16)
mean = torch.load(MEAN_PT, map_location="cuda", weights_only=True).float()

label_dict = json.load(open(LABEL_DICT))
labels = list(label_dict.keys())
ds = CODIDataset(DATA, label_list=labels)
# Subsample 200 records for tiny smoke
ds = CODIDataset(DATA, label_list=labels, sample_indices=list(range(200)))
loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=make_codi_collator())

opt_params = [p for p in model.parameters() if p.requires_grad] + list(latents.parameters())
optimizer = torch.optim.AdamW(opt_params, lr=2e-4, betas=(0.9, 0.999), weight_decay=0.0)
print(f"  trainable params: {sum(p.numel() for p in opt_params)}", flush=True)

embs_all = encode_label_templates_codi(backbone, tok, model, latents, ids, label_dict, mean.to("cuda"))
cache = CODILabelCache(label_dict, embs_all)
print(f"  label cache: {tuple(embs_all.shape)}", flush=True)

cfg = CODILossConfig(k=4, warmup_steps=10, ramp_steps=10)  # short warmup for tiny smoke
rng = random.Random(0)
model.train()

t0 = time.time()
step = 0
while step < N_STEPS:
    # Re-encode label cache periodically
    if step % 10 == 0 and step > 0:
        with torch.no_grad():
            embs_all = encode_label_templates_codi(backbone, tok, model, latents, ids, label_dict, mean.to("cuda"))
            cache = CODILabelCache(label_dict, embs_all)
    for batch in loader:
        if step >= N_STEPS: break
        label_embs = cache.sample_per_label(rng).to("cuda")
        loss, log = codi_step(model, backbone, latents, tok, ids, batch, label_embs,
                                 mean.to("cuda"), cfg, step)
        if not torch.isfinite(loss):
            print(f"NaN at step {step}: {log}"); sys.exit(1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(opt_params, 1.0)
        optimizer.step()
        if step % 5 == 0 or step == N_STEPS - 1:
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"  step {step:3d}: total={loss.item():.3f} L_cls_S={log['L_cls_S']:.3f} "
                  f"acc_S={log['acc_S']:.2f} acc_T={log['acc_T']:.2f} "
                  f"align_gap={log['align_gap']:.3f} mem={mem:.1f}GB", flush=True)
        step += 1
elapsed = time.time() - t0
print(f"\nDONE: {step} steps in {elapsed:.0f}s ({elapsed/step:.1f}s/step)", flush=True)
print(f"Peak memory: {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)
```

- [ ] **Step 2: Run**

```bash
cd /home/ds/quest_023_codi && python scripts/smoke_tiny.py 2>&1 | tee artifacts/smoke_tiny.log
```

Expected: 50 steps complete, no NaN, peak memory < 70 GB (single A100), loss decreasing over the last 20 steps.

- [ ] **Step 3: If memory > 70GB, reduce batch to 2 in this script and `scripts/train_codi.py`**

Edit the `DataLoader(..., batch_size=2, ...)` in both scripts.

- [ ] **Step 4: Commit**

```bash
cd /home/ds/quest_023_codi && git add . && \
git -c user.email=ds@local -c user.name=ds commit -q -m "script: tiny smoke 50 steps + memory check"
```

---

### Task 11: Full 2000-step training script

**Files:**
- Create: `/home/ds/quest_023_codi/scripts/train_codi.py`

- [ ] **Step 1: Implement**

```python
# scripts/train_codi.py
"""Smoke v5: 2000 steps on 11K multi-schema synth, save adapter + latents + tokenizer."""
import os, sys, json, random, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.utils.data import DataLoader
from src.codi_model import load_codi_trainable
from src.codi_data import CODIDataset, make_codi_collator
from src.codi_label_cache import encode_label_templates_codi, CODILabelCache
from src.codi_losses import codi_step, CODILossConfig

OUT_DIR = "/home/ds/quest_023_codi/artifacts/smoke_v5"
DATA = "/tmp/synth_smoke/training_v4/synth_train.jsonl"
LABEL_DICT = "/tmp/synth_smoke/training_v4/label_dict.json"
MEAN_PT = "/home/ds/quest_023_codi/artifacts/mean.pt"
N_STEPS = 2000
CACHE_REFRESH_EVERY = 50

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading model...", flush=True)
model, backbone, tok, latents, ids = load_codi_trainable(k=4, lora_r=16)
mean = torch.load(MEAN_PT, map_location="cuda", weights_only=True).float()
print(f"  mean norm: {mean.norm().item():.2f}", flush=True)

label_dict = json.load(open(LABEL_DICT))
labels = list(label_dict.keys())
ds = CODIDataset(DATA, label_list=labels)
print(f"  dataset: {len(ds)} samples, {len(labels)} labels", flush=True)
loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=make_codi_collator())

opt_params = [p for p in model.parameters() if p.requires_grad] + list(latents.parameters())
optimizer = torch.optim.AdamW(opt_params, lr=2e-4, betas=(0.9, 0.999), weight_decay=0.0)
n_trainable = sum(p.numel() for p in opt_params)
print(f"  trainable: {n_trainable/1e6:.1f}M params", flush=True)

cfg = CODILossConfig(k=4, tau=0.07, alpha=0.5, beta=1.0, gamma=0.5,
                      warmup_steps=200, ramp_steps=200)
rng = random.Random(123)
model.train()

losses, accs_S, accs_T = [], [], []
t0 = time.time()
step = 0
embs_all = encode_label_templates_codi(backbone, tok, model, latents, ids, label_dict, mean.to("cuda"))
cache = CODILabelCache(label_dict, embs_all)

while step < N_STEPS:
    for batch in loader:
        if step >= N_STEPS: break
        if step > 0 and step % CACHE_REFRESH_EVERY == 0:
            with torch.no_grad():
                embs_all = encode_label_templates_codi(backbone, tok, model, latents, ids, label_dict, mean.to("cuda"))
                cache = CODILabelCache(label_dict, embs_all)
        label_embs = cache.sample_per_label(rng).to("cuda")
        loss, log = codi_step(model, backbone, latents, tok, ids, batch, label_embs,
                                 mean.to("cuda"), cfg, step)
        if not torch.isfinite(loss):
            print(f"NaN at step {step}: {log}"); sys.exit(1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(opt_params, 1.0)
        optimizer.step()
        losses.append(loss.item()); accs_S.append(log["acc_S"]); accs_T.append(log["acc_T"])
        if step % 50 == 0 or step == N_STEPS - 1:
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"  step {step:4d}/{N_STEPS}: total={loss.item():.3f} "
                  f"L_cls_S={log['L_cls_S']:.3f} L_cls_T={log['L_cls_T']:.3f} "
                  f"L_align={log['L_align']:.3f} L_distill={log['L_distill']:.3f} "
                  f"acc_S={log['acc_S']:.2f} acc_T={log['acc_T']:.2f} "
                  f"gap={log['align_gap']:.3f} β={log['beta_eff']:.2f} γ={log['gamma_eff']:.2f} "
                  f"mem={mem:.1f}GB", flush=True)
        step += 1

print(f"\nDONE: {step} steps in {time.time()-t0:.0f}s", flush=True)
for i in range(0, len(losses), 200):
    cl = losses[i:i+200]; cas = accs_S[i:i+200]; cat = accs_T[i:i+200]
    print(f"  steps {i:4d}-{i+len(cl)-1:4d}: avg_loss={sum(cl)/len(cl):.3f} "
          f"avg_acc_S={sum(cas)/len(cas):.3f} avg_acc_T={sum(cat)/len(cat):.3f}", flush=True)

# Save adapter + latents + tokenizer
model.save_pretrained(f"{OUT_DIR}/lora_adapter")
torch.save(latents.state_dict(), f"{OUT_DIR}/latents.pt")
tok.save_pretrained(f"{OUT_DIR}/tokenizer")
with open(f"{OUT_DIR}/special_ids.json", "w") as f:
    json.dump(ids, f)
import shutil
shutil.copy(MEAN_PT, f"{OUT_DIR}/mean.pt")
print(f"\nSaved adapter+latents+tokenizer to {OUT_DIR}", flush=True)
```

- [ ] **Step 2: Run (long, ~8h)**

```bash
cd /home/ds/quest_023_codi && python scripts/train_codi.py 2>&1 | tee artifacts/smoke_v5/train.log
```

Expected: 2000 steps complete; loss curve shows L_cls_S decreasing from ~3.0 to <1.5, acc_S from ~0.10 to >0.6.

- [ ] **Step 3: Commit script (don't commit large artifacts)**

```bash
cd /home/ds/quest_023_codi && git add scripts/train_codi.py && \
git -c user.email=ds@local -c user.name=ds commit -q -m "script: full 2000-step smoke v5 training"
```

---

### Task 12: Full-RE eval script

**Files:**
- Create: `/home/ds/quest_023_codi/scripts/eval_codi.py`

- [ ] **Step 1: Implement**

```python
# scripts/eval_codi.py
"""Full-RE eval for trained CODI adapter. NO positive-only — every record (incl. no_relation) is scored.

Reports macro-F1 + RE-style micro-F1 (TP/(TP+FP+FN) over positive predictions, excluding no_relation TNs).
Also supports post-hoc τ_cal threshold for no_relation prediction.
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from src.codi_inputs import build_student_inputs, pool_positions
from src.codi_label_cache import encode_label_templates_codi, CODILabelCache
from src.codi_latents import LearnableLatents
from src.codi_data import fmt_task

QWEN = "/root/shared-nvme/ds-workspace/Qwen3-4B-Base"


def load_codi_eval(adapter_dir: str, tok_dir: str, latents_pt: str, special_ids_json: str,
                    k: int = 4, device: str = "cuda"):
    tok = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(QWEN, dtype=torch.bfloat16,
                                                  attn_implementation="flash_attention_2",
                                                  trust_remote_code=True)
    base.resize_token_embeddings(len(tok))
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.to(device); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    backbone = model.base_model.model.model
    d_model = base.config.hidden_size
    latents = LearnableLatents(k=k, d_model=d_model).to(device=device, dtype=torch.bfloat16)
    latents.load_state_dict(torch.load(latents_pt, map_location=device, weights_only=True))
    for p in latents.parameters(): p.requires_grad_(False)
    ids = json.load(open(special_ids_json))
    return model, backbone, tok, latents, ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--label_dict", required=True)
    p.add_argument("--adapter_dir", required=True)
    p.add_argument("--tok_dir", required=True)
    p.add_argument("--latents_pt", required=True)
    p.add_argument("--special_ids", required=True)
    p.add_argument("--mean_pt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--name", default="dataset")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--tau_cal", type=float, default=0.0,
                    help="Post-hoc no_relation threshold; predict no_relation if (max_pos_score - no_rel_score) < tau_cal")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("Loading model + adapter...", flush=True)
    model, backbone, tok, latents, ids = load_codi_eval(
        args.adapter_dir, args.tok_dir, args.latents_pt, args.special_ids, k=args.k
    )
    mean = torch.load(args.mean_pt, map_location="cuda", weights_only=True).float()

    label_dict = json.load(open(args.label_dict))
    labels = list(label_dict.keys())
    print(f"  {len(labels)} labels: {labels}", flush=True)
    has_no_rel = "no_relation" in labels
    if has_no_rel:
        no_rel_idx = labels.index("no_relation")

    # Encode labels via CODI template
    embs_all = encode_label_templates_codi(backbone, tok, model, latents, ids, label_dict, mean.to("cuda"))
    label_embs = embs_all.mean(dim=1)  # average templates → (n_labels, D)
    label_embs = torch.nn.functional.normalize(label_embs, p=2, dim=-1)

    # Load eval records — FULL RE protocol means every record is included regardless of label
    records = []
    with open(args.data) as f:
        for line in f:
            r = json.loads(line)
            if r["label"] not in labels:
                continue
            records.append(r)
    print(f"  {len(records)} records, label distribution: {Counter(r['label'] for r in records).most_common()}", flush=True)

    # Eval
    preds = []
    t0 = time.time()
    for i in range(0, len(records), args.batch_size):
        batch = records[i:i + args.batch_size]
        texts = [fmt_task(r) for r in batch]
        with torch.inference_mode():
            s_embeds, s_mask, s_pool = build_student_inputs(texts, tok, model, latents, ids, max_task_len=600)
            out = backbone(inputs_embeds=s_embeds, attention_mask=s_mask, use_cache=False)
            h = pool_positions(out.last_hidden_state.float(), s_pool)
            h_c = h - mean.to("cuda").unsqueeze(0)
            h_n = torch.nn.functional.normalize(h_c, p=2, dim=-1)
            scores = h_n @ label_embs.t()  # (B, n_labels)

            if has_no_rel and args.tau_cal > 0:
                # Post-hoc threshold: if max positive - no_rel < tau_cal → predict no_relation
                pos_mask = torch.ones(len(labels), device="cuda", dtype=torch.bool)
                pos_mask[no_rel_idx] = False
                pos_max = scores[:, pos_mask].max(dim=-1).values
                no_rel_score = scores[:, no_rel_idx]
                margin = pos_max - no_rel_score
                pred_idx = scores.argmax(dim=-1)
                pred_idx[margin < args.tau_cal] = no_rel_idx
            else:
                pred_idx = scores.argmax(dim=-1)

            scores_np = scores.cpu().numpy()
            pred_idx = pred_idx.cpu().numpy()
        for r, pi, s in zip(batch, pred_idx, scores_np):
            preds.append({**r, "pred": labels[pi], "scores": s.tolist()})
        if (i // args.batch_size) % 20 == 0:
            done = i + len(batch)
            print(f"  {done}/{len(records)} ({time.time()-t0:.0f}s)", flush=True)

    print(f"  done in {time.time()-t0:.0f}s", flush=True)

    # Metrics — FULL RE protocol (no positive-only filter)
    gold = [r["label"] for r in preds]
    pred = [r["pred"] for r in preds]

    per_class = {}
    f1s_macro = []
    for lbl in labels:
        g = sum(1 for x in gold if x == lbl)
        pp = sum(1 for x in pred if x == lbl)
        tp = sum(1 for r in preds if r["label"] == lbl and r["pred"] == lbl)
        prec = tp / pp if pp else 0.0
        rec = tp / g if g else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[lbl] = {"support": g, "predicted": pp, "tp": tp,
                          "precision": prec, "recall": rec, "f1": f1}
        if g > 0:
            f1s_macro.append(f1)
    macro_f1 = float(np.mean(f1s_macro))

    # RE-style micro-F1: exclude no_relation TNs; sum TP/FP/FN over positive labels
    if has_no_rel:
        pos_labels = [l for l in labels if l != "no_relation"]
    else:
        pos_labels = labels
    tp_sum = sum(per_class[l]["tp"] for l in pos_labels)
    fp_sum = sum(per_class[l]["predicted"] - per_class[l]["tp"] for l in pos_labels)
    fn_sum = sum(per_class[l]["support"] - per_class[l]["tp"] for l in pos_labels)
    if tp_sum + fp_sum + fn_sum > 0:
        micro_p = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) else 0.0
        micro_r = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) else 0.0
        re_micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    else:
        re_micro_f1 = 0.0

    accuracy_with_no_rel = sum(1 for r in preds if r["label"] == r["pred"]) / len(preds)

    metrics = {
        "name": args.name,
        "n_examples": len(preds),
        "n_labels": len(labels),
        "tau_cal": args.tau_cal,
        "full_re": {
            "macro_f1": macro_f1,
            "re_micro_f1": re_micro_f1,
            "accuracy_incl_no_rel": accuracy_with_no_rel,
            "per_class": per_class,
        },
    }
    with open(f"{args.out}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(f"{args.out}/predictions.jsonl", "w") as f:
        for r in preds:
            f.write(json.dumps(r) + "\n")

    print(f"\n=== Full-RE metrics ({args.name}, tau_cal={args.tau_cal}) ===")
    print(f"  macro-F1:    {macro_f1:.4f}")
    print(f"  RE micro-F1: {re_micro_f1:.4f}  (positive labels only)")
    print(f"  accuracy:    {accuracy_with_no_rel:.4f}  (includes no_relation TNs, NOT a sub for F1)")
    for lbl, m in per_class.items():
        print(f"    {lbl:<25} sup={m['support']:>4} pred={m['predicted']:>4} F1={m['f1']:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run on BioRED full-RE (raw, no τ_cal)**

```bash
cd /home/ds/quest_023_codi && python scripts/eval_codi.py \
  --data /home/ds/cross_eval/data/biored_full_test.jsonl \
  --label_dict /tmp/synth_smoke/training_v4/label_dict.json \
  --adapter_dir artifacts/smoke_v5/lora_adapter \
  --tok_dir artifacts/smoke_v5/tokenizer \
  --latents_pt artifacts/smoke_v5/latents.pt \
  --special_ids artifacts/smoke_v5/special_ids.json \
  --mean_pt artifacts/smoke_v5/mean.pt \
  --out artifacts/smoke_v5/eval_biored \
  --name biored_full --tau_cal 0.0
```

Expected: outputs metrics.json + predictions.jsonl; macro F1 printed.

- [ ] **Step 3: Sweep τ_cal on dev split for BioRED, pick best, re-eval**

```bash
# Split 10% dev (deterministic)
python -c "
import json, random
records = [json.loads(l) for l in open('/home/ds/cross_eval/data/biored_full_test.jsonl')]
random.Random(42).shuffle(records)
n_dev = len(records) // 10
with open('/home/ds/cross_eval/data/biored_dev.jsonl', 'w') as f:
    for r in records[:n_dev]: f.write(json.dumps(r) + '\n')
with open('/home/ds/cross_eval/data/biored_eval.jsonl', 'w') as f:
    for r in records[n_dev:]: f.write(json.dumps(r) + '\n')
print(f'dev={n_dev}, eval={len(records)-n_dev}')
"
# Sweep tau_cal in {0.05, 0.10, 0.15, 0.20, 0.30}
for tau in 0.05 0.10 0.15 0.20 0.30; do
  python scripts/eval_codi.py \
    --data /home/ds/cross_eval/data/biored_dev.jsonl \
    --label_dict /tmp/synth_smoke/training_v4/label_dict.json \
    --adapter_dir artifacts/smoke_v5/lora_adapter \
    --tok_dir artifacts/smoke_v5/tokenizer \
    --latents_pt artifacts/smoke_v5/latents.pt \
    --special_ids artifacts/smoke_v5/special_ids.json \
    --mean_pt artifacts/smoke_v5/mean.pt \
    --out artifacts/smoke_v5/eval_biored_dev_tau${tau} \
    --name biored_dev --tau_cal $tau
done
# Pick the τ_cal with highest macro_f1 on dev, then run eval split with that τ_cal
```

- [ ] **Step 4: Run BioTriplex + DrugProt full-RE evals**

```bash
# BioTriplex
python scripts/eval_codi.py \
  --data /home/ds/cross_eval/data/biotriplex_test.jsonl \
  --label_dict /home/ds/cross_eval/label_dicts/biotriplex.json \
  --adapter_dir artifacts/smoke_v5/lora_adapter \
  --tok_dir artifacts/smoke_v5/tokenizer \
  --latents_pt artifacts/smoke_v5/latents.pt \
  --special_ids artifacts/smoke_v5/special_ids.json \
  --mean_pt artifacts/smoke_v5/mean.pt \
  --out artifacts/smoke_v5/eval_biotriplex \
  --name biotriplex --tau_cal 0.0

# DrugProt (subsample first 5K for smoke)
head -5000 /home/ds/cross_eval/data/drugprot_full_test.jsonl > /tmp/drugprot_smoke5k.jsonl
python scripts/eval_codi.py \
  --data /tmp/drugprot_smoke5k.jsonl \
  --label_dict /tmp/synth_smoke/training_v4/label_dict.json \
  --adapter_dir artifacts/smoke_v5/lora_adapter \
  --tok_dir artifacts/smoke_v5/tokenizer \
  --latents_pt artifacts/smoke_v5/latents.pt \
  --special_ids artifacts/smoke_v5/special_ids.json \
  --mean_pt artifacts/smoke_v5/mean.pt \
  --out artifacts/smoke_v5/eval_drugprot \
  --name drugprot_full_5k --tau_cal 0.0
```

- [ ] **Step 5: Commit eval script**

```bash
cd /home/ds/quest_023_codi && git add scripts/eval_codi.py && \
git -c user.email=ds@local -c user.name=ds commit -q -m "script: full-RE eval (positive-only forbidden)"
```

---

### Task 13: Embedding crowding diagnostic on v5 adapter

**Files:**
- Create: `/home/ds/quest_023_codi/scripts/embedding_diag_v5.py`

- [ ] **Step 1: Implement**

```python
# scripts/embedding_diag_v5.py
"""Re-run embedding crowding diagnostic on v5 adapter judgment vectors.

Compares against v4's mean inter-label cosine (~0.40 training, ~0.94 worst pair).
Acceptance gate: opposite-direction pair cosine < 0.7.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from scripts.eval_codi import load_codi_eval
from src.codi_inputs import build_student_inputs, pool_positions
from src.codi_label_cache import encode_label_templates_codi

OUT_DIR = "/home/ds/quest_023_codi/artifacts/smoke_v5"
LABEL_DICT = "/tmp/synth_smoke/training_v4/label_dict.json"
MEAN_PT = f"{OUT_DIR}/mean.pt"

model, backbone, tok, latents, ids = load_codi_eval(
    f"{OUT_DIR}/lora_adapter", f"{OUT_DIR}/tokenizer",
    f"{OUT_DIR}/latents.pt", f"{OUT_DIR}/special_ids.json", k=4
)
mean = torch.load(MEAN_PT, map_location="cuda", weights_only=True).float()

label_dict = json.load(open(LABEL_DICT))
labels = list(label_dict.keys())
embs_all = encode_label_templates_codi(backbone, tok, model, latents, ids, label_dict, mean.to("cuda"))
label_embs = F.normalize(embs_all.mean(dim=1), p=2, dim=-1)

sim = label_embs @ label_embs.t()
n = sim.shape[0]
off = sim - torch.eye(n, device=sim.device)
mean_off = off.sum().item() / (n * (n - 1))
max_off = off.max().item()

print(f"Training labels (n={n}) inter-cosine:")
print(f"  mean: {mean_off:+.3f}  (v4 baseline: +0.401)")
print(f"  max:  {max_off:+.3f}  (v4 baseline: +0.936)")

# Top-5 closest pairs
pairs = []
for i in range(n):
    for j in range(i+1, n):
        pairs.append((sim[i, j].item(), labels[i], labels[j]))
pairs.sort(reverse=True)
print("Top-5 closest pairs:")
for s, a, b in pairs[:5]:
    print(f"  {s:+.3f}  {a} ↔ {b}")

# Opposite-direction check — specific gate
opposite = ["INDIRECT-UPREGULATOR", "INDIRECT-DOWNREGULATOR"]
if all(l in labels for l in opposite):
    i, j = labels.index(opposite[0]), labels.index(opposite[1])
    print(f"\nGATE: {opposite[0]} ↔ {opposite[1]} cosine = {sim[i,j].item():+.3f}")
    print(f"  v4 baseline: +0.936; v5 acceptance: < +0.7")
    print(f"  → {'PASS' if sim[i,j].item() < 0.7 else 'FAIL'}")
```

- [ ] **Step 2: Run**

```bash
cd /home/ds/quest_023_codi && python scripts/embedding_diag_v5.py | tee artifacts/smoke_v5/diag.log
```

- [ ] **Step 3: Commit**

```bash
cd /home/ds/quest_023_codi && git add scripts/embedding_diag_v5.py && \
git -c user.email=ds@local -c user.name=ds commit -q -m "script: v5 embedding crowding diagnostic"
```

---

### Task 14: Decision point — pivot success or fail

**Files:**
- Create: `/home/ds/quest_023_codi/artifacts/smoke_v5/DECISION.md`

- [ ] **Step 1: Gather all v5 numbers**

```bash
cd /home/ds/quest_023_codi && \
for f in artifacts/smoke_v5/eval_biored/metrics.json \
         artifacts/smoke_v5/eval_biotriplex/metrics.json \
         artifacts/smoke_v5/eval_drugprot/metrics.json; do
  echo "=== $f ==="
  python -c "import json; m=json.load(open('$f')); fr=m['full_re']; print(f'  macro_F1: {fr[\"macro_f1\"]:.4f}'); print(f'  RE_micro_F1: {fr[\"re_micro_f1\"]:.4f}')"
done
cat artifacts/smoke_v5/diag.log | grep -E "^GATE|mean:|max:"
```

- [ ] **Step 2: Write DECISION.md against acceptance gates**

Use the gates from the design spec (§5.2). Write findings to `artifacts/smoke_v5/DECISION.md`:

```markdown
# Quest 023 CODI Smoke v5 Decision

**Date**: <YYYY-MM-DD>

## Results vs acceptance gates

| Gate | Target | v5 result | Pass/Fail |
|---|---|---|---|
| BioRED full-RE macro F1 | ≥ 0.25 | <fill> | <P/F> |
| Opposite-direction cosine | < 0.70 | <fill> | <P/F> |
| BioTriplex macro F1 | ≥ 0.08 | <fill> | <P/F> |
| DrugProt macro F1 | ≥ 0.25 | <fill> | <P/F> |
| Training stability | no NaN, finite loss | <fill> | <P/F> |

## Comparison to v4 baseline

| Dataset | v4 (single-token) | v5 (CODI 4 latents) | Δ |
|---|---|---|---|
| BioRED macro | 0.201 | <fill> | <fill> |
| BioTriplex macro | 0.024 | <fill> | <fill> |
| Embedding mean inter-cos | +0.401 | <fill> | <fill> |
| INDIRECT-UPREG↔DOWNREG cos | +0.936 | <fill> | <fill> |

## Decision

- [ ] **PIVOT SUCCESS**: Both must-have gates pass. Scale to 30K synth, launch quest 023 full.
- [ ] **PARTIAL SUCCESS**: One must-have fails but trend positive. Sweep k=[2, 8], add bidi attention experiment.
- [ ] **PIVOT FAIL**: Both must-have fail. Move to Option D (generative pivot).

## Next action

<fill>
```

- [ ] **Step 3: Commit**

```bash
cd /home/ds/quest_023_codi && git add artifacts/smoke_v5/DECISION.md && \
git -c user.email=ds@local -c user.name=ds commit -q -m "doc: smoke v5 decision against gates"
```

---

## Self-Review

**Spec coverage:**
- Section 2.1 (input layout) → Tasks 2, 4, 5
- Section 2.2 (two-branch training) → Tasks 5, 7
- Section 2.3 (judgment pool) → Task 5 (`pool_positions`)
- Section 2.4 (label encoding) → Task 6
- Section 2.5 (4-loss) → Task 7
- Section 2.6 (warmup) → Task 7 (`_warmup_weights`)
- Section 2.7 (inference) → Task 12 (eval_codi.py uses student branch only)
- Section 2.8 (post-hoc τ_cal) → Task 12 (`--tau_cal` flag)
- Section 3 (hyperparameters) → Task 11 (`CODILossConfig` defaults match)
- Section 4.1 (file list) → Tasks 1-13 cover all listed files
- Section 5.1 (smoke run config) → Task 11
- Section 5.2 (acceptance gates incl. full-RE mandate) → Task 12 (forbids positive-only), Task 14 (gate check)
- Section 5.3 (diagnostics) → Task 13

**Placeholder scan:** Task 14 Step 2 has `<fill>` placeholders intentionally — those are runtime values the engineer fills after running steps. All code blocks are complete.

**Type consistency:** `load_codi_trainable` returns `(model, backbone, tok, latents, ids)` consistently across Tasks 4, 9, 10, 11. `codi_step` signature stable. `pool_positions` arg order `(hidden_states, pool_idx)` stable. `CODILossConfig` field names stable.

**Gaps:** None identified.
