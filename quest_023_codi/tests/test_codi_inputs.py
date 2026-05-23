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
