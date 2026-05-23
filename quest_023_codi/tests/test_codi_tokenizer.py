import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.codi_tokenizer import add_codi_specials, init_special_embeddings

QWEN = "/root/shared-nvme/ds-workspace/Qwen3-4B-Base"

def test_add_specials_returns_ids():
    tok = AutoTokenizer.from_pretrained(QWEN, trust_remote_code=True)
    base_n = len(tok)
    vocab_before = set(tok.get_vocab().keys())
    think_id, answer_id = add_codi_specials(tok)
    n_new = sum(1 for t in ["<think>", "<answer>"] if t not in vocab_before)
    assert len(tok) == base_n + n_new
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
