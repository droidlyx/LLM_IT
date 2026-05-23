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
