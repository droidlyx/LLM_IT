"""Compare v3's multi-pair prompt to test_llm.py's construct_llm_input prompt
for the same doc/source-entity. Find any literal difference."""
import sys, argparse, difflib
sys.path.insert(0, '.'); sys.path.insert(0, 'posthoc')

from prepro import read_biored
from transformers import AutoTokenizer
from utils import construct_llm_input
from score_pairs_v3 import build_multipair_prompt

prepro_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
features = read_biored("./dataset/biored/processed_test.pubtator",
                       prepro_tok, max_seq_length=1792,
                       max_samples=None, use_direction=False)
feature = features[0]
print(f"Doc 0: {len(feature['entity_pos'])} entities")

# v3 path
ns_v3 = argparse.Namespace(
    max_input_len=0,
    extract_prompt=open("./meta/baseline/extract.txt").read(),
    use_direction=False,
    prepro_tokenizer=prepro_tok,
)
q_v3 = build_multipair_prompt(ns_v3, feature)
v3_prompt0 = q_v3[0]['prompt']

# test_llm.py path
class NS:
    use_direction = False
    extract_prompt = open("./meta/baseline/extract.txt").read()
    max_input_len = 0
    prepro_tokenizer = prepro_tok
    label_weights = None
ns_test = NS()
# construct_llm_input expects feature wrapped in lists (collate fashion)
feat_wrapped = {
    'input_ids': feature['input_ids'].unsqueeze(0) if hasattr(feature['input_ids'],'unsqueeze') else [feature['input_ids']],
    'hts': [feature['hts']],
    'entity_pos': [feature['entity_pos']],
    'entity_types': [feature['entity_types']],
    'rel_list': [feature['rel_list']],
    'dataset_name': [feature.get('dataset_name', 'BioRED')],
}
q_test = construct_llm_input(ns_test, feat_wrapped, labels=None, generate_data=False,
                              previous_outputs=None, aug_rate=0, shuffle=False)
test_prompt0 = q_test[0]['input']

print(f"\nv3 prompt length: {len(v3_prompt0)} chars")
print(f"test_llm prompt length: {len(test_prompt0)} chars")
print(f"identical: {v3_prompt0 == test_prompt0}")

if v3_prompt0 != test_prompt0:
    print("\n--- diff (first 60 lines) ---")
    diff = list(difflib.unified_diff(
        test_prompt0.splitlines(keepends=True),
        v3_prompt0.splitlines(keepends=True),
        fromfile='test_llm', tofile='v3', lineterm=''
    ))
    for ln in diff[:60]:
        print(ln, end='' if ln.endswith('\n') else '\n')
