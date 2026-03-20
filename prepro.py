from math import e
from tqdm import tqdm
import ujson as json
from transformers import AutoConfig, AutoModel, AutoTokenizer

def chunks(l, n):
    res = []
    for i in range(0, len(l), n):
        assert len(l[i:i + n]) == n
        res += [l[i:i + n]]
    return res


def read_docred(file_in, tokenizer, max_seq_length=1024, max_samples = None, use_direction = None):
    docred_rel2id = json.load(open('./dataset/docred/rel2id.json', 'r'))
    i_line = 0
    pos_samples = 0
    neg_samples = 0
    features = []
    if file_in == "":
        return None
    with open(file_in, "r") as fh:
        data = json.load(fh)

    if max_samples is not None:
        data = data[:max_samples]

    maxlen = 0
    for sample in tqdm(data, desc="Example"):
        sents = []
        sent_map = []

        entities = sample['vertexSet']
        entity_start, entity_end = [], []
        for entity in entities:
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]
                entity_start.append((sent_id, pos[0],))
                entity_end.append((sent_id, pos[1] - 1,))
        for i_s, sent in enumerate(sample['sents']):
            new_map = {}
            for i_t, token in enumerate(sent):
                tokens_wordpiece = tokenizer.tokenize(token)
                if (i_s, i_t) in entity_start:
                    tokens_wordpiece = ["*"] + tokens_wordpiece
                if (i_s, i_t) in entity_end:
                    tokens_wordpiece = tokens_wordpiece + ["*"]
                new_map[i_t] = len(sents)
                sents.extend(tokens_wordpiece)
            new_map[i_t + 1] = len(sents)
            sent_map.append(new_map)

        train_triple = {}
        if "labels" in sample:
            for label in sample['labels']:
                evidence = label['evidence']
                r = int(docred_rel2id[label['r']])
                if (label['h'], label['t']) not in train_triple:
                    train_triple[(label['h'], label['t'])] = [
                        {'relation': r, 'evidence': evidence}]
                else:
                    train_triple[(label['h'], label['t'])].append(
                        {'relation': r, 'evidence': evidence})

        entity_pos = []
        for e in entities:
            entity_pos.append([])
            for m in e:
                start = sent_map[m["sent_id"]][m["pos"][0]]
                end = sent_map[m["sent_id"]][m["pos"][1]]
                entity_pos[-1].append((start, end,))

        relations, hts = [], []
        for h, t in train_triple.keys():
            relation = [0] * len(docred_rel2id)
            for mention in train_triple[h, t]:
                relation[mention["relation"]] = 1
                evidence = mention["evidence"]
            relations.append(relation)
            hts.append([h, t])
            pos_samples += 1

        for h in range(len(entities)):
            for t in range(len(entities)):
                if h != t and [h, t] not in hts:
                    relation = [1] + [0] * (len(docred_rel2id) - 1)
                    relations.append(relation)
                    hts.append([h, t])
                    neg_samples += 1

        assert len(relations) == len(entities) * (len(entities) - 1)

        maxlen = max(maxlen, len(sents))
        sents = sents[:max_seq_length - 2]
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]

        i_line += 1
        feature = {'input_ids': input_ids,
                   'entity_pos': entity_pos,
                   'labels': relations,
                   'hts': hts,
                   'title': sample['title'],
                   }
            
        features.append(feature)

    print("# of documents {}.".format(i_line))
    print("# of positive examples {}.".format(pos_samples))
    print("# of negative examples {}.".format(neg_samples))
    print("Max document length: {}.".format(maxlen))
    return features

def read_biored(file_in, tokenizer, max_seq_length=1024, max_samples = None, use_direction = False):
    pmids = set()
    features = []
    maxlen = 0
    
    with open(file_in, 'r') as infile:
        lines = infile.readlines()
    
    # Group lines by PMID
    pmid_data = {}
    temp_rel_list = []
    for line in lines:
        line = line.rstrip()
        if not line:
            continue
        parts = line.split('\t')
        pmid = parts[0]
        if '|t|' in pmid:
            pmid = pmid.split('|t|')[0]
        if '|a|' in pmid:
            pmid = pmid.split('|a|')[0]
        
        if pmid not in pmid_data:
            pmid_data[pmid] = {'title': '', 'abstract': '', 'entities': [], 'relations': [], 'directions': []}
        
        # Parse title and abstract
        if '|t|' in line:
            pmid_data[pmid]['title'] = line.split('|t|')[1]
        elif '|a|' in line:
            pmid_data[pmid]['abstract'] = line.split('|a|')[1]
        # Parse entity annotations (has 6 parts: pmid, start, end, text, type, id)
        elif len(parts) == 6:
            sub_ids = parts[5].split(';')
            for sub_id in sub_ids:
                try:
                    pmid_data[pmid]['entities'].append({
                        'start': int(parts[1]),
                        'end': int(parts[2]),
                        'text': parts[3],
                        'type': parts[4],
                        'id': sub_id
                    })
                except ValueError:
                    continue
        # Parse relations (has 4 or 5 parts)
        elif len(parts) in [4,5]:
            if parts[1].strip() not in temp_rel_list:
                temp_rel_list.append(parts[1].strip())
            pmid_data[pmid]['relations'].append({
                'type': parts[1],
                'arg1': parts[2],
                'arg2': parts[3],
                'novelty': parts[4] if len(parts) == 5 else None
            })

    f = open(f'dataset/rel_list/biored_rel_list.txt', 'w', encoding='utf-8')
    for rel in temp_rel_list:
        f.write(f'{rel}\n')

    # Process each PMID
    key_list = list(pmid_data.keys())
    if max_samples is not None:
        key_list = key_list[:max_samples]
        
    for pmid in tqdm(key_list, desc="Processing"):
        pmids.add(pmid)        
        data = pmid_data[pmid]
        text = data['title'] + ' ' + data['abstract']
        
        # Build entity ID to character positions mapping
        entity_id_to_char_positions = {}
        for ent in data['entities']:
            ent_id = ent['id']
            if ent_id not in entity_id_to_char_positions:
                entity_id_to_char_positions[ent_id] = []
            entity_id_to_char_positions[ent_id].append((ent['start'], ent['end']))
        
        # Collect all entity boundaries
        entity_pos_set = set()
        for positions in entity_id_to_char_positions.values():
            for start, end in positions:
                entity_pos_set.add((start, end))
        
        # Build character to word token mapping (optimized)
        char_to_word_token = {}
        word_tokens = []
        word_idx = 0
        
        # Pre-define delimiters for faster lookup
        delimiters = {' ', '/', '-', ',', '.', ':', ';', '%'}
        
        i = 0
        while i < len(text):
            char = text[i]
            if char in delimiters:
                if char != ' ':
                    char_to_word_token[i] = word_idx
                    word_tokens.append(char)
                    word_idx += 1
                i += 1
                continue
            
            # Start of a word
            word_start = i
            while i < len(text) and text[i] not in delimiters:
                char_to_word_token[i] = word_idx
                i += 1
            word_tokens.append(text[word_start:i])
            word_idx += 1
        
        # Tokenize with entity markers (optimized)
        new_sents = []
        sent_map = {}
        
        # Create dictionaries for O(1) entity position lookup
        entity_start_positions = {start for start, _ in entity_pos_set}
        entity_end_positions = {end for _, end in entity_pos_set}
        
        # Pre-compute character to word token reverse mapping for faster lookup
        word_token_to_char_range = {}
        for char_idx, word_idx in char_to_word_token.items():
            if word_idx not in word_token_to_char_range:
                word_token_to_char_range[word_idx] = [char_idx, char_idx + 1]
            else:
                word_token_to_char_range[word_idx][1] = char_idx + 1
        
        for i_t, word in enumerate(word_tokens):
            tokens_wordpiece = tokenizer.tokenize(word)
            
            # Get character range for this word token
            char_range = word_token_to_char_range.get(i_t)
            if char_range:
                word_char_start, word_char_end = char_range
                
                # Check if any entity starts or ends at this position (O(1) lookup)
                if word_char_start in entity_start_positions:
                    tokens_wordpiece = ["*"] + tokens_wordpiece
                if word_char_end in entity_end_positions:
                    tokens_wordpiece = tokens_wordpiece + ["*"]
            
            sent_map[i_t] = len(new_sents)
            new_sents.extend(tokens_wordpiece)
        
        sent_map[len(word_tokens)] = len(new_sents)
        sents = new_sents
        
        # Build entity positions in wordpiece token space
        ent2idx = {}
        entity_pos = []
        
        for ent_id in entity_id_to_char_positions.keys():
            if ent_id not in ent2idx:
                ent2idx[ent_id] = len(ent2idx)
                positions = []
                
                for start_char, end_char in entity_id_to_char_positions[ent_id]:
                    # Find word token indices for character positions
                    start_word_idx = char_to_word_token.get(start_char)
                    # Find the last character of the entity
                    end_word_idx = char_to_word_token.get(end_char - 1)
                    
                    if start_word_idx is not None and end_word_idx is not None:
                        # Convert to wordpiece token positions
                        start_wp_idx = sent_map.get(start_word_idx)
                        end_wp_idx = sent_map.get(end_word_idx + 1)
                        
                        if start_wp_idx is not None and end_wp_idx is not None:
                            positions.append((start_wp_idx, end_wp_idx))
                
                entity_pos.append(positions)
        
        # Process relations (optimized)
        # Create relation type to ID mapping for O(1) lookup (relations start from index 1)
        rel_type_to_id = {rel.strip().lower(): idx + 1 for idx, rel in enumerate(temp_rel_list)}
        
        train_triples = {(h_id, t_id):[] for h_id in ent2idx.values() for t_id in ent2idx.values()}
        for rel in data['relations']:
            rel_type = rel['type']
            arg1 = rel['arg1']
            arg2 = rel['arg2']
            if arg1 in ent2idx and arg2 in ent2idx:
                h_id = ent2idx[arg1]
                t_id = ent2idx[arg2]
                rel_id = rel_type_to_id.get(rel_type.strip().lower(), 0)
                if rel_id > 0:
                    train_triples[(h_id, t_id)].append({'relation': rel_id})
                    if not use_direction:
                        train_triples[(t_id, h_id)].append({'relation': rel_id})
                else:
                    raise Exception('Relation not found!')

        # Build relations and hts
        relations, hts = [], []
        pos_hts = []
        
        # Process positive relations
        for h, t in train_triples.keys():
            if train_triples[h, t]:  # Only if there are positive relations
                relation = [0] * (len(temp_rel_list) + 1)  # +1 for "no relation" class
                for mention in train_triples[h, t]:
                    relation[mention["relation"]] = 1
                relations.append(relation)
                hts.append([h, t])
                pos_hts.append([h, t])
        
        # Generate negative samples (entity pairs with no relations)
        for h in range(len(ent2idx)):
            for t in range(len(ent2idx)):
                if h != t and [h, t] not in pos_hts:
                    relation = [1] + [0] * len(temp_rel_list)  # "no relation" = 1 at index 0
                    relations.append(relation)
                    hts.append([h, t])
        
        maxlen = max(maxlen, len(sents))
        if len(sents) > max_seq_length - 2:
            print("\nDocument {} is too long. Truncating.".format(pmid))
        sents = sents[:max_seq_length - 2]
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]

        feature = {
            'input_ids': input_ids,
            'entity_pos': entity_pos,
            'labels': relations,
            'hts': hts,
            'title': pmid,
            'rel_list': temp_rel_list,
            'dataset_name': file_in.split('/')[-1].split('\\')[-1].split('.')[0]
        }
        features.append(feature)

    print("Number of documents: {}.".format(len(features)))
    print("Max document length: {}.".format(maxlen))
    return features

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
        "./base_models/bert-base-cased"
    )
    features = read_biored("./dataset/biored/bioredirect_train_dev.pubtator", tokenizer)
    # features = read_docred("./dataset/docred/train_annotated.json", tokenizer)
    pass