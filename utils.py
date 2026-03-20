import torch
import random
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from openai import OpenAI

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    entity_pos = [f["entity_pos"] for f in batch]
    hts = [f["hts"] for f in batch]
    rel_list = [f['rel_list'] for f in batch]
    dataset_name = [f['dataset_name'] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    output = (input_ids, input_mask, labels, entity_pos, hts, rel_list, dataset_name)
    return output

def text2data(args, original_batch, text_input):
    # based on <number|entity name> format, construct new data
    clean_text = ''
    entities = []
    i = 0
    while i < len(text_input):
        if text_input[i] == '{':
            j = i
            while(j < len(text_input) and text_input[j] != '}'):
                j += 1
            if j == len(text_input):
                return None
            temp_str = text_input[i+1:j]
            if '|' not in temp_str:
                return None
            entity_id = temp_str.split('|')[0]
            entity_name = temp_str.split('|')[1]
            try:
                entity_id = int(entity_id)
            except:
                return None
            entities.append({
                'start': len(clean_text),
                'end': len(clean_text)+len(entity_name),
                'id': entity_id
            })
            clean_text += entity_name
            i = j + 1
        else:
            clean_text += text_input[i]
            i += 1    

    text = clean_text
    # Build entity ID to character positions mapping
    entity_id_to_char_positions = {}
    for ent in entities:
        ent_id = ent['id']
        if ent_id not in entity_id_to_char_positions:
            entity_id_to_char_positions[ent_id] = []
        entity_id_to_char_positions[ent_id].append((ent['start'], ent['end']))
    
    # Collect all entity boundaries
    entity_pos_set = set()
    for positions in entity_id_to_char_positions.values():
        for start, end in positions:
            entity_pos_set.add((start, end))
    
    # Build character to word token mapping
    char_to_word_token = {}
    word_tokens = []
    i = 0
    word_idx = 0
    while i < len(text):
        if text[i] in [' ', '/', '-', ',', '.', ':', ';', '%']:
            if text[i] in ['/', '-', ',', '.', ':', ';', '%']:
                char_to_word_token[i] = word_idx
                word_tokens.append(text[i])
                word_idx += 1
            i += 1
            continue
        # Start of a word
        word_start = i
        while i < len(text) and text[i] not in [' ', '/', '-', ',', '.', ':', ';', '%']:
            char_to_word_token[i] = word_idx
            i += 1
        word_tokens.append(text[word_start:i])
        word_idx += 1
    
    # Tokenize with entity markers
    new_sents = []
    sent_map = {}
    
    for i_t, word in enumerate(word_tokens):
        tokens_wordpiece = args.prepro_tokenizer.tokenize(word)
        
        # Find character range for this word token
        word_char_start = None
        word_char_end = None
        for char_idx, tok_idx in char_to_word_token.items():
            if tok_idx == i_t:
                if word_char_start is None:
                    word_char_start = char_idx
                word_char_end = char_idx + 1
        
        # Check if any entity starts at this word
        for start, end in entity_pos_set:
            if word_char_start is not None and start == word_char_start:
                tokens_wordpiece = ["*"] + tokens_wordpiece
            if word_char_end is not None and end == word_char_end:
                tokens_wordpiece = tokens_wordpiece + ["*"]
        
        sent_map[i_t] = len(new_sents)
        new_sents.extend(tokens_wordpiece)
    
    sent_map[len(word_tokens)] = len(new_sents)
    sents = new_sents
    
    # Build entity positions in wordpiece token space
    entity_pos = [[] for i in range(len(original_batch[3][0]))]
    
    for ent_id in entity_id_to_char_positions.keys():
        if ent_id >= 0 and ent_id < len(entity_pos):
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
            
            entity_pos[ent_id] = positions

    # Add mission entites to keep the format correct
    for i in range(len(entity_pos)):
        if len(entity_pos[i]) == 0:
            entity_pos[i] = [(0, 1)]

    if len(sents) > args.max_seq_length - 2:
        print("\nDocument is too long. Truncating.")
    sents = sents[:args.max_seq_length - 2]
    input_ids = args.prepro_tokenizer.convert_tokens_to_ids(sents)
    input_ids = args.prepro_tokenizer.build_inputs_with_special_tokens(input_ids)

    feature = {
        'input_ids': input_ids,
        'entity_pos': entity_pos,
        'labels': original_batch[2][0],
        'hts': original_batch[4][0],
    }
    dataloader = DataLoader([feature], batch_size=1, shuffle=False, collate_fn=collate_fn, drop_last=False)

    out_feature = None
    for batch in dataloader:
        out_feature = {'input_ids': batch[0].to(args.device),
                    'attention_mask': batch[1].to(args.device),
                    'entity_pos': batch[3],
                    'hts': batch[4],
                    }

    return out_feature

def remove_spaces(input_text):
    tokens = input_text.split(' ')
    output_text = ''
    remove_space = 1
    inside = 0
    last_token = ''
    special_tokens1 = [',', '.', ')', ']', '}', ':', ';', '%', '?', '!']
    special_tokens2 = ['(', '-', '>', '<', '/', '+', '*', '\'', '=', '\n']
    special_tokens2_2 = ['[', '{']
    special_tokens3 = ['\"']
    for token in tokens:
        if token in special_tokens1:
            processed_token = token
            remove_space = 0
        elif token in special_tokens2:
            processed_token = token
            remove_space = 1
        elif token in special_tokens2_2:
            processed_token = token if remove_space else ' ' + token
            remove_space = 1
        elif token in special_tokens3:
            if last_token.isdigit():
                processed_token = token if remove_space else ' ' + token
                remove_space = 0
            elif inside == 0:
                processed_token = token if remove_space else ' ' + token
                remove_space = 1
                inside = 20
            else:
                processed_token = token
                inside = 0
        else:
            processed_token = token if remove_space or (token.isdigit() and last_token in ['.', ',',':',')']) else ' ' + token
            remove_space = 0

        if inside > 0:
            inside -= 1
        last_token = token
        output_text += processed_token

    return output_text

def feature2text(args, input_ids, entity_pos, aug_rate = 0):
    # Decode input_ids to tokens
    tokens = args.prepro_tokenizer.convert_ids_to_tokens(input_ids)
    
    # Remove special tokens [CLS] and [SEP]
    tokens = tokens[1:-1]  # Remove [CLS] at start and [SEP] at end
    
    # Build entity index to token positions mapping
    entity_token_positions = {}
    for entity_idx, positions in enumerate(entity_pos):
        adjusted_positions = [(start + 1, end - 1) for start, end in positions]
        entity_token_positions[entity_idx] = adjusted_positions
    
    # Extract entity names from tokens
    entity_names = []
    for entity_idx, positions in entity_token_positions.items():
        if positions:
            # Try to find a mention that doesn't appear in other entities' mention lists
            selected_position = None
            
            # Collect all mentions from other entities
            other_entity_mentions = set()
            for other_idx, other_positions in entity_token_positions.items():
                if other_idx != entity_idx:
                    for start, end in other_positions:
                        other_entity_mentions.add((start, end))
            
            # First try to find a mention not in other entities' mentions
            for pos in positions:
                if pos not in other_entity_mentions:
                    selected_position = pos
                    break
            
            # If no unique mention found, use the first one
            if selected_position is None:
                selected_position = positions[0]
            
            start, end = selected_position
            entity_tokens = tokens[start:end]
            # Join tokens, handling subword tokens (starting with ##)
            entity_text = []
            for token in entity_tokens:
                if token.startswith('##'):
                    entity_text.append(token[2:])
                else:
                    if entity_text:
                        entity_text.append(' ')
                    entity_text.append(token)
            if random.random() < aug_rate:
                entity_names.append(f'Entity{entity_idx}')
            else:
                entity_names.append(''.join(entity_text).strip())
    entity_names = {i:entity_names[i] for i in range(len(entity_names))}
    
    # Build labeled text with entity markers
    token_info = []
    for token_idx, token in enumerate(tokens):
        entity_starts = []
        entity_ends = []
        
        for entity_idx, positions in entity_token_positions.items():
            for start, end in positions:
                if token_idx == start:
                    entity_starts.append(entity_idx)
                if token_idx == end - 1:
                    entity_ends.append(entity_idx)
        
        token_info.append({
            'token': token,
            'entity_starts': entity_starts,
            'entity_ends': entity_ends
        })
    
    # Build labeled text
    temp_str = None
    labeled_text = ''
    for info in token_info:
        # Add entity start markers
        for entity_idx in info['entity_starts']:
            labeled_text += '{' + f'{entity_idx}|'
            temp_str = ""
        
        # Add token (handle subword tokens)
        token = info['token']
        prefix = ' '
        if temp_str == '':
            prefix = ''
        if token.startswith('##'):
            token = token[2:]
            prefix = ''
        if temp_str is not None:
            temp_str += prefix + token
        else:
            labeled_text += prefix + token
        
        # Add entity end markers
        for entity_idx in info['entity_ends']:
            if temp_str is not None:
                ent_name = remove_spaces(temp_str)
                if random.random() < aug_rate:
                    ent_name = f'Entity{entity_idx}'
                labeled_text += ent_name + "}"
                temp_str = None

    labeled_text = labeled_text.replace('*{', '{')
    labeled_text = labeled_text.replace('} *', '}')
    
    # Remove unnecessary spaces
    labeled_text = remove_spaces(labeled_text)
    return labeled_text, entity_names

def load_soft_prompt(checkpoint_path):
    """Load soft prompt from checkpoint"""
    soft_prompt_path = os.path.join(checkpoint_path, 'soft_prompt.pt')
    if os.path.exists(soft_prompt_path):
        checkpoint = torch.load(soft_prompt_path, map_location='cpu')
        return checkpoint['soft_prompt'], checkpoint['num_soft_tokens']
    else:
        raise FileNotFoundError(f"Soft prompt not found at {soft_prompt_path}")

def api_generate(prompt, run_id = 0, temperature = 0.0, stream_output = True, history=None):
    # using deepseek API
    client = OpenAI(api_key="sk-7d7062314650e4b50cf065b6291a91ff", base_url="https://apis.iflow.cn/v1")
    # Build messages: start with optional history, then system+user from current prompt
    messages = []
    if history:
        messages.extend(history)
    messages.append({"role": "system", "content": prompt['instruction']})
    messages.append({"role": "user", "content": prompt['input']})

    for _ in range(10):
        try:
            response = client.chat.completions.create(
                model='deepseek-v3.2',
                messages=messages,
                temperature=temperature,
                stream=stream_output
            )
            break
        except Exception as e:
            print(e)
            time.sleep(5)

    if not stream_output:
        reasoning_content = None
        if '</think>' in response.choices[0].message.content:
            reasoning_content = response.choices[0].message.content.split('</think>')[0].split('<think>')[1]
            response.choices[0].message.content = response.choices[0].message.content.split('</think>')[1]
        if hasattr(response.choices[0].message, 'reasoning_content'):
            reasoning_content = response.choices[0].message.reasoning_content
        return response.choices[0].message.content, reasoning_content

    content_parts = []
    reasoning_parts = []
    for event in response:
        if not hasattr(event, 'choices') or not event.choices:
            continue
        delta = getattr(event.choices[0], 'delta', None)
        if delta is None:
            continue

        token = getattr(delta, 'content', None)
        if token:
            print(token, end='', flush=True)
            content_parts.append(token)

        token_reasoning = getattr(delta, 'reasoning_content', None)
        if token_reasoning:
            print(token_reasoning, end='', flush=True)
            reasoning_parts.append(token_reasoning)

    print('', flush=True)
    content = ''.join(content_parts)
    reasoning_content = ''.join(reasoning_parts) if reasoning_parts else None
    if '</think>' in content:
        reasoning_content = content.split('</think>')[0].split('<think>')[1]
        content = content.split('</think>')[1]

    # full_content = content if reasoning_content is None else reasoning_content + content
    # messages.append({"role": "assistant", "content": full_content})
    return content

def pair_generate(args, original_doc, questions, answers, rel_list_str):
    f = open(f'./meta/baseline/augment.txt', 'r', encoding='utf-8')
    generate_prompt = f.read()
    f.close()

    llm_input = generate_prompt.replace('[Input Text]', original_doc)
    llm_input = llm_input.replace('[Relation List]', rel_list_str)
    llm_input = llm_input.replace('[Questions]', questions)
    llm_input = llm_input.replace('[Answers]', answers)
    query = {'instruction': '', 'input': llm_input, 'output': ''}
    summary = api_generate(query, temperature = 1.0, stream_output = True)
    return summary


def construct_llm_input(args, feature, labels = None, generate_data = False, previous_outputs = None, aug_rate = 0, shuffle = False):
    original_doc, entity_names = feature2text(args, feature['input_ids'][0], feature['entity_pos'][0], aug_rate = aug_rate)
    queries = []
    rel_dict = {}
    rel_list = feature['rel_list'][0]
    rel_list_str = '\n'.join(rel_list)
    extract_prompt = args.extract_prompt
    max_input_len = args.max_input_len
    use_direction = args.use_direction

    if shuffle:
        sents = original_doc.split('.')
        random.shuffle(sents)
        original_doc = '.'.join(sents)

    # Pre-build relation dictionary more efficiently
    if labels is not None:
        for local_idx, ht in enumerate(feature['hts'][0]):
            h_idx, t_idx = ht[0], ht[1]
            label_rels = np.nonzero(labels[local_idx])[0]
            for rel_id in label_rels:
                if rel_id != 0:
                    rel_dict.setdefault((h_idx, t_idx), []).append(rel_id - 1)

    # Pre-compute entity strings to avoid repeated formatting
    entity_strings = [f'{{{i}|{entity_names[i]}}}' for i in range(len(entity_names))]
    
    # Pre-compute phrase to avoid conditional in loop
    phrase_template = 'from {} to' if use_direction else 'between {} and'

    for i in range(len(entity_names)):
        output_str_parts = []
        ent1 = entity_strings[i]
        phrase = phrase_template.format(ent1)
        questions_parts = [f'What is the relation {phrase} the following entities?\n']
        
        for j in range(len(entity_names)):
            if i != j:
                ent2 = entity_strings[j]
                rel_pair = rel_dict.get((i, j), [])
                
                if rel_pair:
                    output_type = rel_list[rel_pair[0]]
                    if len(rel_pair) > 1:
                        output_type += ',' + ','.join(rel_list[rel] for rel in rel_pair[1:])
                else:
                    output_type = 'None'
                
                output_str_parts.append(f'{len(output_str_parts) + 1}. ' + output_type + '\n')
                questions_parts.append(f'{len(output_str_parts)}.{ent2}\n')

        output_str = ''.join(output_str_parts)
        questions = ''.join(questions_parts)

        # Use string formatting with pre-computed template
        llm_input = extract_prompt.replace('[Input Text]', original_doc)
        llm_input = llm_input.replace('[Relation List]', rel_list_str)
        llm_input = llm_input.replace('[Questions]', questions)
        
        input_len = len(llm_input)
        if input_len > max_input_len:
            max_input_len = input_len
            args.max_input_len = input_len
            
        if previous_outputs is not None and labels is not None:
            output_str = previous_outputs[i]
        query = {'instruction': '', 'input': llm_input, 'output': output_str if labels is not None else ''}
        queries.append(query)

        if generate_data:
            output = pair_generate(args, original_doc, questions, output_str, rel_list_str)
            llm_input = extract_prompt.replace('[Input Text]', output)
            llm_input = llm_input.replace('[Relation List]', rel_list_str)
            llm_input = llm_input.replace('[Questions]', questions)
            
            input_len = len(llm_input)
            if input_len > max_input_len:
                max_input_len = input_len
                args.max_input_len = input_len
                
            query = {'instruction': '', 'input': llm_input, 'output': output_str if labels is not None else ''}
            queries.append(query)
        
    return queries

class SoftPromptWrapper(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module, args, tokenizer):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.num_soft_tokens = int(args.num_soft_tokens)

        emb = self.base_model.get_input_embeddings()
        if emb is None:
            raise ValueError("Base model does not expose input embeddings via get_input_embeddings().")
        hidden_size = int(getattr(emb, "embedding_dim", emb.weight.shape[1]))

        self.soft_prompt = torch.nn.Parameter(torch.empty(self.num_soft_tokens, hidden_size))
        torch.nn.init.normal_(self.soft_prompt, mean=0.0, std=0.02)

    def __getattr__(self, name):
        if name not in {"base_model", "num_soft_tokens", "soft_prompt"}:
            base = self.base_model
            if base is not None and hasattr(base, name):
                return getattr(base, name)
        return super().__getattr__(name)

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def find_instructions_position(self, input_ids, tokenizer):
        """Find the position of the soft prompt marker token"""
        marker_token_id = tokenizer.convert_tokens_to_ids("<|reserved_special_token_0|>")
        marker_positions = (input_ids[0] == marker_token_id).nonzero(as_tuple=True)[0]
        if len(marker_positions) == 0:
            raise Exception("Soft prompt marker token <|reserved_special_token_0|> not found in input_ids.")
        insert_pos = marker_positions[0].item()
        return insert_pos

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if input_ids is None:
            raise ValueError("SoftPromptWrapper requires input_ids.")

        inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
        bsz = inputs_embeds.shape[0]

        # Find where to insert soft tokens
        insert_pos = self.find_instructions_position(input_ids, self.tokenizer)
        
        # Insert soft tokens at the found position
        soft = self.soft_prompt.unsqueeze(0).expand(bsz, -1, -1).to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        inputs_embeds = torch.cat([
            inputs_embeds[:, :insert_pos, :], 
            soft, 
            inputs_embeds[:, insert_pos:, :]
        ], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        
        # Insert attention mask for soft tokens
        soft_attn = torch.ones((bsz, self.num_soft_tokens), dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([
            attention_mask[:, :insert_pos], 
            soft_attn, 
            attention_mask[:, insert_pos:]
        ], dim=1)

        if labels is not None:
            # Insert labels for soft tokens (masked with -100)
            soft_labels = torch.full((bsz, self.num_soft_tokens), -100, dtype=labels.dtype, device=labels.device)
            labels = torch.cat([
                labels[:, :insert_pos], 
                soft_labels, 
                labels[:, insert_pos:]
            ], dim=1)

        return self.base_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, **kwargs)

    def save_pretrained(self, save_directory, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        # if hasattr(self.base_model, 'save_pretrained'):
        #     self.base_model.save_pretrained(save_directory, **kwargs)
        print('Saving...')
        torch.save(
            {
                'num_soft_tokens': int(self.num_soft_tokens),
                'soft_prompt': self.soft_prompt.detach().cpu(),
            },
            os.path.join(save_directory, 'soft_prompt.pt'),
        )

