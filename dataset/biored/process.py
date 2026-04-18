def read_biored(file_in):
    pmids = set()
    
    with open(file_in, 'r') as infile:
        lines = infile.readlines()
    
    # Group lines by PMID
    pmid_data = {}
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
        elif len(parts) == 5:
            if (parts[2] > parts[3]):
                pmid_data[pmid]['relations'].append({
                    'type': parts[1],
                    'arg1': parts[3],
                    'arg2': parts[2],
                    'novelty': parts[4]
                })
            else:
                pmid_data[pmid]['relations'].append({
                    'type': parts[1],
                    'arg1': parts[2],
                    'arg2': parts[3],
                    'novelty': parts[4]
            })
        elif len(parts) == 4:
            if (parts[1] > parts[2]):
                pmid_data[pmid]['directions'].append({
                    'arg1': parts[2],
                    'arg2': parts[1],
                    'subject': parts[3].split(':')[-1]
                })
            else:
                pmid_data[pmid]['directions'].append({
                    'arg1': parts[1],
                    'arg2': parts[2],
                    'subject': parts[3].split(':')[-1]
                })

    for pmid in list(pmid_data.keys()):
        dir_dict = {}
        for dir in pmid_data[pmid]['directions']:
            dir_dict[(dir['arg1'], dir['arg2'])] = dir['subject']
        for i in range(len(pmid_data[pmid]['relations'])):
            arg1 = pmid_data[pmid]['relations'][i]['arg1']
            arg2 = pmid_data[pmid]['relations'][i]['arg2']
            if (arg1, arg2) in dir_dict:
                subject = dir_dict[(arg1, arg2)]
                if subject == arg2:
                    pmid_data[pmid]['relations'][i]['arg1'] = arg2
                    pmid_data[pmid]['relations'][i]['arg2'] = arg1

    return pmid_data

def write_biored(data, file_out):
    with open(file_out, 'w') as outfile:
        for pmid, content in data.items():
            # Write title
            if content['title']:
                outfile.write(f"{pmid}|t|{content['title']}\n")
            
            # Write abstract
            if content['abstract']:
                outfile.write(f"{pmid}|a|{content['abstract']}\n")
            
            # Write entities
            for entity in content['entities']:
                outfile.write(f"{pmid}\t{entity['start']}\t{entity['end']}\t{entity['text']}\t{entity['type']}\t{entity['id']}\n")
            
            # Write relations
            for relation in content['relations']:
                outfile.write(f"{pmid}\t{relation['type']}\t{relation['arg1']}\t{relation['arg2']}\t{relation['novelty']}\n")
            outfile.write(f"\n")

if __name__ == "__main__":
    data = read_biored("./bioredirect_bc8_test.pubtator")
    write_biored(data, "./biored_bc8_test.pubtator")
    print(f"Processed {len(data)} PMIDs and wrote to output")