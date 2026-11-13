"""Structural/source/leakage checks; does not establish independent label truth."""
import argparse
import hashlib
import html
import json
import re
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'datasets/eval-contract-challenge-v0.1'


def read(path):return [json.loads(x) for x in path.read_text().splitlines()]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def canon(ops):return json.dumps([{'kind':o['kind'],'text':o['text']} for o in ops],sort_keys=True,ensure_ascii=False)
def text(ops,accepted):return ''.join(o['text'] for o in ops if o['kind'] in ('equal','insert' if accepted else 'delete'))
def shingles(s):
    w=re.findall(r'\w+',s.lower());return {tuple(w[i:i+5]) for i in range(len(w)-4)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--tokenizer',type=Path);args=p.parse_args()
    inputs=read(DATA/'inputs.jsonl');labels=read(DATA/'annotations.jsonl');pairs=read(DATA/'pairs.jsonl')
    assert len(inputs)==len(labels)==96 and len(pairs)==32
    by_id={r['id']:r for r in inputs}; by_label={r['id']:r for r in labels}
    assert len(by_id)==len(by_label)==96 and set(by_id)==set(by_label)
    source_files=json.loads((DATA/'sources/manifest.json').read_text())
    for s in source_files:
        assert sha(ROOT/s['local_path'])==s['sha256']
    for name in ('Mutual-NDA','CSA','DPA','PSA'):
        assert 'CC BY 4.0' in (DATA/'sources'/f'{name}-README.md').read_text()
    groups=defaultdict(list); max_tokens=0; tok=None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(str(args.tokenizer),local_files_only=True)
    old_ops=set();old_source={};old_target=set();files=[]
    for root in (ROOT/'datasets',ROOT/'training'):
        for file in sorted(root.rglob('*.jsonl')):
            if DATA in file.parents or 'artifacts' in file.parts or 'predictions' in file.name:continue
            touched=False
            for row in read(file):
                ops=row.get('operations')
                for m in row.get('messages',[]):
                    if m['role']=='user':
                        try:ops=json.loads(m['content']).get('operations')
                        except (ValueError,AttributeError):pass
                if not ops or not isinstance(ops,list) or not all(isinstance(o,dict) and 'kind' in o and 'text' in o for o in ops):continue
                old_ops.add(canon(ops));old_source[text(ops,False)]=str(file.relative_to(ROOT));old_target.add(text(ops,True));touched=True
            if touched:files.append({'path':str(file.relative_to(ROOT)),'sha256':sha(file)})
    overlap=[];old_sh=[(shingles(s),path) for s,path in old_source.items()]
    from scripts.surgeon_format_v04 import messages
    from open_surgeon.evidence import ground_verdict
    guard=[]
    for row in inputs:
        label=by_label[row['id']]
        assert set(row)=={'id','messages'}, 'Model-facing input contains metadata or labels'
        assert [m['role'] for m in row['messages']]==['system','user']
        ops=json.loads(row['messages'][1]['content'])['operations']
        assert row['messages']==messages(ops)
        assert all(set(o)=={'index','kind','text'} and o['index']==i and o['kind'] in ('equal','delete','insert') and isinstance(o['text'],str) and o['text'] for i,o in enumerate(ops))
        assert hashlib.sha256(canon(ops).encode()).hexdigest()==row['id']
        original=text(ops,False);revision=text(ops,True)
        s=label['source'];source_path=DATA/s['file'];raw=source_path.read_text().splitlines()[s['line']-1]
        extracted=re.sub(r'\s+',' ',html.unescape(re.sub(r'<[^>]+>','',raw)).replace('**','')).strip()
        assert original==extracted and sha(source_path)==s['source_sha256']
        assert original!=revision and original not in old_source and revision not in old_target and canon(ops) not in old_ops
        assert label['partition']=='test_only' and not label['independent_review']
        expected=label['expected']
        if label['binary_score_eligible']:
            assert expected['decision'] in ('surgical','not_surgical') and label['label_status']=='author_silver'
            assert bool(expected['issues'])==(expected['decision']=='not_surgical')
            for issue in expected['issues']:
                assert issue['operation_indices'] and all(type(i) is int and 0<=i<len(ops) and ops[i]['kind']!='equal' for i in issue['operation_indices'])
            grounded=ground_verdict(ops,expected)
            if grounded['needs_review']:guard.append({'id':row['id'],'errors':grounded['errors']})
        else:
            assert expected['decision'] is None and label['label_status']=='unresolved_preference'
        if tok:
            prompt=tok.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            n=len(tok.encode(prompt,add_special_tokens=False));max_tokens=max(n,max_tokens)
            assert n+256<=4096,(label['family_id'],n)
        sh=shingles(original)
        near=max(((len(sh&other)/len(sh|other) if sh|other else 0,path) for other,path in old_sh),default=(0,None))
        overlap.append({'id':row['id'],'max_source_5word_jaccard':near[0],'nearest_prior_file':near[1]})
        groups[label['family_id']].append((original,revision,label['variant']))
    for family,rs in groups.items():
        assert len(rs)==3 and len({(a,b) for a,b,_ in rs})==1
        assert {v for _,_,v in rs}=={'focused','defect','boundary'}
    for pair in pairs:
        assert (pair['original'],pair['revision'])==groups[pair['family_id']][0][:2]
    result={'rows':96,'families':32,'source_documents':4,'language':'en',
      'binary_core':64,'labels':dict(Counter(x['expected']['decision'] for x in labels if x['binary_score_eligible'])),
      'preference_probes_excluded_from_binary_score':32,'source_reconstruction_matches':96,'same_clean_text_within_family':True,
      'duplicate_inputs':0,'exact_prior_input_source_target_matches':0,'prior_input_files':files,
      'prior_unique_operations':len(old_ops),'prior_unique_sources':len(old_source),
      'max_source_5word_jaccard':max(x['max_source_5word_jaccard'] for x in overlap),
      'model_input_label_metadata_leakage':False,'tokenizer_checked':bool(tok),'max_prompt_tokens':max_tokens or None,
      'context_limit':4096,'reserved_output_tokens':256,'guard_diagnostics_only':guard,
      'guard_used_as_label_oracle':False,'independent_semantic_review':False,'target_model_calls':0}
    (DATA/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (DATA/'overlap-audit.json').write_text(json.dumps(overlap,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='prior_input_files'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
