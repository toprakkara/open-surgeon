"""Read pinned first-turn references, requiring exact clean-paragraph matches."""
import json,hashlib,difflib
from pathlib import Path
from zipfile import ZipFile
from collections import Counter,defaultdict
from lxml import etree as ET
from scripts.extract_redlinebench_paragraphs import ROOT,NS,extract
from scripts.prepare_format_history_v032 import project_original_xml,extract_properties_ignored
from scripts.prepare_cancelled_history_v033 import extract_endpoint
from scripts.validate_contract_challenge import read,sha,canon,text

OUT=ROOT/'datasets/contractbench-native-v0.37'
POOL=ROOT/'datasets/contractbench-source-pool'

def project(p):
    if p.xpath('.//w:ins//w:del',namespaces=NS):
        ops,history=extract_endpoint(p);return ops,'canceled_insertion_endpoint',history
    if p.xpath('.//w:rPrChange | .//w:pPrChange',namespaces=NS):
        return extract_properties_ignored(p),'property_history_ignored',[]
    return extract(p),'flat_native_text',[]

def paragraphs(path):
    with ZipFile(path) as z:
        root=ET.fromstring(z.read('word/document.xml'),ET.XMLParser(resolve_entities=False,no_network=True))
    return root.xpath('//w:body//w:p',namespaces=NS)

def main():
    assert not (OUT/'selected.jsonl').exists()
    plan=json.loads((OUT/'source-selection-plan.json').read_text());sources=json.loads((OUT/'source-selection.json').read_text());mp=POOL/'document-manifest-v037.json';manifest=json.loads(mp.read_text())
    assert sha(OUT/'source-selection-plan.json')==manifest['selection_plan_sha256'] and manifest['revision']==plan['revision']
    for path,h in [(POOL/'manifest.json',plan['source_pool_manifest_sha256_at_plan']),(POOL/'inventory.json',plan['source_inventory_sha256']),(OUT/'source-selection.json',plan['source_selection_sha256'])]:assert sha(path)==h
    hashes={f['path']:f['sha256'] for f in manifest['files']}
    for path,h in hashes.items():assert sha(POOL/path)==h
    old_ids=set();prior_hashes={}
    for path in (ROOT/'datasets').glob('*/development/inputs.jsonl'):
        prior_hashes[str(path.relative_to(ROOT))]=sha(path);old_ids.update(r['id'] for r in read(path))
    candidates={};excluded=[];document_checks=[]
    for source in sources:
        refpath,basepath=[POOL/source[k] for k in ('reference_path','baseline_path')]
        refs,bases=paragraphs(refpath),paragraphs(basepath)
        baseline_revisions=sum(bool(p.xpath('.//w:ins | .//w:del',namespaces=NS)) for p in bases)
        baseline_map=defaultdict(list);baseline_unsupported=Counter()
        for index,p in enumerate(bases):
            try:ops,mode,history=project(p)
            except ValueError as error:baseline_unsupported[str(error)]+=1;continue
            if any(o['kind']!='equal' for o in ops):continue
            baseline_map[text(ops,False)].append(index)
        bp=[project_original_xml(p,False) for p in bases];rp=[project_original_xml(p,False) for p in refs]
        sequence=difflib.SequenceMatcher(None,bp,rp,autojunk=False)
        document_checks.append({'task':source['task'],'scenario':source['scenario'],'baseline_paragraphs':len(bases),'reference_paragraphs':len(refs),'baseline_revision_paragraphs':baseline_revisions,'baseline_unsupported':dict(baseline_unsupported),'raw_rejected_paragraph_sequences_identical':bp==rp,'sequence_ratio':sequence.ratio(),'non_equal_sequence_regions':sum(tag!='equal' for tag,*_ in sequence.get_opcodes()),'interpretation':'Sequence difference may include paragraph-mark edits andempty inserted paragraphs aswellasuntrackedchanges. No whole-document integrityclaim. Selected original mustexactlymatchone supported cleanparagraph.'})
        for index,p in enumerate(refs):
            if not p.xpath('.//w:ins | .//w:del',namespaces=NS):continue
            prov={**source,'path':str(refpath.relative_to(ROOT)),'baseline_file':str(basepath.relative_to(ROOT)),'paragraph_index':index,'docx_sha256':hashes[source['reference_path']],'baseline_docx_sha256':hashes[source['baseline_path']],'xml_sha256':hashlib.sha256(ET.tostring(p)).hexdigest(),'inside_table_cell':bool(p.xpath('ancestor::w:tc',namespaces=NS))}
            if baseline_revisions:excluded.append({**prov,'reason':'baseline_not_clean'});continue
            try:ops,mode,history=project(p)
            except ValueError as error:excluded.append({**prov,'reason':str(error)});continue
            original,revision=text(ops,False),text(ops,True);reason=None
            if original==revision or not any(o['kind']=='equal' for o in ops):reason='no_change_or_retained_context'
            elif sum(o['kind']!='equal' for o in ops)<6:reason='fewer_than_six_changed_operations'
            elif max(len(original),len(revision))<80 or len(canon(ops))>10000:reason='length_screen'
            matches=baseline_map.get(original,[])
            if reason is None and len(matches)!=1:reason='no_unique_exact_supported_clean_paragraph_match'
            ident=hashlib.sha256(canon(ops).encode()).hexdigest()
            if ident in old_ids:reason='prior_exact_development_input'
            if reason:excluded.append({**prov,'reason':reason});continue
            baseline_index=matches[0];base_node=bases[baseline_index];base_ops,_,_=project(base_node)
            assert text(base_ops,False)==original==project_original_xml(p,False)
            assert revision==project_original_xml(p,True)
            prov.update(projection_mode=mode,canceled_history=history,baseline_paragraph_index=baseline_index,baseline_xml_sha256=hashlib.sha256(ET.tostring(base_node)).hexdigest())
            if ident not in candidates:candidates[ident]={'id':ident,'operations':ops,'original':original,'revision':revision,'provenance':[]}
            candidates[ident]['provenance'].append(prov)
    selected=[];used=set()
    for source in sources:
        available=[r for r in candidates.values() if r['id'] not in used and any(p['task']==source['task'] for p in r['provenance'])]
        available.sort(key=lambda r:hashlib.sha256(('contractbench37/paragraph/'+r['id']).encode()).hexdigest())
        for r in available[:2]:selected.append({**r,'sampling_task':source['task'],'sampling_scenario':source['scenario'],'selection_index':len(selected)});used.add(r['id'])
    for name,rows in [('candidates.jsonl',list(candidates.values())),('selected.jsonl',selected),('exclusions.jsonl',excluded)]:
        (OUT/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    (OUT/'document-integrity-review.json').write_text(json.dumps(document_checks,indent=2)+'\n')
    report={'status':'source_projection_complete_no_labels_or_calls','source_reference_documents':42,'paired_baseline_documents':42,'source_scenarios':21,'eligible_unique_paragraphs':len(candidates),'selected':len(selected),'selected_by_scenario':dict(Counter(r['sampling_scenario'] for r in selected)),'selected_by_task':dict(Counter(r['sampling_task'] for r in selected)),'exclusions':len(excluded),'exclusion_reasons':dict(Counter(r['reason'] for r in excluded)),'all_selected_endpoints_verified':True,'all_selected_originals_match_unique_supported_clean_paragraph':True,'whole_document_baseline_integrity_claimed':False,'source_documents_modified':False,'source_sha256':{str((POOL/p).relative_to(ROOT)):h for p,h in hashes.items()},'prior_input_sha256':prior_hashes,'document_manifest_sha256':sha(mp),'selection_plan_sha256':sha(OUT/'source-selection-plan.json'),'extractor_sha256':sha(Path(__file__)),'helper_sha256':{p:sha(ROOT/p) for p in ['scripts/extract_redlinebench_paragraphs.py','scripts/prepare_format_history_v032.py','scripts/prepare_cancelled_history_v033.py']},'target_calls':0,'student_work':False,'project_locked_final_observed':False}
    (OUT/'extraction.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['eligible_unique_paragraphs','selected','selected_by_scenario','exclusion_reasons']},indent=2))

if __name__=='__main__':main()
