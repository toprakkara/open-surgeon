"""Read-only text projection of flat revisions with property-history metadata.

Do not execute older source builders: their frozen artifacts remain immutable.
This extension removes only property-history elements in an in-memory XML copy.
It does not accept document changes, alter source files, or judge visual layout.
"""
import json
import hashlib
from copy import deepcopy
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile
from lxml import etree as ET

from scripts.extract_redlinebench_paragraphs import ROOT, NS, W, extract
from scripts.validate_contract_challenge import read, sha, canon, text

OUT=ROOT/'datasets/redlinebench-format-history-v0.32'
POOLS=[ROOT/'datasets/redlinebench-source-pool',ROOT/'datasets/redlinebench-later-source-pool']
REV='eee1b6790982ed1279e86bec7616b662a61993e6'


def project_original_xml(p, accepted):
    excluded='del' if accepted else 'ins'
    nodes=p.xpath('.//*[self::w:t or self::w:delText or self::w:tab or self::w:br or self::w:cr]'
                  f'[not(ancestor::w:{excluded})][not(ancestor::w:pPr)][not(ancestor::w:rPr)]',namespaces=NS)
    return ''.join({'tab':'\t','br':'\n','cr':'\n'}.get(ET.QName(n).localname,n.text or '') for n in nodes)


def extract_properties_ignored(p):
    before=ET.tostring(p);copy=deepcopy(p)
    histories=copy.xpath('.//w:rPrChange | .//w:pPrChange',namespaces=NS)
    if not histories:raise ValueError('no_property_history')
    for node in histories:
        parent=node.getparent()
        expected=W+('rPr' if node.tag==W+'rPrChange' else 'pPr')
        if parent is None or parent.tag!=expected:raise ValueError('unexpected_property_history_location')
        if node.xpath('.//w:t | .//w:delText | .//w:br | .//w:cr',namespaces=NS):
            raise ValueError('text_content_inside_property_history')
        parent.remove(node)
    # All text revision wrappers, current paragraph-mark changes and unsupported
    # field/move/nested structures remain subject to the original extractor.
    ops=extract(copy)
    for accepted in (False,True):assert text(ops,accepted)==project_original_xml(p,accepted)
    assert before==ET.tostring(p),'Source XML mutated'
    return ops


def main():
    assert not OUT.exists()
    cells=[f's{s}-t{t}' for s in (1,2,3) for t in (1,2,3,4)]
    plan={'created_at':datetime.now(timezone.utc).isoformat(),'source_revision':REV,
          'target_max_inputs':48,'cells':cells,'quota_per_cell':4,
          'selection':'First4 globally unique unseen inputs by hash format32/id in each scenario/turn cell in listed order;all available if fewer. No label oroutcome filtering.',
          'screen':'At least one rPrChange/pPrChange history;after removing onlythose histories in-memory,original flat extractor must succeed. >=6changed operations,retained context,different original/target,maxtext>=80chars,serializedops<=10000chars. Prior development IDs excluded.',
          'independent_projection':'For both states,compare extracted projection withXPath over original untouched XML excluding relevant insert/delete ancestors andall property ancestors.',
          'scope':'Aggregate pending text revisions in native attorney-redline snapshots. Formatting,automatic numbering,comments,page layout andlawyer intent are unscored. Current paragraph-mark revisions,nested revisions,moves,fields andunsupported structures still excluded.',
          'labels':'Same-author pre-output silver;attorney authorship is not an automatic positive;unresolved conventions remain unscored. Preserve every selected case.',
          'inference':'Exactv018 prompt,gpt-6-astra medium,existingChatGPTsubscription,fresh sessions,no successful retry orfallback.',
          'context_ceiling_including_256_reserve':8192,'target_calls_at_plan':0,'student_work':False,'training_allowed':False,'final_test_observed':False,
          'source_use':'external_evaluation_only_never_training','previous_canonical_inputs_retained':1793,
          'references':['https://learn.microsoft.com/en-us/dotnet/api/documentformat.openxml.wordprocessing.runpropertieschange?view=openxml-3.0.1',
                        'https://learn.microsoft.com/en-us/office/open-xml/word/how-to-accept-all-revisions-in-a-word-processing-document'],
          'limits':'Reused three RedlineBench simulated contract scenarios;not new independent contracts orpublisher surgicality labels. Native XML text projection only,no rendered-layout assertion.'}
    OUT.mkdir();(OUT/'selection-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    prior=set();prior_inputs={}
    for p in (ROOT/'datasets').glob('*/development/inputs.jsonl'):
        prior_inputs[str(p.relative_to(ROOT))]=sha(p);prior.update(r['id'] for r in read(p))
    candidates={};exclusions=[];source_hashes={};metadata_hashes={};nfiles=0
    for pool in POOLS:
        path=pool/'manifest.json';manifest=json.loads(path.read_text());assert manifest['revision']==REV
        metadata_hashes[str(path.relative_to(ROOT))]=sha(path)
        for item in manifest['files']:
            if not item['path'].endswith('.docx'):continue
            path=pool/item['path'];assert sha(path)==item['sha256'];source_hashes[str(path.relative_to(ROOT))]=sha(path);nfiles+=1
            task=path.parts[-3];cell='-'.join(task.split('-')[1:3]);assert cell in cells
            with ZipFile(path) as z:root=ET.fromstring(z.read('word/document.xml'))
            for index,p in enumerate(root.xpath('//w:body//w:p',namespaces=NS)):
                if not p.xpath('.//w:rPrChange | .//w:pPrChange',namespaces=NS):continue
                if not p.xpath('.//w:ins | .//w:del',namespaces=NS):continue
                provenance={'task':task,'cell':cell,'path':str(path.relative_to(ROOT)),'paragraph_index':index,
                            'docx_sha256':item['sha256'],'xml_sha256':hashlib.sha256(ET.tostring(p)).hexdigest(),
                            'property_histories':dict(Counter(ET.QName(n).localname for n in p.xpath('.//w:rPrChange | .//w:pPrChange',namespaces=NS)))}
                try:ops=extract_properties_ignored(p)
                except ValueError as e:exclusions.append({**provenance,'reason':str(e)});continue
                original,revision=text(ops,False),text(ops,True);reason=None
                if original==revision or not any(o['kind']=='equal' for o in ops):reason='no_change_or_retained_context'
                elif sum(o['kind']!='equal' for o in ops)<6:reason='fewer_than_six_changed_operations'
                elif max(len(original),len(revision))<80 or len(canon(ops))>10000:reason='length_screen'
                ident=hashlib.sha256(canon(ops).encode()).hexdigest()
                if ident in prior:reason='prior_exact_development_input'
                if reason:exclusions.append({**provenance,'reason':reason});continue
                if ident not in candidates:candidates[ident]={'id':ident,'operations':ops,'original':original,'revision':revision,'provenance':[]}
                candidates[ident]['provenance'].append(provenance)
    selected=[];used=set()
    for cell in cells:
        available=[r for r in candidates.values() if r['id'] not in used and any(p['cell']==cell for p in r['provenance'])]
        available.sort(key=lambda r:hashlib.sha256(('format32/'+r['id']).encode()).hexdigest())
        for r in available[:4]:selected.append({**r,'sampling_cell':cell,'selection_index':len(selected)});used.add(r['id'])
    for name,rows in [('candidates.jsonl',list(candidates.values())),('selected.jsonl',selected),('exclusions.jsonl',exclusions)]:
        (OUT/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    report={'status':'source_extraction_complete_no_labels_or_calls','source_documents_read':nfiles,'eligible_unique_paragraphs':len(candidates),'selected':len(selected),
            'selected_by_cell':dict(Counter(r['sampling_cell'] for r in selected)),'exclusions':len(exclusions),'exclusion_reasons':dict(Counter(r['reason'] for r in exclusions)),
            'all_selected_projections_verified':True,'source_documents_modified':False,'source_sha256':source_hashes,'manifest_sha256':metadata_hashes,'prior_input_sha256':prior_inputs,
            'extractor_sha256':sha(Path(__file__)),'base_extractor_sha256':sha(ROOT/'scripts/extract_redlinebench_paragraphs.py'),'target_calls':0,
            'independent_gold':False,'student_work':False,'final_test_observed':False}
    (OUT/'extraction.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('source_documents_read','eligible_unique_paragraphs','selected','selected_by_cell','exclusion_reasons')},indent=2))


if __name__=='__main__':main()
