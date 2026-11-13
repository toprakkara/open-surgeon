"""Read-only endpoint projection of inserted text subsequently deleted.

Only the observed ins/del nesting is supported. Canceled text is absent from
both all-rejected and all-accepted endpoints; its history remains in source
DOCX and provenance, not silently recast as a native flat revision.
"""
import json
import hashlib
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
from zipfile import ZipFile
from lxml import etree as ET
from scripts.extract_redlinebench_paragraphs import ROOT,NS,W,UNSUPPORTED,extract
from scripts.prepare_format_history_v032 import POOLS,REV,project_original_xml
from scripts.validate_contract_challenge import read,sha,canon,text

OUT=ROOT/'datasets/redlinebench-cancelled-history-v0.33'


def extract_endpoint(p):
    before=ET.tostring(p);copy=deepcopy(p)
    for node in list(copy.xpath('.//w:rPrChange | .//w:pPrChange',namespaces=NS)):
        parent=node.getparent();expected=W+('rPr' if node.tag==W+'rPrChange' else 'pPr')
        if parent is None or parent.tag!=expected:raise ValueError('unexpected_property_history_location')
        if node.xpath('.//w:t | .//w:delText | .//w:br | .//w:cr',namespaces=NS):raise ValueError('text_content_inside_property_history')
        parent.remove(node)
    tags={ET.QName(n).localname for n in copy.iter()}
    if tags & UNSUPPORTED:raise ValueError('unsupported:'+','.join(sorted(tags & UNSUPPORTED)))
    if copy.xpath('./w:pPr//w:ins | ./w:pPr//w:del',namespaces=NS):raise ValueError('paragraph_mark_revision')
    canceled=[]
    for node in copy.xpath('.//w:ins | .//w:del',namespaces=NS):
        if any(a.tag in (W+'rPr',W+'pPr') for a in node.iterancestors()):continue
        chain=[ET.QName(a).localname for a in reversed(list(node.iterancestors())) if a.tag in (W+'ins',W+'del')]+[ET.QName(node).localname]
        if len(chain)>1:
            if chain!=['ins','del']:raise ValueError('unsupported_nested_pattern:'+ '/'.join(chain))
            canceled.append(node)
    if not canceled:raise ValueError('no_canceled_insertion_history')
    history=[]
    for node in canceled:
        value=''.join({'tab':'\t','br':'\n','cr':'\n'}.get(ET.QName(n).localname,n.text or '') for n in node.xpath('.//*[self::w:t or self::w:delText or self::w:tab or self::w:br or self::w:cr][not(ancestor::w:pPr)][not(ancestor::w:rPr)]',namespaces=NS))
        history.append({'ancestor_pattern':'ins/del','text':value,'text_sha256':hashlib.sha256(value.encode()).hexdigest(),'node_sha256':hashlib.sha256(ET.tostring(node)).hexdigest()})
        node.getparent().remove(node)
    ops=extract(copy)
    for accepted in (False,True):assert text(ops,accepted)==project_original_xml(p,accepted)
    assert before==ET.tostring(p)
    return ops,history


def main():
    assert not OUT.exists();cells=[f's{s}-t{t}' for s in (1,2,3) for t in (1,2,3,4)]
    plan={'created_at':datetime.now(timezone.utc).isoformat(),'source_revision':REV,'target_max_inputs':48,'cells':cells,'quota_per_cell':4,
          'selection':'First4 globally unique unused IDs by hash cancel33/id per scenario/turn cell in listed order;all if fewer. No outcome orlabel filtering.',
          'screen':'Require ins/del nested cancellation. Ignore onlyproperty histories in-memory;reject any other nested pattern,current paragraph marks,fields,moves orunsupported nodes eveninside cancellation. >=6remainingchanged ops,retained context,different endpoints,maxtext>=80chars,serializedops<=10000chars. Prior developmentIDs excluded.',
          'projection':'Omit ins/del content because it is absent fromboth all-rejected andall-accepted endpoint text. Preserve exact canceled nodes/text hashes inprovenance. Verify both endpoints independently againstXPath overoriginal XML. Sourcefiles nevermodified.',
          'scope':'Endpoint text presentation after canceled intermediate insertion history is projected away. NOT the complete native visual revision stream,not per-turn attorney delta. Formatting,automaticnumbering,comments andlayout unscored.',
          'labels':'Same-author pre-output silver;review allselected operations/endpoints beforecalls;uncertain conventions unscored;attorney authorship notautomaticpositive.',
          'inference':'Exactv018 prompt,gpt-6-astra medium,existingChatGPTsubscription,fresh sessions,no successful retry orfallback.',
          'context_ceiling_including_256_reserve':8192,'target_calls_at_plan':0,'student_work':False,'training_allowed':False,'final_test_observed':False,
          'source_use':'external_evaluation_only_never_training','previous_canonical_inputs_retained':1809,
          'references':['https://learn.microsoft.com/en-us/dotnet/api/documentformat.openxml.wordprocessing.insertedrun?view=openxml-3.0.1','https://learn.microsoft.com/en-us/dotnet/api/documentformat.openxml.wordprocessing.deletedrun?view=openxml-3.0.1'],
          'limits':'Same3publicRedlineBench simulated scenarios;no new independent contract orpublisher surgicality label. This derivedendpoint projection omits intermediary review history,whose quality is not evaluated.'}
    OUT.mkdir();(OUT/'selection-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    prior=set();prior_hashes={}
    for path in (ROOT/'datasets').glob('*/development/inputs.jsonl'):
        prior_hashes[str(path.relative_to(ROOT))]=sha(path);prior.update(r['id'] for r in read(path))
    candidates={};exclusions=[];source_hashes={};manifest_hashes={};files=0
    for pool in POOLS:
        mp=pool/'manifest.json';manifest=json.loads(mp.read_text());assert manifest['revision']==REV;manifest_hashes[str(mp.relative_to(ROOT))]=sha(mp)
        for item in manifest['files']:
            if not item['path'].endswith('.docx'):continue
            path=pool/item['path'];assert sha(path)==item['sha256'];files+=1;source_hashes[str(path.relative_to(ROOT))]=sha(path)
            task=path.parts[-3];cell='-'.join(task.split('-')[1:3]);assert cell in cells
            with ZipFile(path) as z:root=ET.fromstring(z.read('word/document.xml'))
            for index,p in enumerate(root.xpath('//w:body//w:p',namespaces=NS)):
                if not p.xpath('.//w:ins//w:del',namespaces=NS):continue
                prov={'task':task,'cell':cell,'path':str(path.relative_to(ROOT)),'paragraph_index':index,'docx_sha256':item['sha256'],'xml_sha256':hashlib.sha256(ET.tostring(p)).hexdigest()}
                try:ops,history=extract_endpoint(p)
                except ValueError as e:exclusions.append({**prov,'reason':str(e)});continue
                original,revision=text(ops,False),text(ops,True);reason=None
                if original==revision or not any(o['kind']=='equal' for o in ops):reason='no_change_or_retained_context'
                elif sum(o['kind']!='equal' for o in ops)<6:reason='fewer_than_six_changed_operations'
                elif max(len(original),len(revision))<80 or len(canon(ops))>10000:reason='length_screen'
                ident=hashlib.sha256(canon(ops).encode()).hexdigest()
                if ident in prior:reason='prior_exact_development_input'
                if reason:exclusions.append({**prov,'reason':reason});continue
                if ident not in candidates:candidates[ident]={'id':ident,'operations':ops,'original':original,'revision':revision,'provenance':[]}
                candidates[ident]['provenance'].append({**prov,'canceled_history':history})
    selected=[];used=set()
    for cell in cells:
        available=[r for r in candidates.values() if r['id'] not in used and any(p['cell']==cell for p in r['provenance'])]
        available.sort(key=lambda r:hashlib.sha256(('cancel33/'+r['id']).encode()).hexdigest())
        for r in available[:4]:selected.append({**r,'sampling_cell':cell,'selection_index':len(selected)});used.add(r['id'])
    for name,rows in [('candidates.jsonl',list(candidates.values())),('selected.jsonl',selected),('exclusions.jsonl',exclusions)]:
        (OUT/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    report={'status':'source_projection_complete_no_labels_or_calls','source_documents_read':files,'eligible_unique_paragraphs':len(candidates),'selected':len(selected),
            'selected_by_cell':dict(Counter(r['sampling_cell'] for r in selected)),'exclusions':len(exclusions),'exclusion_reasons':dict(Counter(r['reason'] for r in exclusions)),
            'all_selected_endpoints_verified':True,'source_documents_modified':False,'source_sha256':source_hashes,'manifest_sha256':manifest_hashes,'prior_input_sha256':prior_hashes,
            'extractor_sha256':sha(Path(__file__)),'base_extractor_sha256':sha(ROOT/'scripts/extract_redlinebench_paragraphs.py'),'projection_helper_sha256':sha(ROOT/'scripts/prepare_format_history_v032.py'),
            'target_calls':0,'student_work':False,'final_test_observed':False}
    (OUT/'extraction.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('source_documents_read','eligible_unique_paragraphs','selected','selected_by_cell','exclusion_reasons')},indent=2))


if __name__=='__main__':main()
