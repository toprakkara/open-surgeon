"""Read native tracked text, retaining provenance; no surgicality labels inferred."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
from lxml import etree as ET

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / 'datasets/redlinebench-source-pool'
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
W = '{' + NS['w'] + '}'
UNSUPPORTED = {'drawing', 'object', 'pict', 'fldChar', 'instrText', 'fldSimple',
               'moveFrom', 'moveTo', 'sdt', 'rPrChange', 'pPrChange',
               'footnoteReference', 'endnoteReference', 'sym'}


def extract(p):
    tags = {ET.QName(e).localname for e in p.iter()}
    if tags & UNSUPPORTED:
        raise ValueError('unsupported:' + ','.join(sorted(tags & UNSUPPORTED)))
    if p.xpath('./w:pPr//w:ins | ./w:pPr//w:del', namespaces=NS):
        raise ValueError('paragraph_mark_revision')
    operations = []
    for node in p.iter():
        tag = ET.QName(node).localname
        if tag not in ('t', 'delText', 'tab', 'br', 'cr'):
            continue
        # Formatting properties may contain tab-stop definitions, not text tabs.
        ancestors = list(node.iterancestors())
        if any(ET.QName(a).localname in ('pPr', 'rPr') for a in ancestors):
            continue
        changes = [ET.QName(a).localname for a in ancestors if a.tag in (W+'ins', W+'del')]
        if len(changes) > 1:
            raise ValueError('nested_revision')
        kind = {'ins': 'insert', 'del': 'delete'}.get(changes[0] if changes else None, 'equal')
        if tag == 'delText' and kind != 'delete':
            raise ValueError('unscoped_deleted_text')
        value = {'tab': '\t', 'br': '\n', 'cr': '\n'}.get(tag, node.text or '')
        if not value:
            continue
        if operations and operations[-1]['kind'] == kind:
            operations[-1]['text'] += value
        else:
            operations.append({'kind': kind, 'text': value})
    # Independent XPath projection of accepted and rejected text.
    for accepted in (False, True):
        excluded = 'del' if accepted else 'ins'
        nodes = p.xpath('.//*[self::w:t or self::w:delText or self::w:tab or self::w:br or self::w:cr]'
                        f'[not(ancestor::w:{excluded})][not(ancestor::w:pPr)][not(ancestor::w:rPr)]', namespaces=NS)
        expected = ''.join({'tab': '\t', 'br': '\n', 'cr': '\n'}.get(ET.QName(n).localname, n.text or '') for n in nodes)
        actual = ''.join(o['text'] for o in operations if o['kind'] != ('delete' if accepted else 'insert'))
        assert expected == actual
    return operations


def main():
    manifest = json.loads((POOL/'manifest.json').read_text())
    candidates, excluded = {}, []
    for item in manifest['files']:
        if not item['path'].endswith('.docx'):
            continue
        path = POOL/item['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
        task = path.parts[-3]
        with ZipFile(path) as z:
            root = ET.fromstring(z.read('word/document.xml'))
        for index, p in enumerate(root.xpath('//w:body//w:p', namespaces=NS)):
            if not p.xpath('.//w:ins | .//w:del', namespaces=NS):
                continue
            provenance = {'task': task, 'path': str(path.relative_to(ROOT)), 'paragraph_index': index,
                          'docx_sha256': item['sha256'], 'xml_sha256': hashlib.sha256(ET.tostring(p)).hexdigest()}
            try:
                ops = extract(p)
            except ValueError as e:
                excluded.append({**provenance, 'reason': str(e)})
                continue
            original = ''.join(o['text'] for o in ops if o['kind'] != 'insert')
            revision = ''.join(o['text'] for o in ops if o['kind'] != 'delete')
            if not ops or original == revision or not any(o['kind']=='equal' for o in ops):
                excluded.append({**provenance, 'reason': 'no_text_change_or_no_retained_context'})
                continue
            if max(len(original), len(revision)) < 80 or len(json.dumps(ops)) > 10000:
                excluded.append({**provenance, 'reason': 'length_screen'})
                continue
            ident = hashlib.sha256(json.dumps(ops, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if ident not in candidates:
                candidates[ident] = {'id': ident, 'operations': ops, 'original': original,
                                     'revision': revision, 'provenance': []}
            candidates[ident]['provenance'].append(provenance)
    # Source-stratified deterministic sample, independent of labels/model outputs.
    selected, used = [], set()
    tasks = sorted({p['task'] for r in candidates.values() for p in r['provenance']})
    for task in tasks:
        available = [r for r in candidates.values() if r['id'] not in used and any(p['task']==task for p in r['provenance'])]
        available.sort(key=lambda r: hashlib.sha256(('20261105-native-firstturn-'+r['id']).encode()).hexdigest())
        for r in available[:6]:
            selected.append({**r, 'sampling_task': task})
            used.add(r['id'])
    out = ROOT/'datasets/redlinebench-native-review-v0.8'
    out.mkdir(exist_ok=True)
    for name, rows in [('candidates.jsonl', list(candidates.values())), ('selected.jsonl', selected), ('exclusions.jsonl', excluded)]:
        (out/name).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    report = {'eligible_unique_paragraphs': len(candidates), 'selected': len(selected),
              'excluded': len(excluded), 'exclusion_reasons': dict(Counter(r['reason'] for r in excluded)),
              'selected_by_task': dict(Counter(r['sampling_task'] for r in selected)),
              'selection_rule': 'First 6 by seeded hash per source task, exact duplicates collapsed',
              'labels_inferred_from_attorney_authorship': False, 'target_calls': 0,
              'reconstruction_check': 'Two independent XML projections per eligible paragraph',
              'scope': 'Paragraph text only. Adjacent identical-kind runs merged; visual formatting, comments, headers and tables as layout are not evaluated.',
              'use': 'External evaluation only, never training; observed subset will be development, not final test'}
    (out/'extraction.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__=='__main__':
    main()
