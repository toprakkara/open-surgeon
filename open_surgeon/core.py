"""Lossless redlines and an explicitly heuristic readability baseline."""

import hashlib
import json
import re
from difflib import SequenceMatcher

VERSION = "heuristic-v1"
MODES = ("targeted", "word", "character", "whole")
MAX_CHARS = 6000
TOKEN = re.compile(r"\w+(?:[ \t]+\(\d+\))?|\s+|[^\w\s]", re.UNICODE)


def validate_pair(source, target):
    if not isinstance(source, str) or not isinstance(target, str):
        raise ValueError("Both texts must be strings.")
    if max(len(source), len(target)) > MAX_CHARS:
        raise ValueError("Her metin en fazla 6.000 karakter olabilir.")
    if not source and not target:
        raise ValueError("Enter at least one text.")


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def redline(source, target, mode="targeted"):
    validate_pair(source, target)
    if mode not in MODES:
        raise ValueError("Unknown comparison method.")
    if mode == "whole":
        ops = ([{"kind": "equal", "text": source}] if source == target else
               [{"kind": k, "text": t} for k, t in (("delete", source), ("insert", target)) if t])
    else:
        if mode == "character":
            left, right = list(source), list(target)
        elif mode == "word":
            left, right = re.findall(r"\w+|\s+|[^\w\s]", source), re.findall(r"\w+|\s+|[^\w\s]", target)
        else:
            left, right = TOKEN.findall(source), TOKEN.findall(target)
        ops = []
        for tag, a, b, c, d in SequenceMatcher(None, left, right, autojunk=False).get_opcodes():
            old, new = "".join(left[a:b]), "".join(right[c:d])
            if tag == "equal":
                ops.append({"kind": "equal", "text": old})
            elif mode == "targeted" and tag == "replace" and old.casefold() == new.casefold():
                # Isolate case-only changes, retaining all other characters.
                for inner, i, j, k, l in SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
                    if inner == "equal":
                        ops.append({"kind": "equal", "text": old[i:j]})
                    else:
                        if old[i:j]:
                            ops.append({"kind": "delete", "text": old[i:j]})
                        if new[k:l]:
                            ops.append({"kind": "insert", "text": new[k:l]})
            else:
                if old:
                    ops.append({"kind": "delete", "text": old})
                if new:
                    ops.append({"kind": "insert", "text": new})
    compact = []
    for op in ops:
        if compact and compact[-1]["kind"] == op["kind"]:
            compact[-1]["text"] += op["text"]
        else:
            compact.append(dict(op))
    assert reconstruct(compact, False) == source
    assert reconstruct(compact, True) == target
    return compact


def reconstruct(ops, accepted=True):
    omitted = "delete" if accepted else "insert"
    return "".join(op["text"] for op in ops if op["kind"] != omitted)


def measure(source, target, ops, reference_chars):
    changes = [op for op in ops if op["kind"] != "equal"]
    changed = sum(len(op["text"]) for op in changes)
    fragments = 0
    old_pos = new_pos = 0
    for op in ops:
        text, kind = op["text"], op["kind"]
        if kind != "equal":
            document, start = (source, old_pos) if kind == "delete" else (target, new_pos)
            end = start + len(text)
            split = ((start > 0 and document[start - 1].isalnum() and text[0].isalnum()) or
                     (end < len(document) and text[-1].isalnum() and document[end].isalnum()))
            fragments += int(split)
        if kind != "insert":
            old_pos += len(text)
        if kind != "delete":
            new_pos += len(text)
    whitespace = sum(op["text"].isspace() for op in changes)
    excess = max(0, changed - reference_chars)
    excess_ratio = excess / max(changed, 1)
    penalty = 40 * excess_ratio + min(30, fragments * 5) + min(20, max(0, len(changes) - 6) * 2) + min(10, whitespace * 2)
    score = round(max(0, 100 - penalty))
    reasons = []
    if excess:
        reasons.append(f"{excess} more characters are marked than with the targeted method.")
    if fragments:
        reasons.append(f"{fragments} changes split a word or number; case corrections are counted too.")
    if len(changes) > 6:
        reasons.append("Many separate changes can add review steps.")
    if whitespace:
        reasons.append(f"{whitespace} changes contain only whitespace; they may be needed for formatting.")
    if not changes:
        reasons.append("The texts are identical; there is no change to assess.")
    elif not reasons:
        reasons.append("These rules found no clear presentation problem; a human review is still needed.")
    return {"score": score, "changed_characters": changed, "tracked_changes": len(changes),
            "fragmented_changes": fragments, "whitespace_changes": whitespace, "reasons": reasons}


def compare(source, target):
    validate_pair(source, target)
    alternatives = {mode: redline(source, target, mode) for mode in MODES}
    reference = sum(len(op["text"]) for op in alternatives["targeted"] if op["kind"] != "equal")
    group_id = digest([source, target])
    return {"source": source, "target": target, "group_id": group_id, "version": VERSION,
            "identical": source == target,
            "candidates": [{"id": digest([group_id, ops]), "mode": mode, "operations": ops,
                            "metrics": measure(source, target, ops, reference)}
                           for mode, ops in alternatives.items()]}
