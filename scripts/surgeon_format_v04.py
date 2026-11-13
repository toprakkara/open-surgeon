"""Explicit operation identifiers; preserve the v0.3 task and output schema."""
import json
from scripts.surgeon_format_v03 import SYSTEM


def messages(operations):
    return [
        {'role': 'system', 'content': SYSTEM + '\nIn the input, every operation has an explicit index field. Copy these index values when citing operations; do not count positions yourself. Verify each cited kind and exact text before answering. Never describe a delete or insert as equal.'},
        {'role': 'user', 'content': json.dumps({'operations': [
            {'index': i, 'kind': op['kind'], 'text': op['text']}
            for i, op in enumerate(operations)
        ]}, ensure_ascii=False)},
    ]
