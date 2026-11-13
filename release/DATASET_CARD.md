---
license: cc-by-4.0
language:
  - en
task_categories:
  - text-classification
tags:
  - legal
  - contracts
  - redline
  - tracked-changes
size_categories:
  - n<1K
source_datasets:
  - micro1-inc/ContractBench
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train.jsonl
      - split: test
        path: data/test.jsonl
---

# Open Surgeon redlines v0.53

A test and a training set for one question: is this contract redline *surgical*, or does it strike and retype text
that did not need to change? Built to follow the test described in Crosby's post
[How to Train your Surgeon](https://crosby.ai/blog/how-to-train-your-surgeon) as closely as possible without attorney
review. Built by Toprak Kara ([@toprakkarra](https://x.com/toprakkarra) on X). Not affiliated with or endorsed by Crosby. Code
and all measurements: https://github.com/toprakkara/open-surgeon.

```python
from datasets import load_dataset

data = load_dataset("toprakkara/open-surgeon-redlines-v0.53")
print(data["test"][0]["label"], data["test"][0]["redline"][:2])
```

## Structure

| File | Rows | Content |
|---|---:|---|
| `data/test.jsonl` | 289 | The test: 126 yes, 163 no, from 126 clauses |
| `data/train.jsonl` | 418 | Labelled training items: 182 yes, 236 no, from 180 clauses |
| `combined-traces.jsonl` | 812 | The chat examples the released adapter was trained on, with teacher reasoning |
| `system-prompt.txt` | | The instruction every system received |
| `agent-task.txt` | | The instruction the two agents received |
| `frozen/` | | The four files exactly as frozen before any measurement; their hashes are pinned in the code repository |

Each row of `data/*.jsonl`:

| Field | Meaning |
|---|---|
| `id` | Hash of the instruction and the redline; the key used in every run's predictions |
| `redline` | Ordered operations, each `{index, kind, text}`. `equal` keeps text, `delete` removes original text, `insert` adds revised text. Accepting every change gives the revised clause; rejecting every change gives the original |
| `label` | `yes` (surgical) or `no` |
| `cell` | `attorney` for yes items; for no items the share of untouched text the agent retyped: `retyped-25`, `retyped-50`, `retyped-75` |
| `group` | Source paragraph. Resample by group, not by row: a clause contributes a yes and one or two no items |
| `clause_pair`, `scenario` | The clause and the ContractBench scenario it comes from |
| `agents`, `retyped`, `share` | For no items: which agent wrote it, how many characters of untouched text it retyped, and that as a share |

`combined-traces.jsonl` holds 812 examples, more than `data/train.jsonl`, because it also reuses teacher traces from two
earlier rounds of the project where their labels satisfy this dataset's rule.

## How it was made

**Source.** [ContractBench](https://huggingface.co/datasets/micro1-inc/ContractBench) by micro1 (CC BY 4.0): Word
contracts with attorneys' tracked-change redlines. Original and revised clause pairs were extracted together with the
attorney's markup. Scenarios are split into three folds; the third is the test. Clauses that occur in more than one
fold, and training clauses whose original text occurs in the test, are left out.

**Labels.**

- **yes**: the attorney's own markup of the clause.
- **no**: a redline of the same clause, with exactly the same original and revised text, written by an AI agent (Claude
  Sonnet or Claude Haiku) that edits with a find-and-replace tool and never sees the attorney's markup. It is kept only
  if it strikes and retypes at least 40 characters, and at least 25%, of the text the attorney left untouched.
- Agent redlines that equal the attorney's markup, or differ without meeting that rule, give no item.

No person labelled or reviewed individual items. The rule was fixed before any test redline or model output existed; it
was revised once after a pilot on training clauses. The record is `plan.json` in the code repository.

**Personal and sensitive information.** None known. ContractBench states that its contracts and redlines are fictional
material written by attorneys for the benchmark, with invented parties, and that attorney identifiers were removed.

## Limits

- **No attorney reviewed these labels.** "No" is decided by a width rule against the attorney's markup. Only clear
  cases are included; the hard middle is absent, so this is narrower than an attorney-labelled test.
- **Derived from a benchmark.** The clauses and "yes" labels come from ContractBench's reference redlines. A model
  trained on this data should not be evaluated on ContractBench.
- One source, five agreement types, English only, two agents from one model family.
- The attorneys' markups in the source are heavy edits (a median of 282 changed characters against a median clause
  length of 422), so "yes" items are not small tidy changes.
- 53 "yes" test items also appear in the project's earlier tests. No model was trained on any test clause.

## Results on the test

| System | Balanced accuracy |
|---|---:|
| Open Surgeon v0.53 (Qwen3.8-27B + LoRA) | 85.9% ± 2.3 |
| GPT-6 Astra, same instruction plus eight worked examples | 78.6% ± 2.2 |
| GPT-6 Astra, same instruction | 74.8% ± 2.2 |
| Heuristic baseline | 66.6% ± 2.6 |

## Licence and attribution

CC BY 4.0. This dataset is adapted from ContractBench by micro1 (CC BY 4.0): clauses were extracted from its documents,
the attorneys' tracked changes were converted to operation lists, and AI-agent redlines, labels and teacher reasoning
were added. Agent redlines and teacher reasoning were generated with Anthropic's Claude models. micro1 does not endorse
this dataset.

## Contact

Questions, corrections and pull requests are welcome in the GitHub repository. If you work on legal AI and want to
collaborate, reach out on X at [@toprakkarra](https://x.com/toprakkarra).

## Citation

```bibtex
@misc{opensurgeon2026,
  title  = {Open Surgeon v0.53: an open replication of a redline surgicality verifier},
  author = {Kara, Toprak},
  year   = {2026},
  url    = {https://github.com/toprakkara/open-surgeon}
}
```
