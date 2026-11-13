---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: adapter
library_name: peft
pipeline_tag: text-generation
language:
  - en
tags:
  - legal
  - contracts
  - redline
  - tracked-changes
  - lora
datasets:
  - toprakkara/open-surgeon-redlines-v0.53
---

# Open Surgeon v0.53 (LoRA adapter for Qwen3.8-27B)

A verifier for contract redlines. Given one redline as a list of `equal`, `delete` and `insert` operations, it writes a
short reasoning and ends with `{"surgical":"yes"}` or `{"surgical":"no"}`: would a careful attorney send this redline
out exactly as marked, or does it strike and retype text that did not need to change?

It is an open replication of the first training stage described in Crosby's post
[How to Train your Surgeon](https://crosby.ai/blog/how-to-train-your-surgeon). **It is not Crosby's model, was not
built with Crosby's data, and is not affiliated with or endorsed by Crosby.** Code, data and every measurement:
https://github.com/toprakkara/open-surgeon.

## Model details

- **Developed by:** Toprak Kara ([@toprakkarra](https://x.com/toprakkarra) on X).
- **Model type:** LoRA adapter (r=16) for a causal language model; text in, text out.
- **Base model:** [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), revision
  `37561816e66b5b9153b1b64c10968c18c1e6f4a0`, Apache-2.0.
- **Language:** English.
- **Licence:** Apache-2.0 for the adapter weights.

## Uses

**Intended use.** Research on redline quality: checking whether a tracked-change presentation of an already agreed edit
is needlessly wide, for example as a filter or a signal in a contract-drafting pipeline that a person supervises.

**Out of scope.**

- It does not judge whether an edit is legally correct, complete or advisable. It is not legal advice and must not
  replace an attorney's review.
- It has only been measured on English commercial-contract clauses from one source. Do not rely on it for other
  languages, other document types or long documents.
- Do not report this model's results on [ContractBench](https://huggingface.co/datasets/micro1-inc/ContractBench): its
  training data is derived from that benchmark's reference redlines.

## Result

Test `v0.53`, 289 redlines (126 yes, 163 no) from 126 held-out clauses of ContractBench. Balanced accuracy, ±1σ from a
bootstrap over source paragraphs.

| System | Balanced accuracy | Recall on yes | Recall on no |
|---|---:|---:|---:|
| **This adapter** | **85.9% ± 2.3** | 84.1% | 87.7% |
| GPT-6 Astra, same instruction | 74.8% ± 2.2 | 50.8% | 98.8% |
| Heuristic baseline | 66.6% ± 2.6 | 89.7% | 43.6% |

Control, registered before it was run: Astra with eight worked examples from the training set added to the instruction
scores 78.6% ± 2.2. The adapter stays 7.4 points ahead, in 99.5% of paired draws.

The adapter is 11.2 points above Astra and ahead in 100% of 2,000 paired bootstrap draws, under a claim rule fixed
before any output. Astra was run through the Codex CLI on a ChatGPT subscription at medium reasoning effort.

## Limitations and risks

- **No attorney reviewed the test.** "Yes" is the attorney's own markup; "no" is an AI agent's redline of the same
  clause that retypes at least 40 characters, and at least a quarter, of the text the attorney left untouched. Only such
  clear cases are in the test.
- **In-distribution.** The adapter was trained on other clauses from the same source and on redlines from the same two
  agents. Transfer to other contracts, other drafting styles or other agents is not shown.
- **Not comparable with the numbers in Crosby's post**, which come from a private, attorney-labelled test of 500
  redlines.
- **It can be wrong in both directions.** On the test it rejects about 16% of attorney markups and accepts about 12% of
  needlessly wide redlines. Its reasoning text is not checked for faithfulness.
- **One style of "surgical".** It learned the habits of the attorneys who wrote this source's redlines and of a teacher
  model. Other practices may reasonably mark changes differently.
- One training run, one measurement. Supervised fine-tuning only: the post's second stage (GRPO) was not done, and the
  post uses one epoch of fine-tuning where this adapter uses two.

## Use

`usage_example.py` ships with this adapter and holds the complete code. It uses the settings of the measurement:
greedy decoding, thinking disabled, 512 new tokens.

```python
from usage_example import EXAMPLE, judge, load

tokenizer, model = load("toprakkara/open-surgeon-v0.53-qwen3.8-27b-lora")
system = open("system-prompt.txt").read()   # ships with this adapter; the model was trained with exactly this text
print(judge(tokenizer, model, system, EXAMPLE))
```

`EXAMPLE` shows the input form:

```python
[
    {"index": 0, "kind": "equal", "text": "Payment is due within "},
    {"index": 1, "kind": "delete", "text": "thirty (30)"},
    {"index": 2, "kind": "insert", "text": "sixty (60)"},
    {"index": 3, "kind": "equal", "text": " days of invoice."},
]
```

The base model needs about 60 GB of GPU memory in bfloat16.

Checked on 13 November 2026 on one H100 with exactly this file: the example returns a short reasoning ending in
`{"surgical":"yes"}`, and six test items give the same verdicts as the published measurement. The reasoning text can differ
slightly from the measurement, which decoded in batches of eight. Record: `training/surgical-wide-v053/usage-check/result.json`
in the code repository.

## Training

- **Data:** 812 teacher traces, 406 yes and 406 no, from the training folds of the dataset above
  (`combined-traces.jsonl`). No training example shares a clause or a redline with the test.
- **Method:** a teacher model (Claude Sonnet) answered each training item without seeing its label; only answers with
  the correct verdict were kept (rejection sampling). The student was fine-tuned to reproduce the kept reasoning and
  verdict.
- **Hyperparameters:** LoRA r=16, alpha=32, dropout 0 on attention and MLP projections; learning rate 5e-5, cosine decay
  with 3% warm-up; 2 epochs; batch 8; seed 42; loss on the reply only. Final epoch used; no selection on the test.
- **Compute:** 38 minutes on one NVIDIA H100.
- **Software:** torch 2.15.0, transformers 5.22.1, peft 0.21.2.

## Licence and sources

- Adapter weights: Apache-2.0, as the base model.
- Training data is derived from [ContractBench](https://huggingface.co/datasets/micro1-inc/ContractBench) by micro1,
  CC BY 4.0; the contracts in it are fictional and written by attorneys for that benchmark.
- Agent redlines and teacher reasoning were generated with Anthropic's Claude models.
- The task and the recipe come from Crosby's post by Julian Viret and Sharan Ramjee.

## Contact

Questions, ideas and pull requests are welcome in the GitHub repository. If you work on legal AI and want to
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
