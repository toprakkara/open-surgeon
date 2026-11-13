# v0.53: Crosby-style surgicality test, "no" = a needlessly wide redline

The rules were written to `plan.json` before any result. Each question holds one redline; the model writes a short
reasoning and then answers `{"surgical":"yes"}` or `{"surgical":"no"}`.

- **Yes:** the attorney's own markup.
- **No:** a redline of the same clause by an AI agent (Claude Sonnet or Claude Haiku) that edits with a find-and-replace
  tool, kept only if it is clearly wider than the attorney's. Rule: the agent struck and retyped at least 40 characters,
  and at least a quarter, of the text the attorney left untouched.
- Redlines between the two are left out of the test; they cannot be labelled without attorney review (this is what went
  wrong in v0.52).

## Result: 289 questions (126 yes, 163 no), balanced accuracy

| Model | Balanced accuracy | ±1σ | Recall on yes | Recall on no | Invalid |
|---|---:|---:|---:|---:|---:|
| **Fine-tuned Qwen3.8-27B, v0.53 (this work)** | **85.9%** | 2.3 | 84.1% | 87.7% | 0 |
| GPT-6 Astra (same instruction, ChatGPT subscription) | 74.8% | 2.2 | 50.8% | 98.8% | 0 |
| Heuristic (`core.measure`, v0.51 threshold unchanged) | 66.6% | 2.6 | 89.7% | 43.6% | 0 |

±1σ comes from a bootstrap that resamples source paragraphs (2,000 draws, 100 paragraphs).

**Claim rule fixed in advance** (a higher score and ahead in at least 97.5% of paired draws):

- The fine-tuned model is 11.2 points above Astra and ahead in 100% of draws. On this test **it beats Astra.**
- The fine-tuned model is 19.3 points above the heuristic and ahead in 100% of draws.
- Astra is 8.2 points above the heuristic and ahead in 99.8% of draws.

Details: `comparisons.json`.

## Control: Astra with worked examples

In the main comparison Astra sees only the instruction. To check whether the gap comes from that, a control was
registered in `plan.json` before it was run: the same 289 questions with eight worked examples from the training data
added to the instruction (four yes, four no, with the teacher's reasoning). One run; the examples and the wording were
not changed after the output.

| Model | Balanced accuracy | ±1σ | Recall on yes | Recall on no | Invalid |
|---|---:|---:|---:|---:|---:|
| GPT-6 Astra, instruction with examples | 78.6% | 2.2 | 57.1% | 100.0% | 0 |

- The examples raised Astra by 3.8 points (higher in 99.3% of draws).
- The fine-tuned model is 7.4 points above Astra with examples and ahead in 99.45% of draws; the claim rule holds for
  this comparison too.
- Astra with examples still rejects 43% of the attorneys' markups. Eight examples were not enough to change that; more
  examples or a different instruction were not measured.

Data: `datasets/attorney-surgical-wide-fewshot-control-v0.53/`, run: `astra-fewshot-control/`.

## Where the gap comes from

Astra catches almost every wide redline but also says "no" to half of the attorneys' own markups. The fine-tuned model
is balanced on both sides. On the same questions: the model said "yes" to 44 attorney markups that Astra rejected, and
Astra caught 18 wide redlines that the model accepted.

Recall on "no" by the share of the attorney's untouched text that the agent retyped:

| Share retyped | Questions | v0.53 model | Astra | Heuristic |
|---|---:|---:|---:|---:|
| 25–50% | 102 | 83.3% | 98.0% | 32.4% |
| 50–75% | 28 | 89.3% | 100.0% | 46.4% |
| 75% and more | 33 | 100.0% | 100.0% | 75.8% |

## Read with care

- **A rule decides "no", not an attorney.** In Crosby's test attorneys read every redline. Here only clear cases are
  included: in 126 of the 477 test clauses at least one agent produced a clearly wide redline. For the rest the agent's
  redline either equalled the attorney's or fell in between and was left out.
- **The result is in-distribution.** The model was trained on attorney markups of other clauses from the same source
  (ContractBench) and on wide redlines from the same two agents. Astra never saw this source. Crosby's model also wins
  partly because it is trained on its own data; still, transfer to other contracts or other agents is not shown here.
- In the main table Astra got no examples; the instruction is the same as our model's. The eight-example control is
  above. Astra was measured on a ChatGPT subscription through the Codex CLI at medium reasoning effort; higher effort was
  not tried.
- This design was chosen after the v0.52 results were seen. The "no" rule was revised once after a 64-clause pilot on
  training clauses (the first rule almost never applied to this data); the change was made before any test redline or
  model output was seen and is recorded in `plan.json`. The second agent (Haiku) was added so the test would not be too
  small.
- The agents and the teacher that wrote the reasoning are Claude models. The teacher was right on 196 of 214 new wide
  redlines and on 27 of 56 attorney markups; only correct answers were kept.
- The training data has 812 examples (406 yes, 406 no). Every "no" example meets this test's rule; earlier rounds' "no"
  labels that do not meet it were not used. Eight earlier examples whose original clause appears in the test were
  removed.
- 53 "yes" questions (attorney markups) also appeared in the v0.51 or v0.52 tests. No model was trained on these
  clauses.
- One training run, one measurement. The v0.51 and v0.52 models were not measured on this test (Modal budget).
  Crosby's second stage (GRPO) was not done.
- Not comparable with Crosby's numbers. Their test is 500 redlines labelled by attorneys: Surgeon 90.6%, Astra 83.1%,
  heuristic 60.4%. The ordering here is similar, but the data and the labels differ.

## Files

- Data: `datasets/attorney-surgical-wide-v0.53/`
- Agent redlines: `agent/` (two agents, `answers/sonnet/` and `answers/haiku/`)
- Teacher reasoning: `teacher/` (training file `teacher/combined-traces.jsonl`)
- Runs: `trained-27b/`, `astra/`, `astra-fewshot-control/`, `heuristic/`; training: `train-27b/`; usage check:
  `usage-check/`
- Adapter: [toprakkara/open-surgeon-v0.53-qwen3.8-27b-lora](https://huggingface.co/toprakkara/open-surgeon-v0.53-qwen3.8-27b-lora)
- Compute: $5.05 on Modal for this round (training $3.00, measurement $2.05).
