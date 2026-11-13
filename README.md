# Open Surgeon

An open replication of the recipe behind Crosby's [Surgeon](https://crosby.ai/blog/how-to-train-your-surgeon): a small
verifier model that reads one contract redline and says whether it is *surgical*, with the data, the test and every
measurement in the open.

Built by [Toprak Kara](https://x.com/toprakkarra). Not affiliated with, sponsored by or endorsed by Crosby. This
repository contains none of Crosby's code, model weights or private data. Nothing here is legal advice.

- **Model:** [toprakkara/open-surgeon-v0.53-qwen3.8-27b-lora](https://huggingface.co/toprakkara/open-surgeon-v0.53-qwen3.8-27b-lora) (LoRA adapter for Qwen3.8-27B)
- **Data and test:** [toprakkara/open-surgeon-redlines-v0.53](https://huggingface.co/datasets/toprakkara/open-surgeon-redlines-v0.53)

## The task

Two redlines can produce the same final text and still cost a reviewer very different effort.

Surgical, as the attorney marked it:

> Ee L. Kew may engage Agents to perform its obligations under this Agreement**{+ only with Customer's prior written
> approval in each instance (email being sufficient)+}**.

Not surgical, as an AI agent marked the same change:

> Ee L. Kew may engage Agents to **[-perform its obligations under this Agreement.-]{+perform its obligations under
> this Agreement only with Customer's prior written approval in each instance (email being sufficient).+}**

The verifier sees one redline at a time, writes a short reasoning, and ends with `{"surgical":"yes"}` or
`{"surgical":"no"}`. It judges how the change is presented, not whether the change is legally right.

## Result

Test `v0.53`: 289 redlines (126 yes, 163 no) from 126 clauses of
[ContractBench](https://huggingface.co/datasets/micro1-inc/ContractBench) that no model here was trained on.

| System | Balanced accuracy | ±1σ | Recall on yes | Recall on no |
|---|---:|---:|---:|---:|
| **Open Surgeon v0.53** (Qwen3.8-27B + LoRA, this work) | **85.9%** | 2.3 | 84.1% | 87.7% |
| GPT-6 Astra, same instruction | 74.8% | 2.2 | 50.8% | 98.8% |
| Heuristic (`open_surgeon.core.measure`, threshold fixed earlier) | 66.6% | 2.6 | 89.7% | 43.6% |

Control, registered before it was run: Astra with the same instruction plus eight worked examples from the training
set scores 78.6% (±2.2; recall 57.1% on yes, 100% on no). The trained model stays 7.4 points ahead, in 99.5% of draws.

The claim rule was written down before any output: a model beats Astra only if its balanced accuracy is higher and a
paired bootstrap over source paragraphs shows it higher in at least 97.5% of draws. The trained model is 11.2 points
higher and ahead in 100% of 2,000 draws.

Astra catches almost every needlessly wide redline but also rejects half of the attorneys' own markups. The trained
model is balanced on both sides.

Full report: [training/surgical-wide-v053/REPORT.md](training/surgical-wide-v053/REPORT.md).
Rules fixed in advance: [training/surgical-wide-v053/plan.json](training/surgical-wide-v053/plan.json).

### Read this before quoting the number

- **No attorney reviewed the test.** Crosby's test is 500 redlines labelled by its attorneys. Here "yes" is the
  attorney's own markup and "no" is decided by a rule: an AI agent's redline of the same clause that strikes and retypes
  at least 40 characters, and at least a quarter, of the text the attorney left untouched. Redlines between the two are
  left out, so the test holds only clear cases.
- **The result is in-distribution.** The model was trained on other clauses from the same source and on redlines from
  the same two agents. Astra saw neither, apart from the eight examples in the control. Transfer to other contracts or
  other agents is not shown.
- **These numbers are not comparable with Crosby's** (Surgeon 90.6%, Astra 83.1%, heuristic 60.4% on their private
  test). The ordering is the same; the data and the labels are different.
- The agents that produced the "no" redlines and the teacher that wrote the reasoning traces are Claude models.
- Astra was run through the Codex CLI on a ChatGPT subscription at medium reasoning effort. Higher effort and other
  frontier models were not measured.
- One training run, one measurement, supervised fine-tuning only. Crosby's second stage (GRPO) was not done.
- ContractBench is itself a benchmark, and its reference redlines are the training material here. The released model
  should not be evaluated on ContractBench.
- The design was chosen after an earlier version failed (see below), and the "no" rule was revised once after a pilot on
  training clauses, before any test redline or model output existed. Both are recorded in `plan.json`.

## How this differs from Crosby's Surgeon

The task and the first training stage follow the post. The rest differs, and the differences matter for how far the
result can be read.

| | Crosby's post | This repository |
|---|---|---|
| Task | One redline in; reasoning and a yes/no verdict out | Same |
| Base model | Qwen-3.5-27B | Qwen3.8-27B |
| Training data | Thousands of redlines from Crosby's own agent and attorneys, plus synthetic examples | 812 examples from ContractBench (fictional contracts redlined by attorneys) and two Claude agents |
| Training label | 1 only if the agent's tracked changes exactly match what the attorney sent | Yes for the attorney's markup; no for an agent redline that meets a width rule |
| Training stages | One epoch of fine-tuning on teacher reasoning, then GRPO with an LLM-judge reward | Two epochs of fine-tuning on teacher reasoning; no GRPO |
| Test | 500 redlines relabelled by attorneys, 38% no | 289 redlines labelled by a rule, 56% no, clear cases only |
| Models compared | Ten frontier models, same prompt | GPT-6 Astra only, same prompt, plus a few-shot control |
| Instruction given to the models | Not published | Written for this project (`system-prompt.txt`) |
| Released | Nothing, as of November 2026 | Code, data, test, adapter and every raw output |

## How the test is built

1. **Clauses.** Original and revised clause pairs with the attorney's tracked changes, extracted from ContractBench.
   Scenarios are split into three folds; the third fold is the test.
2. **Yes.** The attorney's markup of the clause.
3. **No.** Two agents (Claude Sonnet and Claude Haiku) edit the clause with a find-and-replace tool until it equals the
   revised text. They never see the attorney's markup. Each replaced span is shown as deleted and its replacement as
   inserted, which is how needlessly wide redlines arise in agent harnesses. A redline becomes a "no" item only if it
   meets the rule above.
4. **Frozen before measurement.** `datasets/attorney-surgical-wide-v0.53/protocol.json` pins the hash of every file,
   script and agent answer.

Of 477 test clauses, 126 produced at least one clearly wide agent redline. For the rest the agents either matched the
attorney or differed in ways that cannot be labelled without attorney review.

## How the model is trained

- Base model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0), pinned revision.
- Supervised fine-tuning on teacher reasoning, as in the first stage of Crosby's recipe: a teacher model answers each
  training item without seeing the label, and only answers with the correct verdict are kept (rejection sampling).
- 812 kept traces, balanced: 406 yes (attorney markups) and 406 no (redlines that meet the test's rule).
- LoRA r=16, alpha=32, learning rate 5e-5, 2 epochs, batch 8. 38 minutes on one H100, about $3.
- No training example shares a clause or a redline with the test.

## What did not work

Kept in the repository on purpose.

- **v0.51** used an algorithm's minimal diff as the "no" example. The trained model beat Astra (58.6% vs 50.5% on 216
  items) but was not shown to beat the heuristic. Everything sat near chance.
- **v0.52** labelled agent redlines by exact match with the attorney, which is Crosby's *training* label. Astra scored
  57.4% and the trained model 54.7%. Most "no" labels turned out to be harmless boundary choices: in Crosby's data exact
  match gave 2% "no" before attorney review, here it gave 73%. v0.53 was built to ask the question Crosby's test asks.

Reports: [v0.51](training/surgical-verifier-v051/REPORT.md), [v0.52](training/surgical-agent-v052/REPORT.md).

## Reproduce

Scoring needs only Python 3.9+ and no packages:

```sh
# balanced accuracy, bootstrap interval and per-band recall for a run
python3 -m scripts.score_surgical_wide_v053 --run training/surgical-wide-v053/trained-27b --name "Open Surgeon v0.53"

# paired bootstrap between two runs
python3 -m scripts.score_surgical_wide_v053 --compare training/surgical-wide-v053/trained-27b training/surgical-wide-v053/astra

# the few-shot control
python3 -m scripts.fewshot_control_v053 compare training/surgical-wide-v053/trained-27b training/surgical-wide-v053/astra-fewshot-control
```

Training and measuring the 27B model run on [Modal](https://modal.com) (one H100, about $3 and $2):

```sh
uvx --from modal==1.6.3 modal run --detach scripts/modal_surgical_verifier_v051.py::fit \
  --traces training/surgical-wide-v053/teacher/combined-traces.jsonl --name my-adapter --out runs/train

uvx --from modal==1.6.3 modal run scripts/modal_surgical_verifier_v051.py::measure \
  --out runs/measure --adapter my-adapter --data datasets/attorney-surgical-wide-v0.53
```

Every intermediate file is included: the frozen inputs and labels, each agent's raw redlines, each teacher answer, and
each model's raw output. The scripts that built the datasets are included for reading. Re-running them from the original
ContractBench documents also needs the documents themselves and an extraction step that is not part of this repository.

The scripts check the hashes pinned in each `protocol.json` before they run. Reports, READMEs and messages in scripts
were rewritten in English for this release, so their hashes differ from the pinned ones; every data file is
byte-identical. [PROVENANCE.md](PROVENANCE.md) lists each file that differs and both hashes.

## Repository map

| Path | What it is |
|---|---|
| `datasets/attorney-surgical-wide-v0.53/` | The v0.53 test (`development/inputs.jsonl`, `test/annotations.jsonl`) and training items |
| `datasets/attorney-surgical-wide-fewshot-control-v0.53/` | The same test with eight worked examples in the instruction |
| `datasets/attorney-surgical-agent-v0.52/`, `datasets/attorney-surgical-verifier-v0.51/` | The two earlier tests |
| `training/surgical-wide-v053/` | Plan, agent redlines, teacher traces, every run's raw predictions, report |
| `training/surgical-agent-v052/`, `training/surgical-verifier-v051/` | The same for the earlier rounds |
| `scripts/` | Scoring, training, measurement and dataset scripts for v0.51 to v0.53 |
| `open_surgeon/core.py` | The redline engine and the heuristic baseline |
| `release/` | The model and dataset cards and the usage example published on Hugging Face |

## Contact

Questions, ideas and pull requests are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). If you work on legal AI and
want to collaborate, reach out on X at [@toprakkarra](https://x.com/toprakkarra) or open an issue.

## Sources and licences

- Code: [MIT](LICENSE). Third-party notices and the relationship to Crosby: [NOTICE.md](NOTICE.md). To cite:
  [CITATION.cff](CITATION.cff).
- [ContractBench](https://huggingface.co/datasets/micro1-inc/ContractBench) by micro1, CC BY 4.0. The datasets here are
  derived from it and carry the same licence.
- Base model Qwen3.8-27B, Apache-2.0.
- Agent redlines and teacher reasoning traces were generated with Anthropic's Claude models.
- The idea, the task definition and the recipe come from Crosby's post
  [How to Train your Surgeon](https://crosby.ai/blog/how-to-train-your-surgeon) by Julian Viret and Sharan Ramjee.
