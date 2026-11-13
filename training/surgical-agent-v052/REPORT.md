# v0.52: a test built with Crosby's training label (agent redlines, attorney match)

The rules were written to `plan.json` before any result. The test is built like Crosby's training labels: an AI agent
(Claude Sonnet) sees the old and the new version of a clause, never the attorney's markup, and writes its own redline.
The label is yes when the agent's redline is exactly the attorney's and no otherwise. There is no attorney relabelling.
Each question holds one redline; the model writes a short reasoning and then answers `{"surgical":"yes"}` or
`{"surgical":"no"}`.

## Result: 477 questions (130 yes, 347 no), balanced accuracy

| Model | Balanced accuracy | ±1σ | Recall on yes | Recall on no | Invalid |
|---|---:|---:|---:|---:|---:|
| GPT-6 Astra (same instruction, ChatGPT subscription) | **57.4%** | 1.5 | 93.8% | 21.0% | 0 |
| Fine-tuned Qwen3.8-27B, v0.52 (this work) | 54.7% | 1.2 | 96.9% | 12.4% | 0 |
| Heuristic (`core.measure`, v0.51 threshold unchanged) | 52.3% | 0.7 | 100.0% | 4.6% | 0 |
| The v0.51 model, as is | 51.0% | 0.7 | 98.5% | 3.5% | 0 |

±1σ comes from a bootstrap that resamples source paragraphs (2,000 draws).

**Claim rule fixed in advance** (a higher score and ahead in at least 97.5% of paired draws):

- The new model **did not beat Astra.** Astra is 2.8 points ahead; the new model is ahead in only 3% of draws. Astra is
  ahead in 97% of draws, just under the threshold, so "Astra is clearly better" cannot be claimed either.
- The new model is 2.4 points above the heuristic but ahead in only 95.25% of draws; **no advantage over the heuristic
  was shown.**
- The new model is 3.7 points above the v0.51 model and ahead in 100% of draws. Training on agent redlines helped, a
  little.
- Astra is above the heuristic (5.1 points, 99.95%) and above the v0.51 model (6.5 points, 100%).

Details: `comparisons.json`.

## Read with care

- **No system solves this test.** All of them say "yes" to almost every redline. Even Astra, the best, finds only 21% of
  the "no" items, 7 points above chance (50%).
- Not comparable with Crosby's test. Crosby's test labels were reviewed by attorneys and 38% of items were "no". Here
  the label only asks "is it exactly the attorney's?", and 73% of items are "no". Crosby's 90.6% and Astra's 83.1% there
  are not on the same scale as this table.
- The agent was Claude, not Crosby's agent. The teacher that wrote the reasoning was also Claude (Sonnet).
- The teacher could not recognise the "no" items: it was right on 249 of 278 "yes" items but on only 87 of 462 "no"
  items. Since only correct answers are kept, 75 agent-type "no" examples reached training. The training data has 888
  examples (444 yes, 444 no): 680 synthetic examples from v0.51 and 208 agent redlines.
- The teacher skipped one item; it was not asked again and was left out of training.
- Crosby's second stage (GRPO) was not done.
- **Correction (2026-11-10):** the statement in `plan.json` that no training paragraph is in the test is wrong for a
  small part of the test. 12 of the 477 questions share their original clause text (not the redline) with 9 training
  examples of the v0.52 model and 8 of the v0.51 model. Any effect favours the two trained models, both of which stayed
  below Astra. Scores and labels are unchanged. Details: `erratum.json`.

## Looked at afterwards: what does "no" mean?

This section was done after the results were seen (`scripts/diagnose_surgical_agent_v052.py`, `diagnostic.json`). It
changes no label and no score.

On the 347 "no" questions the agent's redline differs from the attorney's as follows:

| Difference | Questions | Astra | v0.52 model | Heuristic | v0.51 model |
|---|---:|---:|---:|---:|---:|
| Agent marked fewer characters | 164 | 35 | 24 | 4 | 8 |
| Agent marked more characters | 127 | 31 | 12 | 12 | 3 |
| Only the position of a space differs | 45 | 5 | 5 | 0 | 0 |
| Same amount, different place | 11 | 2 | 2 | 0 | 1 |

The four columns on the right show on how many questions each system said "no".

In about half of the "no" questions (164 of 347) the agent marked **less** than the attorney. Three typical cases:

- Agent marked less: the agent marked only the inserted words; at the same place the attorney wrote
  "[-with -]{+with, and … of, +}", deleting and retyping "with" as well.
- Agent marked more: the attorney marked "[-thirty-]{+sixty+} ([-3-]{+6+}0)"; the agent deleted the whole of
  "thirty (30)" and typed "sixty (60)".
- Only a space: the space next to an inserted word sits on the left in one markup and on the right in the other.

So on most questions "no" does not mean "a bad redline" but "a redline different from the one the attorney chose".
Where the agent marked less, any system that prefers small, clean changes will naturally say "yes". Without attorney
review the two cannot be told apart; the attorney step in Crosby's test is what separates them.

## Files

- Data: `datasets/attorney-surgical-agent-v0.52/`
- Agent redlines: `agent/`; teacher reasoning: `teacher/` (training file `teacher/combined-traces.jsonl`)
- Runs: `astra/`, `trained-27b/`, `heuristic/`, `v051-model/`; training: `train-27b/`
- Comparisons: `comparisons.json`; the later diagnostic: `diagnostic.json`
- The v0.52 adapter is not published.
