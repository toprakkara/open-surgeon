# v0.51: Crosby-style surgicality verifier, first result

The rules were written to `plan.json` before any result. Each question holds one redline; the model writes a short
reasoning and then answers `{"surgical":"yes"}` or `{"surgical":"no"}`. The label is attorney match: yes when the
redline is exactly the attorney's, no when it is the `targeted` diff of the same change.

## Result: 216 questions (109 yes, 107 no), balanced accuracy

| Model | Balanced accuracy | ±1σ | Recall on yes | Recall on no | Invalid |
|---|---:|---:|---:|---:|---:|
| **Fine-tuned Qwen3.8-27B (this work)** | **58.6%** | 3.4 | 81.7% | 35.5% | 0 |
| Heuristic (`core.measure`, threshold chosen on training data) | 55.2% | 2.1 | 89.0% | 21.5% | 0 |
| GPT-6 Astra (same instruction, ChatGPT subscription) | 50.5% | 3.3 | 47.7% | 53.3% | 0 |
| Qwen3.8-27B, not fine-tuned | 40.6% | 2.8 | 50.5% | 30.8% | 76 |

±1σ comes from a bootstrap that resamples source paragraphs (2,000 draws).

**Claim rule fixed in advance:** the fine-tuned model is 8.1 points above Astra and ahead in 99.75% of paired bootstrap
draws (threshold 97.5%). On this test **it beats Astra**. It is 3.3 points above the heuristic but ahead in only 90.4% of
draws; **no advantage over the heuristic was shown.**

## Read with care

- The test is very hard: the "no" items are `targeted` diffs, that is, clean-looking redlines. Astra is at chance on
  this test.
- The model leans towards "yes": it finds 82% of the "yes" items but only 36% of the "no" items.
- One fold and one test. The reasoning traces were written by Claude (Sonnet); the teacher was right on 72.5% of the
  training items, only correct answers were kept, and both classes were balanced at 311 examples each.
- Crosby's second stage (GRPO) was not done.
- The labels are attorney match. They are not comparable with Crosby's 500-redline test relabelled by attorneys;
  Crosby's 90.6% and Astra's 83.1% there are not on the same scale as this table.

## Files

- Data: `datasets/attorney-surgical-verifier-v0.51/`
- Teacher reasoning: `teacher/` (kept traces in `teacher/kept-traces.jsonl`)
- Runs: `trained-27b/`, `untrained-27b/`, `astra/`, `heuristic/`; training: `train-27b/`
- The v0.51 adapter is not published.
