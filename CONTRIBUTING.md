# Contributing

Issues and pull requests are welcome: bug reports, new baselines, measurements of other models on the v0.53 test,
better extraction, or a stronger training recipe.

## Ground rules

- **Do not change a frozen round.** A dataset with a `protocol.json` is frozen: its labels, instruction and data files
  stay as they are, so earlier results stay checkable. New behaviour goes into a new version (new folder, new script
  with a new `_vNNN` suffix).
- **Fix the rules before you look at outputs.** For a new measurement, write the plan (what is measured, the claim rule,
  what would count as a failure) into a `plan.json` before running any model, and report the result whatever it is.
- **Report runs whole.** Keep the raw predictions of every run, including failed or disappointing ones.
- **Data you add must be shareable.** Use synthetic text or text you have the right to share, with no personal or
  confidential information. No real client contracts.
- **Results stay honest about their limits.** The v0.53 labels come from a rule, not attorney review, and the numbers
  are not comparable with Crosby's private test. Keep that wording when you build on them.

## Practicalities

- Python 3.9 or later. Scoring needs no packages. Dataset scripts that read Word documents need `lxml`.
- Run scripts from the repository root as modules, for example `python3 -m scripts.score_surgical_wide_v053 --help`.
- Training and measuring the 27B model run on Modal and cost money; say in the pull request what you ran and what it
  cost.
- In the pull request, describe the change, why you made it and how you checked it.

Contributions are accepted under the MIT licence for code and CC BY 4.0 for data.
