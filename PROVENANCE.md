# Provenance

Every dataset folder has a `protocol.json` that pins the SHA-256 of its data files, the scripts that built it and the
agent answers it was built from. For this public release, reports, READMEs and the messages inside scripts were
rewritten in English. Data files, labels, predictions, agent answers and teacher answers are byte-identical to the
pinned versions. The scoring and training scripts accept a pinned file only if it matches its pinned hash or is listed
here with both hashes.

## Rewritten for release

| File | Pinned SHA-256 | Release SHA-256 |
|---|---|---|
| `datasets/attorney-surgical-agent-v0.52/README.md` | `15486e572a7a035c…` | `bcec33aef615b1f1…` |
| `datasets/attorney-surgical-verifier-v0.51/README.md` | `a892a6d3f221996e…` | `2c7038c42049fe80…` |
| `datasets/attorney-surgical-wide-fewshot-control-v0.53/README.md` | `302cb9a433520444…` | `eca47492904c02af…` |
| `datasets/attorney-surgical-wide-v0.53/README.md` | `aa2fcc7c4f64b13c…` | `34345d843aa9ba3c…` |
| `scripts/agent_redlines_v052.py` | `27eb700bac325cf0…` | `6ce3b77651fd38a7…` |
| `scripts/fewshot_control_v053.py` | `9a25eb583fcbf566…` | `59db133e828e97b4…` |
| `scripts/prepare_attorney_choice_folds_v050.py` | `9e4a1c4e6ca36a3d…` | `a30705aaaaef5b29…` |
| `scripts/prepare_attorney_choice_sft_v049.py` | `c67d4dbf43789638…` | `42119c194bd6f74c…` |
| `scripts/prepare_attorney_choice_v046.py` | `23d80a9128e289fe…` | `63efb17cf50eef37…` |
| `scripts/prepare_surgical_verifier_v051.py` | `ae379b344fc4abfc…` | `fa4f3bf9a24a4570…` |
| `scripts/score_surgical_agent_v052.py` | `ac0a57bb5ca4c557…` | `3f6a046c3a845f2d…` |
| `scripts/score_surgical_verifier_v051.py` | `f7f1288080c06392…` | `9266a7e01ad974bc…` |
| `scripts/wide_redlines_v053.py` | `e4cc6608a28a60e9…` | `f945dd66ab30a975…` |

## Pinned but not included

None.

Full hashes are in `provenance.json`.
