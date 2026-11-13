# Notices

Open Surgeon is an independent project by Toprak Kara. Its own code is released under the [MIT licence](LICENSE).

## Relationship to Crosby

"Surgeon" is the name Crosby uses for the model described in its post
[How to Train your Surgeon](https://crosby.ai/blog/how-to-train-your-surgeon). This project replicates the method
described there from public information. It is not affiliated with, sponsored by or endorsed by Crosby, and it contains
none of Crosby's model weights, training data or private benchmark. The name is used only to say what is being
replicated.

## Not legal advice

Nothing in this repository is legal advice. The models and scores here concern how a change is presented in a redline,
not whether the change is legally correct or advisable.

## Third-party data

| Source | Owner | Licence | Where it is used |
|---|---|---|---|
| [ContractBench](https://huggingface.co/datasets/micro1-inc/ContractBench) | micro1 | CC BY 4.0 | Clauses and attorney redlines behind every dataset in this repository |

The datasets in this repository are adaptations of ContractBench: clauses were extracted from its documents, tracked
changes were converted to operation lists, and labels, AI-agent redlines and model outputs were added. They are shared
under CC BY 4.0. micro1 does not endorse this project.

ContractBench is itself a benchmark. Its reference redlines were used here as training and test material, which its
licence permits; models trained on the data in this repository should therefore not be evaluated on ContractBench.

## Models

The released adapter is a LoRA adapter for [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), which is
licensed under Apache-2.0. The adapter is shared under Apache-2.0.

## AI-generated material

- The agent redlines and the teacher reasoning traces were generated with Anthropic's Claude models.
- The run folders contain raw outputs of OpenAI's GPT-6 Astra, kept as measurement records.

Neither company endorses this project.
