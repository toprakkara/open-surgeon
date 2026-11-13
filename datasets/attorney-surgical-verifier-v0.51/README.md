# Crosby-style surgicality verifier, v0.51

Each question holds one redline; the model writes a short reasoning and answers yes or no to "is this surgical?".
The label is attorney match: yes when the redline is exactly the attorney's markup, no when it is another, automatic
markup of the same change. This is Crosby's training label; Crosby's test is a private set relabelled by attorneys, so
the numbers are not comparable with Crosby's. The result measures likeness to the attorney.

Test: fold 3 (fold index 2) of v0.46 and v0.47, with the `targeted` diff as the "no" example. Training: scenarios of
folds 1 and 2, with the `word`, `character` and `whole` diffs as "no" examples. Test scenarios and paragraphs are kept
out of training.

Source: [ContractBench](https://huggingface.co/datasets/micro1-inc/ContractBench), CC BY 4.0.
