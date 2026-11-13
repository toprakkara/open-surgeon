"""Minimal use of the Open Surgeon v0.53 adapter, with the settings of the published measurement.

  python usage_example.py --adapter <hf repo id or local folder> --system-prompt system-prompt.txt

Needs torch, transformers 5.22.1, peft and a GPU with about 60 GB of memory.
"""
import argparse
import json

import peft
import torch
import transformers

BASE, REVISION = "Qwen/Qwen3.8-27B", "37561816e66b5b9153b1b64c10968c18c1e6f4a0"
EXAMPLE = [
    {"index": 0, "kind": "equal", "text": "Payment is due within "},
    {"index": 1, "kind": "delete", "text": "thirty (30)"},
    {"index": 2, "kind": "insert", "text": "sixty (60)"},
    {"index": 3, "kind": "equal", "text": " days of invoice."},
]


def load(adapter):
    tokenizer = transformers.AutoTokenizer.from_pretrained(BASE, revision=REVISION)
    model = transformers.AutoModelForImageTextToText.from_pretrained(BASE, revision=REVISION, dtype=torch.bfloat16, device_map="cuda:0")
    return tokenizer, peft.PeftModel.from_pretrained(model, adapter).eval()


def judge(tokenizer, model, system, redline, max_new_tokens=512):
    """redline: ordered operations, each {"index", "kind": equal|delete|insert, "text"}. Returns reasoning ending in the verdict."""
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps({"redline": redline}, ensure_ascii=False)}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    stop = sorted({tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id})
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None,
                                eos_token_id=stop, pad_token_id=stop[0])
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--system-prompt", required=True)
    args = parser.parse_args()
    tokenizer, model = load(args.adapter)
    print(judge(tokenizer, model, open(args.system_prompt).read(), EXAMPLE))
