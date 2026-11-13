"""Train and measure the v0.51 reasoning verifier (training/surgical-verifier-v051/plan.json) on Modal GPUs.

Run from the repo root with `uvx --from modal==1.6.3 modal run scripts/modal_surgical_verifier_v051.py::<entrypoint>`:
  ::measure --out <dir> [--adapter <name>] [--limit N] [--check-batching]   score-ready run folder on the 216 test items
  --detach ::fit --traces <jsonl> --name <name> --out <dir> [--resume]      start training and return
  ::collect --name <name> --out <dir>                                      copy a finished adapter to training/artifacts/

Self-contained on purpose: the container only receives this file.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import modal

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'datasets/attorney-surgical-verifier-v0.51'
PLAN = ROOT / 'training/surgical-verifier-v051/plan.json'
BASE, REVISION = 'Qwen/Qwen3.8-27B', '37561816e66b5b9153b1b64c10968c18c1e6f4a0'
GPU = 'H100'
CACHE = '/cache'
MEASURE_TIMEOUT = 60 * 60
TRAIN_TIMEOUT = 2 * 60 * 60
MAX_NEW_TOKENS = 512
TARGET = (r'model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(up|down|gate)_proj'
          r'|linear_attn\.(in_proj_a|in_proj_b|in_proj_qkv|in_proj_z|out_proj))')

app = modal.App('open-surgeon-surgical-verifier-v051')
image = modal.Image.debian_slim(python_version='3.12').pip_install('torch', 'transformers==5.22.1', 'peft', 'accelerate', 'safetensors')
volume = modal.Volume.from_name('open-surgeon-v049', create_if_missing=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path, limit=0):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    return rows[:limit] if limit else rows


def check_frozen(data=DATA):
    """Pinned files must match their hash, or be listed in provenance.json as rewritten for release or not included."""
    provenance = json.loads((ROOT / 'provenance.json').read_text())
    for name, digest in json.loads((data / 'protocol.json').read_text())['frozen_sha256'].items():
        path = (data / name).resolve()
        rel = str(path.relative_to(ROOT))
        if path.exists() and sha(path) == digest:
            continue
        entry = provenance['rewritten'].get(rel)
        if entry and entry['original_sha256'] == digest and path.exists() and sha(path) == entry['release_sha256']:
            continue
        assert rel in provenance['not_included'] and not path.exists(), name


def load_base():
    os.environ['HF_HOME'] = f'{CACHE}/hf'
    import torch
    import transformers
    volume.reload()
    tokenizer = transformers.AutoTokenizer.from_pretrained(BASE, revision=REVISION)
    model = transformers.AutoModelForImageTextToText.from_pretrained(BASE, revision=REVISION, dtype=torch.bfloat16, device_map='cuda:0')
    return tokenizer, model


def versions():
    import peft
    import torch
    import transformers
    return {'torch': str(torch.__version__), 'transformers': str(transformers.__version__), 'peft': str(peft.__version__),
            'device': torch.cuda.get_device_name(0)}


def check_adapter(model, directory):
    import torch
    from safetensors.torch import load_file
    live = dict(model.named_parameters())
    saved = load_file(f'{directory}/adapter_model.safetensors')
    for name, tensor in saved.items():
        if not torch.equal(live[name.replace('.weight', '.default.weight')].detach().cpu().to(tensor.dtype), tensor):
            raise ValueError(f'Adapter tensor did not load: {name}')
    return len(saved)


def encode(tokenizer, messages):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tokenizer(prompt, add_special_tokens=False)['input_ids']


def generate(model, tokenizer, prompts, batch_size):
    """Greedy decoding. Left padding lets a batch share one call; check_batching compares it with batch size 1."""
    import torch
    stop_ids = sorted({tokenizer.convert_tokens_to_ids('<|im_end|>'), tokenizer.eos_token_id})
    pad = stop_ids[0]
    outputs = [None] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    for start in range(0, len(order), batch_size):
        members = order[start:start + batch_size]
        width = max(len(prompts[i]) for i in members)
        ids = torch.tensor([[pad] * (width - len(prompts[i])) + prompts[i] for i in members]).to('cuda')
        mask = torch.tensor([[0] * (width - len(prompts[i])) + [1] * len(prompts[i]) for i in members]).to('cuda')
        began = time.monotonic()
        with torch.inference_mode():
            result = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, temperature=None,
                                    top_p=None, top_k=None, repetition_penalty=1.0, eos_token_id=stop_ids, pad_token_id=pad)
        seconds = round((time.monotonic() - began) / len(members), 2)
        for row, index in enumerate(members):
            new = result[row][width:].tolist()
            stopped = any(token in stop_ids for token in new)
            if stopped:
                new = new[:min(new.index(token) for token in stop_ids if token in new)]
            outputs[index] = {'output': tokenizer.decode(new, skip_special_tokens=True), 'finish_reason': 'stop' if stopped else 'length',
                              'input_tokens': len(prompts[index]), 'output_tokens': len(new), 'seconds': seconds}
        print(f'{min(start + batch_size, len(order))}/{len(order)}', flush=True)
    return outputs


@app.function(image=image, gpu=GPU, timeout=MEASURE_TIMEOUT, volumes={CACHE: volume})
def evaluate(rows, adapter=None, batch_size=8, check_batching=False):
    import peft
    import torch
    tokenizer, model = load_base()
    tensors = 0
    if adapter:
        model = peft.PeftModel.from_pretrained(model, f'{CACHE}/adapters/{adapter}')
        tensors = check_adapter(model, f'{CACHE}/adapters/{adapter}')
    model.eval()
    torch.manual_seed(42)
    prompts = [encode(tokenizer, row['messages']) for row in rows]
    outputs = generate(model, tokenizer, prompts, batch_size)
    check = None
    if check_batching:
        single = generate(model, tokenizer, prompts, 1)
        check = {'same_output': sum(a['output'] == b['output'] for a, b in zip(outputs, single)), 'n': len(rows),
                 'single': [row['output'] for row in single]}
    return {'results': [dict(output, id=row['id']) for row, output in zip(rows, outputs)], 'adapter_tensors_checked': tensors,
            'versions': versions(), 'batching_check': check}


@app.function(image=image, gpu=GPU, timeout=TRAIN_TIMEOUT, volumes={CACHE: volume})
def train(recipe, name, examples_in, resume=False):
    import peft
    import torch
    import transformers
    random.seed(recipe['seed'])
    torch.manual_seed(recipe['seed'])
    tokenizer, model = load_base()
    volume.commit()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    directory = Path(f'{CACHE}/adapters/{name}')
    log = json.loads((directory / 'metrics.json').read_text())['log'] if resume else []
    if resume:
        model = peft.PeftModel.from_pretrained(model, str(directory), is_trainable=True)
        print(f'resuming after epoch {len(log)}; checked {check_adapter(model, str(directory))} tensors', flush=True)
    else:
        model = peft.get_peft_model(model, peft.LoraConfig(
            r=recipe['r'], lora_alpha=recipe['lora_alpha'], lora_dropout=recipe['lora_dropout'], bias='none', task_type='CAUSAL_LM', target_modules=TARGET))
    trainable = [p for p in model.parameters() if p.requires_grad]
    model.train()
    examples = [(encode(tokenizer, row['messages'][:-1]), tokenizer(row['messages'][-1]['content'] + '<|im_end|>', add_special_tokens=False)['input_ids'])
                for row in examples_in]
    optimizer = torch.optim.AdamW(trainable, lr=recipe['learning_rate'], weight_decay=0.0)
    batch = recipe['batch_size']
    per_epoch = math.ceil(len(examples) / batch)
    total = per_epoch * recipe['epochs']
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, math.ceil(recipe['warmup_ratio'] * total), total)
    step = per_epoch * len(log)
    started = time.monotonic() - (log[-1]['seconds'] if log else 0)
    state_file = directory / 'training-state.pt'
    if resume:
        state = torch.load(state_file, map_location='cuda')
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
    for epoch in range(len(log), recipe['epochs']):
        order = list(range(len(examples)))
        random.Random(recipe['seed'] + epoch).shuffle(order)
        running, seen = 0.0, 0
        for start in range(0, len(order), batch):
            members = order[start:start + batch]
            for index in members:
                prompt, answer = examples[index]
                ids = torch.tensor([prompt + answer]).to('cuda')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(input_ids=ids, logits_to_keep=len(answer) + 1, use_cache=False).logits
                loss = torch.nn.functional.cross_entropy(logits[0, :-1].float(), torch.tensor(answer).to('cuda'))
                (loss / len(members)).backward()
                running += loss.item()
                seen += 1
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0:
                print(f'step {step}/{total} loss {running / seen:.4f} {time.monotonic() - started:.0f}s', flush=True)
        log.append({'epoch': epoch + 1, 'train_loss': running / seen, 'seconds': round(time.monotonic() - started)})
        print(json.dumps(log[-1]), flush=True)
        model.save_pretrained(str(directory))
        torch.save({'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, state_file)
        (directory / 'metrics.json').write_text(json.dumps({'log': log}, indent=2))
        volume.commit()
    metrics = {'log': log, 'steps': step, 'examples': len(examples), 'trainable_parameters': sum(p.numel() for p in trainable),
               'versions': versions(), 'resumed': resume, 'finished': True}
    (directory / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    volume.commit()
    return metrics


@app.local_entrypoint()
def measure(out: str, adapter: str = '', limit: int = 0, batch_size: int = 8, check_batching: bool = False, data: str = ''):
    """data: another frozen dataset folder with the same layout, such as datasets/attorney-surgical-agent-v0.52."""
    source = ROOT / data if data else DATA
    check_frozen(source)
    run = Path(out)
    run.mkdir(parents=True, exist_ok=False)
    outcome = evaluate.remote(read_rows(source / 'development/inputs.jsonl', limit), adapter or None, batch_size, check_batching)
    (run / 'protocol.json').write_text(json.dumps({
        'plan_sha256': sha(PLAN), 'script_sha256': sha(__file__), 'dataset': str(source.relative_to(ROOT)),
        'dataset_protocol_sha256': sha(source / 'protocol.json'), 'inputs_sha256': sha(source / 'development/inputs.jsonl'), 'runner': 'modal', 'gpu': GPU, 'model': BASE, 'revision': REVISION,
        'dtype': 'bfloat16', 'adapter': adapter or None, 'adapter_tensors_checked': outcome['adapter_tensors_checked'],
        'versions': outcome['versions'], 'batch_size': batch_size, 'temperature': 0, 'thinking': False, 'seed': 42,
        'max_new_tokens': MAX_NEW_TOKENS}, indent=2) + '\n')
    with (run / 'predictions.jsonl').open('w') as handle:
        for row in outcome['results']:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    if outcome['batching_check']:
        (run / 'batching-check.json').write_text(json.dumps(outcome['batching_check'], ensure_ascii=False, indent=2) + '\n')
    if not limit:
        (run / 'completed-at.txt').write_text(time.strftime('%Y-%m-%dT%H:%M:%S%z') + '\n')
    print(json.dumps({'rows': len(outcome['results']), 'versions': outcome['versions'],
                      'batching_same': outcome['batching_check'] and outcome['batching_check']['same_output'],
                      'finish': [r['finish_reason'] for r in outcome['results']].count('length')}))


@app.local_entrypoint()
def fit(traces: str, name: str, out: str, resume: bool = False, limit: int = 0, epochs: int = 0):
    """traces: JSONL of kept examples, each {"messages": [system, user, assistant]}. limit and epochs are for a smoke test only."""
    check_frozen()
    run = Path(out)
    run.mkdir(parents=True, exist_ok=False)
    recipe = json.loads(PLAN.read_text())['student']
    if epochs:
        recipe = dict(recipe, epochs=epochs)
    examples = read_rows(traces, limit)
    call = train.spawn(recipe, name, examples, resume)
    (run / 'protocol.json').write_text(json.dumps({
        'plan_sha256': sha(PLAN), 'script_sha256': sha(__file__), 'dataset_protocol_sha256': sha(DATA / 'protocol.json'),
        'traces': traces, 'traces_sha256': sha(traces), 'examples': len(examples), 'runner': 'modal', 'gpu': GPU, 'model': BASE,
        'revision': REVISION, 'adapter_name': name, 'recipe': recipe, 'resume': resume, 'function_call_id': call.object_id,
        'smoke_test': bool(limit or epochs), 'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}, indent=2) + '\n')
    print(call.object_id)


@app.local_entrypoint()
def collect(name: str, out: str):
    run = Path(out)
    metrics = json.loads(b''.join(volume.read_file(f'adapters/{name}/metrics.json')))
    assert metrics.get('finished'), 'Training has not finished'
    (run / 'train-metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    target = ROOT / 'training/artifacts' / f'modal-adapter-{name}'
    target.mkdir(parents=True, exist_ok=True)
    for file in ('adapter_config.json', 'adapter_model.safetensors'):
        with (target / file).open('wb') as handle:
            for chunk in volume.read_file(f'adapters/{name}/{file}'):
                handle.write(chunk)
    print(json.dumps({'log': metrics['log'], 'adapter': str(target), 'adapter_sha256': sha(target / 'adapter_model.safetensors')}))
