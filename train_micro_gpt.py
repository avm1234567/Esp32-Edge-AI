"""Train, evaluate and export the character transformer for ESP32-S3."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

VOCABULARY = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

class Attention(nn.Module):
    def __init__(self, width, heads, context, dropout):
        super().__init__()
        self.heads = heads
        self.c_attn = nn.Linear(width, 3 * width)
        self.c_proj = nn.Linear(width, width)
        self.dropout = dropout
    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.c_attn(x).chunk(3, dim=-1)
        q, k, v = [a.view(b, t, self.heads, c // self.heads).transpose(1, 2) for a in (q, k, v)]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0)
        return self.c_proj(y.transpose(1, 2).contiguous().view(b, t, c))

class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.c_fc = nn.Linear(width, 4 * width)
        self.c_proj = nn.Linear(4 * width, width)
    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))

class Block(nn.Module):
    def __init__(self, width, heads, context, dropout):
        super().__init__()
        self.ln_1 = nn.LayerNorm(width)
        self.attn = Attention(width, heads, context, dropout)
        self.ln_2 = nn.LayerNorm(width)
        self.mlp = MLP(width)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        x = x + self.drop(self.attn(self.ln_1(x)))
        return x + self.drop(self.mlp(self.ln_2(x)))

class MicroGPT(nn.Module):
    def __init__(self, width=64, heads=2, context=64, layers=2, dropout=0.1):
        super().__init__()
        self.config = dict(width=width, heads=heads, context=context, layers=layers, dropout=dropout)
        self.transformer = nn.ModuleDict(dict(wte=nn.Embedding(65, width),
            wpe=nn.Embedding(context, width),
            h=nn.ModuleList([Block(width, heads, context, dropout) for _ in range(layers)]),
            ln_f=nn.LayerNorm(width)))
        self.lm_head = nn.Linear(width, 65)
        self.apply(self.initialize)
    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)
    def forward(self, ids):
        x = self.transformer.wte(ids) + self.transformer.wpe(torch.arange(ids.shape[1], device=ids.device))
        for block in self.transformer.h:
            x = block(x)
        return self.lm_head(self.transformer.ln_f(x))

# Shared mapping also makes the older exported model measurable without retraining it.
LAYER_TENSORS = dict(ln1_gamma="ln_1.weight", ln1_beta="ln_1.bias",
    qkv_weight="attn.c_attn.weight", qkv_bias="attn.c_attn.bias",
    proj_weight="attn.c_proj.weight", proj_bias="attn.c_proj.bias",
    ln2_gamma="ln_2.weight", ln2_beta="ln_2.bias", fc1_weight="mlp.c_fc.weight",
    fc1_bias="mlp.c_fc.bias", fc2_weight="mlp.c_proj.weight", fc2_bias="mlp.c_proj.bias")
OTHER_TENSORS = dict(wte_weight="transformer.wte.weight", wpe_weight="transformer.wpe.weight",
    ln_f_gamma="transformer.ln_f.weight", ln_f_beta="transformer.ln_f.bias",
    lm_head_weight="lm_head.weight", lm_head_bias="lm_head.bias")

def load_old_header(path):
    model = MicroGPT(width=32, context=32, layers=1, dropout=0)
    state = model.state_dict()
    source = Path(path).read_text()
    mapping = dict(OTHER_TENSORS)
    mapping.update({"layer0_" + name: "transformer.h.0." + key for name, key in LAYER_TENSORS.items()})
    for name, key in mapping.items():
        body = re.search(r"const float " + name + r"\[\d+\]\s*=\s*\{(.*?)\};", source, re.S)[1]
        body = re.sub(r"//[^\n]*", "", body)
        values = [float(x.strip().rstrip('f')) for x in body.split(',') if x.strip()]
        state[key] = torch.tensor(values).reshape(state[key].shape)
    model.load_state_dict(state)
    return model.eval()

@torch.no_grad()
def evaluate(model, data, device, starts, batch_size=32):
    """Score identical next-character targets, using each model's context length."""
    model.eval()
    context = model.config['context']
    total = 0.0
    for off in range(0, len(starts), batch_size):
        positions = starts[off:off+batch_size]
        x = torch.stack([data[p-context:p] for p in positions]).to(device)
        y = data[positions].to(device)
        total += F.cross_entropy(model(x)[:, -1], y, reduction='sum').item()
    return total / len(starts)

@torch.no_grad()
def sample(model, prompt, count=160, temperature=0.8, top_k=20):
    device = next(model.parameters()).device
    ids = [VOCABULARY.index(c) for c in prompt]
    for _ in range(count):
        x = torch.tensor([ids[-model.config['context']:]], device=device)
        logits = model(x)[0, -1] / temperature
        cutoff = logits.topk(top_k).values[-1]
        logits[logits < cutoff] = float('-inf')
        ids.append(torch.multinomial(logits.softmax(-1), 1).item())
    return ''.join(VOCABULARY[i] for i in ids)

def export(model, directory):
    directory = Path(directory)
    state = model.cpu().state_dict()
    cfg = model.config
    count = sum(p.numel() for p in model.parameters())
    (directory / 'model_config.h').write_text(
        '#ifndef MODEL_CONFIG_H\n#define MODEL_CONFIG_H\n' +
        f'#define VOCAB_SIZE 65\n#define BLOCK_SIZE {cfg["context"]}\n#define N_EMBD {cfg["width"]}\n'
        f'#define N_HEAD {cfg["heads"]}\n#define N_LAYER {cfg["layers"]}\n#define MODEL_PARAMETERS {count}\n#endif\n')
    arrays = {name: state[key] for name, key in OTHER_TENSORS.items()}
    for name, key in LAYER_TENSORS.items():
        arrays['layer_' + name] = torch.stack([state[f'transformer.h.{i}.{key}'] for i in range(cfg['layers'])])
    with (directory / 'model_weights.h').open('w') as f:
        f.write('// Trained weights; generated by train_micro_gpt.py. Do not edit.\n#ifndef MODEL_WEIGHTS_H\n#define MODEL_WEIGHTS_H\n')
        for name, tensor in arrays.items():
            values = tensor.numpy().flatten()
            f.write(f'static const float {name}[{len(values)}] = {{\n')
            for offset in range(0, len(values), 8):
                f.write(', '.join(f'{float(v):.9e}f' for v in values[offset:offset+8]) + ',\n')
            f.write('};\n')
        f.write('#endif\n')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=4000)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    p.add_argument('--baseline', default='artifacts/baseline_weights.h')
    p.add_argument('--lr', type=float, default=0.003)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    root = Path(__file__).resolve().parent
    output = root / 'artifacts'; output.mkdir(exist_ok=True)
    raw = (root / 'tinyshakespeare.txt').read_bytes()
    text = raw.decode()
    data = torch.tensor([VOCABULARY.index(c) for c in text], dtype=torch.long)
    train, val, test = data[:int(len(data)*.9)], data[int(len(data)*.9):int(len(data)*.95)], data[int(len(data)*.95):]
    gen = torch.Generator().manual_seed(123)
    val_starts = torch.randint(64, len(val), (1024,), generator=gen)
    test_starts = torch.randint(64, len(test), (2048,), generator=gen)
    baseline = load_old_header(args.baseline).to(args.device)
    baseline_loss = evaluate(baseline, test, args.device, test_starts)
    print(f'Baseline test loss: {baseline_loss:.5f}', flush=True)
    del baseline
    model = MicroGPT().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    best = float('inf'); history = []; start = time.monotonic()
    for step in range(args.steps):
        model.train()
        positions = torch.randint(0, len(train)-65, (args.batch_size,))
        indices = positions[:, None] + torch.arange(64)[None, :]
        x = train[indices].to(args.device); y = train[indices+1].to(args.device)
        warmup = min(1.0, (step+1)/100)
        lr = args.lr * warmup * (.1 + .9 * .5 * (1 + math.cos(math.pi*step/args.steps)))
        for group in optimizer.param_groups: group['lr'] = lr
        loss = F.cross_entropy(model(x).reshape(-1,65), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step+1) % 250 == 0 or step == 0 or step+1 == args.steps:
            score = evaluate(model, val, args.device, val_starts)
            record = dict(step=step+1, train_loss=loss.item(), val_loss=score, elapsed_s=time.monotonic()-start)
            history.append(record); print(json.dumps(record), flush=True)
            if score < best:
                best = score
                torch.save(dict(config=model.config, state={k:v.detach().cpu() for k,v in model.state_dict().items()}, step=step+1), output/'best.pt')
    checkpoint = torch.load(output/'best.pt', weights_only=True)
    model.load_state_dict(checkpoint['state']); model.eval()
    score = evaluate(model, test, args.device, test_starts)
    torch.manual_seed(123)
    samples = [sample(model, prompt) for prompt in ['ROMEO:\n', 'First Citizen:\n', 'To be or ']]
    report = dict(seed=42, dataset_sha256=hashlib.sha256(raw).hexdigest(), split='90% train / 5% validation / 5% test',
        parameters=sum(p.numel() for p in model.parameters()), config=model.config, device=args.device,
        baseline_test_loss=baseline_loss, upgraded_test_loss=score, baseline_perplexity=math.exp(baseline_loss),
        upgraded_perplexity=math.exp(score), selected_step=checkpoint['step'], history=history, samples=samples,
        test_note='2048 identical fixed next-character targets; baseline context 32, upgraded context 64. Baseline training provenance unavailable.')
    (root/'training_report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'history'}, indent=2), flush=True)
    if score >= baseline_loss:
        raise RuntimeError('Candidate did not improve test loss; weights not exported')
    export(model, root/'main')

if __name__ == '__main__': main()
