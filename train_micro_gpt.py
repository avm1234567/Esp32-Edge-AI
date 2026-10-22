"""
train_micro_gpt.py - Train a Micro-Transformer and export weights for ESP32-C3

Usage:
    pip install torch
    python train_micro_gpt.py [--epochs 500] [--data your_text.txt]
"""

import os
import sys
import math
import argparse
import urllib.request

try:
    import torch
    import torch.nn as nn
    from torch.nn import functional as F
except ImportError:
    print("=" * 60)
    print("PyTorch is required for training.")
    print("Please install PyTorch: pip install torch")
    print("=" * 60)
    sys.exit(1)

# Hyperparameters matching ESP32 firmware
VOCABULARY = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
VOCAB_SIZE = len(VOCABULARY)
BLOCK_SIZE = 32
N_EMBD = 32
N_HEAD = 2
N_LAYER = 1
HEAD_DIM = N_EMBD // N_HEAD

char_to_id = {ch: i for i, ch in enumerate(VOCABULARY)}
id_to_char = {i: ch for i, ch in enumerate(VOCABULARY)}

class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn = nn.Linear(N_EMBD, 3 * N_EMBD)
        self.c_proj = nn.Linear(N_EMBD, N_EMBD)
        self.register_buffer("bias", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
                                     .view(1, 1, BLOCK_SIZE, BLOCK_SIZE))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(N_EMBD, dim=2)
        k = k.view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)
        q = q.view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)
        v = v.view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(HEAD_DIM))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(N_EMBD, 4 * N_EMBD)
        self.c_proj = nn.Linear(4 * N_EMBD, N_EMBD)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention()
        self.ln_2 = nn.LayerNorm(N_EMBD)
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class MicroGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(VOCAB_SIZE, N_EMBD),
            wpe = nn.Embedding(BLOCK_SIZE, N_EMBD),
            h = nn.ModuleList([Block() for _ in range(N_LAYER)]),
            ln_f = nn.LayerNorm(N_EMBD),
        ))
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.7):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= BLOCK_SIZE else idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def export_to_c_header(model, out_path="main/model_weights.h"):
    state = model.state_dict()
    
    def fmt_1d(arr, name):
        vals = arr.detach().cpu().numpy().flatten()
        lines = [f"const float {name}[{len(vals)}] = {{"]
        for i in range(0, len(vals), 8):
            chunk = ", ".join(f"{v:9.6f}f" for v in vals[i:i+8])
            comma = "," if i + 8 < len(vals) else ""
            lines.append(f"    {chunk}{comma}")
        lines.append("};\n")
        return "\n".join(lines)

    def fmt_2d(arr, name):
        mat = arr.detach().cpu().numpy()
        flat = mat.flatten()
        lines = [f"const float {name}[{len(flat)}] = {{ // shape {list(mat.shape)}"]
        for i in range(0, len(flat), 8):
            chunk = ", ".join(f"{v:9.6f}f" for v in flat[i:i+8])
            comma = "," if i + 8 < len(flat) else ""
            lines.append(f"    {chunk}{comma}")
        lines.append("};\n")
        return "\n".join(lines)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("#ifndef MODEL_WEIGHTS_H\n#define MODEL_WEIGHTS_H\n\n")
        f.write("// Trained Micro-Transformer weights exported from PyTorch\n")
        f.write("#ifndef VOCAB_SIZE\n")
        f.write(f"#define VOCAB_SIZE {VOCAB_SIZE}\n")
        f.write("#endif\n")
        f.write("#ifndef BLOCK_SIZE\n")
        f.write(f"#define BLOCK_SIZE {BLOCK_SIZE}\n")
        f.write("#endif\n")
        f.write("#ifndef N_EMBD\n")
        f.write(f"#define N_EMBD     {N_EMBD}\n")
        f.write("#endif\n")
        f.write("#ifndef N_HEAD\n")
        f.write(f"#define N_HEAD     {N_HEAD}\n")
        f.write("#endif\n")
        f.write("#ifndef N_LAYER\n")
        f.write(f"#define N_LAYER    {N_LAYER}\n")
        f.write("#endif\n\n")

        f.write(fmt_2d(state["transformer.wte.weight"], "wte_weight"))
        f.write(fmt_2d(state["transformer.wpe.weight"], "wpe_weight"))

        # Layer 0
        f.write(fmt_1d(state["transformer.h.0.ln_1.weight"], "layer0_ln1_gamma"))
        f.write(fmt_1d(state["transformer.h.0.ln_1.bias"], "layer0_ln1_beta"))
        f.write(fmt_2d(state["transformer.h.0.attn.c_attn.weight"], "layer0_qkv_weight"))
        f.write(fmt_1d(state["transformer.h.0.attn.c_attn.bias"], "layer0_qkv_bias"))
        f.write(fmt_2d(state["transformer.h.0.attn.c_proj.weight"], "layer0_proj_weight"))
        f.write(fmt_1d(state["transformer.h.0.attn.c_proj.bias"], "layer0_proj_bias"))

        f.write(fmt_1d(state["transformer.h.0.ln_2.weight"], "layer0_ln2_gamma"))
        f.write(fmt_1d(state["transformer.h.0.ln_2.bias"], "layer0_ln2_beta"))
        f.write(fmt_2d(state["transformer.h.0.mlp.c_fc.weight"], "layer0_fc1_weight"))
        f.write(fmt_1d(state["transformer.h.0.mlp.c_fc.bias"], "layer0_fc1_bias"))
        f.write(fmt_2d(state["transformer.h.0.mlp.c_proj.weight"], "layer0_fc2_weight"))
        f.write(fmt_1d(state["transformer.h.0.mlp.c_proj.bias"], "layer0_fc2_bias"))

        # Final LN & LM Head
        f.write(fmt_1d(state["transformer.ln_f.weight"], "ln_f_gamma"))
        f.write(fmt_1d(state["transformer.ln_f.bias"], "ln_f_beta"))
        f.write(fmt_2d(state["lm_head.weight"], "lm_head_weight"))
        f.write(fmt_1d(state["lm_head.bias"], "lm_head_bias"))

        f.write("#endif // MODEL_WEIGHTS_H\n")
    print(f"\n[+] Exported weights successfully to {out_path}!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--data", type=str, default="", help="Path to text data file")
    args = parser.parse_args()

    # Load training text
    if args.data and os.path.exists(args.data):
        with open(args.data, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        cache_file = "tinyshakespeare.txt"
        if not os.path.exists(cache_file):
            print("Downloading Tiny Shakespeare dataset...")
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            urllib.request.urlretrieve(url, cache_file)
        with open(cache_file, "r", encoding="utf-8") as f:
            text = f.read()

    print(f"Dataset length: {len(text):,} characters")

    # Filter text to supported vocabulary
    encoded = [char_to_id[c] for c in text if c in char_to_id]
    data = torch.tensor(encoded, dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    def get_batch(split):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - BLOCK_SIZE, (args.batch_size,))
        x = torch.stack([d[i:i+BLOCK_SIZE] for i in ix])
        y = torch.stack([d[i+1:i+BLOCK_SIZE+1] for i in ix])
        return x, y

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    model = MicroGPT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,} (~{param_count * 4 / 1024:.1f} KB in float32)")

    print("\nStarting training...")
    for step in range(args.steps):
        xb, yb = get_batch("train")
        xb, yb = xb.to(device), yb.to(device)

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (step + 1) % 100 == 0 or step == 0:
            print(f"Step {step+1:4d}/{args.steps:4d} | Loss: {loss.item():.4f}", flush=True)

    # Preview generation
    print("\n--- Model Output Preview ---")
    start_context = torch.tensor([[char_to_id.get(c, 0) for c in "First Citizen:\n"]], dtype=torch.long, device=device)
    generated = model.generate(start_context, max_new_tokens=100)
    gen_text = "".join([id_to_char[i] for i in generated[0].tolist()])
    print(gen_text)
    print("----------------------------")

    # Export to C header
    export_to_c_header(model, "main/model_weights.h")

if __name__ == "__main__":
    main()
