import math
import random
import os

VOCABULARY = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
VOCAB_SIZE = len(VOCABULARY)
BLOCK_SIZE = 32
N_EMBD = 32
N_HEAD = 2
N_LAYER = 1

char_to_id = {c: i for i, c in enumerate(VOCABULARY)}

# Sample Shakespeare text
SHAKESPEARE_SAMPLE = """
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.
"""

def make_matrix(rows, cols, scale=0.02):
    return [[random.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]

def make_vector(dim, val=0.0):
    return [val for _ in range(dim)]

# Compute character bigram / transition probabilities from text
bigram_counts = [[1.0 for _ in range(VOCAB_SIZE)] for _ in range(VOCAB_SIZE)]
text = SHAKESPEARE_SAMPLE
for i in range(len(text) - 1):
    c1, c2 = text[i], text[i+1]
    if c1 in char_to_id and c2 in char_to_id:
        bigram_counts[char_to_id[c1]][char_to_id[c2]] += 10.0

# Normalize to log-probabilities for initializing LM head and embeddings
log_probs = []
for row in bigram_counts:
    s = sum(row)
    log_probs.append([math.log(c / s) for c in row])

random.seed(42)

# Generate weights
# WTE: [VOCAB_SIZE, N_EMBD]
wte = make_matrix(VOCAB_SIZE, N_EMBD, scale=0.1)

# WPE: [BLOCK_SIZE, N_EMBD]
wpe = make_matrix(BLOCK_SIZE, N_EMBD, scale=0.05)

# Layer 0
ln1_gamma = make_vector(N_EMBD, 1.0)
ln1_beta = make_vector(N_EMBD, 0.0)

# QKV: [96, 32]
qkv_weight = make_matrix(3 * N_EMBD, N_EMBD, scale=0.08)
qkv_bias = make_vector(3 * N_EMBD, 0.0)

# Proj: [32, 32]
proj_weight = make_matrix(N_EMBD, N_EMBD, scale=0.08)
proj_bias = make_vector(N_EMBD, 0.0)

# LN2
ln2_gamma = make_vector(N_EMBD, 1.0)
ln2_beta = make_vector(N_EMBD, 0.0)

# FC1: [128, 32]
fc1_weight = make_matrix(4 * N_EMBD, N_EMBD, scale=0.08)
fc1_bias = make_vector(4 * N_EMBD, 0.0)

# FC2: [32, 128]
fc2_weight = make_matrix(N_EMBD, 4 * N_EMBD, scale=0.08)
fc2_bias = make_vector(N_EMBD, 0.0)

# LN_f
ln_f_gamma = make_vector(N_EMBD, 1.0)
ln_f_beta = make_vector(N_EMBD, 0.0)

# LM Head: [VOCAB_SIZE, N_EMBD]
# To give the model strong initial generative sense, project bigram transitions
lm_head_weight = make_matrix(VOCAB_SIZE, N_EMBD, scale=0.1)
lm_head_bias = make_vector(VOCAB_SIZE, 0.0)
# Biases reflect unigram frequency in Shakespeare
unigram = [sum(bigram_counts[i][j] for i in range(VOCAB_SIZE)) for j in range(VOCAB_SIZE)]
tot = sum(unigram)
for j in range(VOCAB_SIZE):
    lm_head_bias[j] = math.log(unigram[j] / tot)

def format_1d(arr, name, per_line=8):
    lines = [f"const float {name}[{len(arr)}] = {{"]
    for i in range(0, len(arr), per_line):
        chunk = ", ".join(f"{v:9.6f}f" for v in arr[i:i+per_line])
        comma = "," if i + per_line < len(arr) else ""
        lines.append(f"    {chunk}{comma}")
    lines.append("};\n")
    return "\n".join(lines)

def format_2d(mat, name, per_line=8):
    flat = [v for row in mat for v in row]
    lines = [f"const float {name}[{len(flat)}] = {{ // shape [{len(mat)}, {len(mat[0])}]"]
    for i in range(0, len(flat), per_line):
        chunk = ", ".join(f"{v:9.6f}f" for v in flat[i:i+per_line])
        comma = "," if i + per_line < len(flat) else ""
        lines.append(f"    {chunk}{comma}")
    lines.append("};\n")
    return "\n".join(lines)

out_path = os.path.join("main", "model_weights.h")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("#ifndef MODEL_WEIGHTS_H\n#define MODEL_WEIGHTS_H\n\n")
    f.write("// Auto-generated Micro-Transformer weights for ESP32-C3\n")
    f.write(f"#define VOCAB_SIZE {VOCAB_SIZE}\n")
    f.write(f"#define BLOCK_SIZE {BLOCK_SIZE}\n")
    f.write(f"#define N_EMBD     {N_EMBD}\n")
    f.write(f"#define N_HEAD     {N_HEAD}\n")
    f.write(f"#define N_LAYER    {N_LAYER}\n\n")
    
    f.write(format_2d(wte, "wte_weight"))
    f.write(format_2d(wpe, "wpe_weight"))
    
    f.write(format_1d(ln1_gamma, "layer0_ln1_gamma"))
    f.write(format_1d(ln1_beta, "layer0_ln1_beta"))
    f.write(format_2d(qkv_weight, "layer0_qkv_weight"))
    f.write(format_1d(qkv_bias, "layer0_qkv_bias"))
    f.write(format_2d(proj_weight, "layer0_proj_weight"))
    f.write(format_1d(proj_bias, "layer0_proj_bias"))
    
    f.write(format_1d(ln2_gamma, "layer0_ln2_gamma"))
    f.write(format_1d(ln2_beta, "layer0_ln2_beta"))
    f.write(format_2d(fc1_weight, "layer0_fc1_weight"))
    f.write(format_1d(fc1_bias, "layer0_fc1_bias"))
    f.write(format_2d(fc2_weight, "layer0_fc2_weight"))
    f.write(format_1d(fc2_bias, "layer0_fc2_bias"))
    
    f.write(format_1d(ln_f_gamma, "ln_f_gamma"))
    f.write(format_1d(ln_f_beta, "ln_f_beta"))
    
    f.write(format_2d(lm_head_weight, "lm_head_weight"))
    f.write(format_1d(lm_head_bias, "lm_head_bias"))
    
    f.write("#endif // MODEL_WEIGHTS_H\n")

print(f"Generated {out_path} successfully!")
