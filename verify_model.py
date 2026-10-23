"""Check the actual firmware math on the host against the trained PyTorch model."""
import argparse
import ctypes
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import torch
from train_micro_gpt import MicroGPT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='artifacts/best.pt')
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parent
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = MicroGPT(**ckpt['config']).eval()
    model.load_state_dict(ckpt['state'])
    rng = np.random.default_rng(12)
    worst = 0.0
    checks = 0
    with tempfile.TemporaryDirectory() as tmp:
        library = Path(tmp)/'micro_gpt.so'
        subprocess.run(['cc', '-O2', '-std=c99', '-Wall', '-Wextra', '-Werror', '-shared', '-fPIC',
            '-I'+str(root/'main'), str(root/'main/micro_gpt.c'), '-lm', '-o', str(library)], check=True)
        c = ctypes.CDLL(str(library))
        iptr = ctypes.POINTER(ctypes.c_int)
        fptr = ctypes.POINTER(ctypes.c_float)
        c.micro_gpt_forward.argtypes = [iptr, ctypes.c_int, fptr]
        c.micro_gpt_sample.argtypes = [fptr, ctypes.c_float, ctypes.c_int, ctypes.c_float]
        c.micro_gpt_sample.restype = ctypes.c_int
        c.micro_gpt_tokenize.argtypes = [ctypes.c_char]
        c.micro_gpt_detokenize.argtypes = [ctypes.c_int]
        c.micro_gpt_detokenize.restype = ctypes.c_char
        from train_micro_gpt import VOCABULARY
        for i, char in enumerate(VOCABULARY):
            assert c.micro_gpt_tokenize(char.encode()) == i
            assert c.micro_gpt_detokenize(i).decode() == char
        assert c.micro_gpt_tokenize(b'@') == -1
        def check(ids):
            nonlocal worst, checks
            ids = np.array(ids, dtype=np.int32)
            got = np.empty(65, dtype=np.float32)
            c.micro_gpt_forward(ids.ctypes.data_as(iptr), len(ids), got.ctypes.data_as(fptr))
            with torch.no_grad(): expected = model(torch.from_numpy(ids.astype(np.int64))[None])[0,-1].numpy()
            worst = max(worst, float(np.max(np.abs(got-expected))))
            np.testing.assert_allclose(got, expected, atol=2e-4, rtol=2e-4)
            checks += 1
        c.micro_gpt_init()
        ids = rng.integers(0,65,160).tolist()
        # Incremental appends, including every position and the sliding-window boundary.
        for end in range(1,len(ids)+1): check(ids[max(0,end-64):end])
        # Repeated prompts, changed suffixes, shorter prompts, and resets.
        for length in [1,2,17,32,63,64]:
            seq = rng.integers(0,65,length).tolist()
            check(seq); check(seq)
            seq[-1] = (seq[-1]+1)%65; check(seq)
            c.micro_gpt_init(); check(seq)
        for end in range(1,67): check([13]*min(end,64))
        logits = np.linspace(-4,4,65,dtype=np.float32)
        ptr = logits.ctypes.data_as(fptr)
        assert c.micro_gpt_sample(ptr,0,20,.9) == 64
        assert c.micro_gpt_sample(ptr,.8,1,.999999) == 64
        for uniform in np.linspace(0, np.nextafter(np.float32(1),np.float32(0)),1000):
            got = c.micro_gpt_sample(ptr,.8,20,float(uniform))
            assert 45 <= got <= 64
        np.testing.assert_array_equal(logits, np.linspace(-4,4,65,dtype=np.float32))
        # Check the actual categorical distribution against independently computed CDF.
        order = np.argsort(-logits)[:20]
        probabilities = np.exp((logits[order]-logits[order[0]])/.8)
        cdf = np.cumsum(probabilities/probabilities.sum())
        for uniform in [.001,.05,.25,.5,.75,.99]:
            assert c.micro_gpt_sample(ptr,.8,20,uniform) == int(order[np.searchsorted(cdf,uniform,side='right')])
    report = dict(forward_checks=checks, max_absolute_logit_error=worst,
        tolerance='atol=2e-4, rtol=2e-4', tokenizer='all 65 IDs round-trip',
        sampling='greedy, top-k bounds, categorical CDF, input unchanged',
        scope='Actual portable C firmware math on host; no hardware execution')
    (root/'verification_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__ == '__main__': main()
