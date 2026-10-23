# ESP32-S3 Edge AI

A character-level transformer runs entirely on one ESP32-S3. The host sends
prompts and displays generated characters over USB serial. No Wi-Fi, web server,
cloud inference, or PSRAM is required by this implementation.

## Build and use

ESP-IDF 5.5.1, target `esp32s3`. The default configuration uses a 240 MHz CPU,
performance compiler optimization, and UART0 at 115200 baud. The connected board's
USB-to-UART bridge exposes that UART to the host; this is not native USB CDC.

```sh
get_idf
idf.py build
# Run these only when ready to replace the connected board's firmware:
idf.py -p /dev/cu.usbmodem5CE51996871 flash
idf.py -p /dev/cu.usbmodem5CE51996871 monitor
```

Use your board's actual port. Type up to 64 characters and press Enter. The model
streams 128 new characters and reports elapsed time and characters per second.
Backspace edits input; Ctrl+] exits the ESP-IDF monitor. Prompts containing
characters outside the fixed Shakespeare vocabulary are rejected rather than
silently changing the input.

This is a Shakespeare-style text continuation model, not an instruction-following
assistant. Try `ROMEO:`, `First Citizen:`, or `To be or `.

## Upgraded model

- Two pre-normalized transformer layers, width 64, two attention heads.
- 64-character context, 65-character vocabulary, 112,577 FP32 parameters.
- Flash-resident weights; 64 KiB of static K/V cache plus working buffers.
- Temperature 0.8, top-k 20; no repeated character penalty.
- Cached inference reuses identical prefixes. Once the context slides, learned
  absolute position embeddings change, so the window is rebuilt. Sustained
  full-window throughput will be slower than cached appends before the window fills.
- `model_config.h` is generated with the weights, keeping C dimensions synchronized.

Sampling settings are in `main/main.c`. Changing architecture requires retraining
and exporting, not just changing a C constant.

## Training and verification

Create a Python 3.12 environment and install `requirements.txt`. The training run
uses four CPU threads; it also accepts `--device mps` or `--device cuda` when available.

The old weights are retained locally under ignored `artifacts/`. To recreate the
baseline after cloning this repository:

```sh
mkdir -p artifacts
git show abf9098:main/model_weights.h > artifacts/baseline_weights.h
```

```sh
.venv/bin/python train_micro_gpt.py --steps 20000 --device cpu
.venv/bin/python verify_model.py
```

Training uses contiguous 90/5/5 percent train/validation/test splits, seed 42,
AdamW, learning-rate warmup and cosine decay, gradient clipping, and dropout.
Validation loss selects the checkpoint; fixed test positions compare it with the
old model. The old model's full training history is unavailable, so the comparison
is an operational baseline rather than a controlled architecture-only experiment.

Outputs:

- `artifacts/best.pt`: selected trained checkpoint (local, ignored by Git).
- `main/model_weights.h`, `main/model_config.h`: deployable weights and dimensions.
- `training_report.json`: data hash, settings, measurements, and unedited samples.
- `verification_report.json`: actual C versus PyTorch numerical checks.

`verify_model.py` compiles the firmware math as a host library and checks cached
appends, repeated and changed prompts, window rollover, tokenizer round trips,
and sampling. It does not flash or run a board. Hardware generation speed must
be measured separately.

To re-export the existing trained checkpoint without retraining:

```sh
.venv/bin/python generate_weights.py --checkpoint artifacts/best.pt
```

The exporter no longer replaces trained weights with random demo weights.
`01_characters.py` remains a separate introductory vocabulary exercise.
