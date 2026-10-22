#ifndef MICRO_GPT_H
#define MICRO_GPT_H

#include <stddef.h>
#include <stdint.h>

#define VOCAB_SIZE 65
#define BLOCK_SIZE 32
#define N_EMBD     32
#define N_HEAD     2
#define HEAD_DIM   (N_EMBD / N_HEAD)
#define N_LAYER    1

#ifdef __cplusplus
extern "C" {
#endif

// The 65-character vocabulary
extern const char VOCABULARY[];

// Initialize model buffers and mutex
void micro_gpt_init(void);

// Thread-safety mutex locks for shared UART/Web inference
void micro_gpt_lock(void);
void micro_gpt_unlock(void);

// Convert single character to token ID (returns -1 if unsupported)
int micro_gpt_tokenize(char c);

// Convert token ID back to character ('?' if out of bounds)
char micro_gpt_detokenize(int token_id);

// Forward pass: takes a prompt of length seq_len (<= BLOCK_SIZE),
// and computes next-token logits into logits_out[VOCAB_SIZE].
void micro_gpt_forward(const int *tokens, int seq_len, float *logits_out);

// Apply repetition penalty to logits based on recent token history (prevents loops like "the the the")
void micro_gpt_apply_repetition_penalty(float *logits, const int *tokens, int seq_len, float penalty);

// Sample next token ID from logits array using temperature
// temperature > 0.0f: stochastic sampling (higher = more random)
// temperature <= 0.01f: greedy (argmax)
int micro_gpt_sample(float *logits, float temperature);

#ifdef __cplusplus
}
#endif

#endif // MICRO_GPT_H
