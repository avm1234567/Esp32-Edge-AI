#ifndef MICRO_GPT_H
#define MICRO_GPT_H
#include "model_config.h"
#define HEAD_DIM (N_EMBD / N_HEAD)
extern const char VOCABULARY[];
void micro_gpt_init(void);
int micro_gpt_tokenize(char c);
char micro_gpt_detokenize(int token);
/* Input: 1..BLOCK_SIZE valid IDs. Output: VOCAB_SIZE next-character scores.
   Not reentrant: static buffers belong to the serial inference task. */
void micro_gpt_forward(const int *tokens, int count, float *logits);
/* random_unit must be in [0,1); top_k<=0 includes the whole vocabulary. */
int micro_gpt_sample(const float *logits, float temperature, int top_k, float random_unit);
#endif
