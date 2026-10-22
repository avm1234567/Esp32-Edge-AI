#include "micro_gpt.h"
#include "model_weights.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "esp_random.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

// 65-character Tiny Shakespeare vocabulary
const char VOCABULARY[] =
    "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

// Static activation buffers (allocated in BSS, consumes ~18 KB of RAM)
static float s_x[BLOCK_SIZE * N_EMBD];
static float s_x_ln1[BLOCK_SIZE * N_EMBD];
static float s_K[BLOCK_SIZE * N_EMBD];
static float s_V[BLOCK_SIZE * N_EMBD];
static float s_Q[N_EMBD];
static float s_attn_concat[N_EMBD];
static float s_attn_proj[N_EMBD];
static float s_scores[BLOCK_SIZE];
static float s_x_ln2[N_EMBD];
static float s_hidden[4 * N_EMBD];
static float s_mlp_out[N_EMBD];
static float s_x_final[N_EMBD];

static SemaphoreHandle_t s_gpt_mutex = NULL;

void micro_gpt_init(void)
{
    memset(s_x, 0, sizeof(s_x));
    if (s_gpt_mutex == NULL) {
        s_gpt_mutex = xSemaphoreCreateMutex();
    }
}

void micro_gpt_lock(void)
{
    if (s_gpt_mutex) {
        xSemaphoreTake(s_gpt_mutex, portMAX_DELAY);
    }
}

void micro_gpt_unlock(void)
{
    if (s_gpt_mutex) {
        xSemaphoreGive(s_gpt_mutex);
    }
}

int micro_gpt_tokenize(char c)
{
    for (int i = 0; i < VOCAB_SIZE; i++) {
        if (VOCABULARY[i] == c) {
            return i;
        }
    }
    return -1;
}

char micro_gpt_detokenize(int token_id)
{
    if (token_id >= 0 && token_id < VOCAB_SIZE) {
        return VOCABULARY[token_id];
    }
    return '?';
}

static inline float gelu(float x)
{
    return 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
}

static void layernorm(float *out, const float *in, const float *gamma, const float *beta, int dim)
{
    float mean = 0.0f;
    for (int i = 0; i < dim; i++) {
        mean += in[i];
    }
    mean /= (float)dim;

    float var = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = in[i] - mean;
        var += diff * diff;
    }
    var /= (float)dim;

    float inv_std = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < dim; i++) {
        out[i] = (in[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

// Matrix-vector multiplication: out = bias + weight * in
// weight shape: [out_dim, in_dim]
static void matvec(float *out, const float *in, const float *weight, const float *bias, int in_dim, int out_dim)
{
    for (int j = 0; j < out_dim; j++) {
        float acc = (bias != NULL) ? bias[j] : 0.0f;
        const float *w_row = &weight[j * in_dim];
        for (int i = 0; i < in_dim; i++) {
            acc += in[i] * w_row[i];
        }
        out[j] = acc;
    }
}

void micro_gpt_forward(const int *tokens, int seq_len, float *logits_out)
{
    if (seq_len <= 0) return;
    if (seq_len > BLOCK_SIZE) seq_len = BLOCK_SIZE;

    int last_idx = seq_len - 1;

    // 1. Embeddings: wte (token) + wpe (position) for all tokens in context
    for (int t = 0; t < seq_len; t++) {
        int tok = tokens[t];
        if (tok < 0 || tok >= VOCAB_SIZE) tok = 0;
        
        const float *tok_emb = &wte_weight[tok * N_EMBD];
        const float *pos_emb = &wpe_weight[t * N_EMBD];
        float *x_t = &s_x[t * N_EMBD];

        for (int i = 0; i < N_EMBD; i++) {
            x_t[i] = tok_emb[i] + pos_emb[i];
        }
    }

    // 2. Transformer Layer 0
    // 2a. Pre-LayerNorm on all tokens in context
    for (int t = 0; t < seq_len; t++) {
        layernorm(&s_x_ln1[t * N_EMBD], &s_x[t * N_EMBD], layer0_ln1_gamma, layer0_ln1_beta, N_EMBD);
    }

    // 2b. Compute K and V for all t in [0, seq_len - 1]
    // qkv_weight layout: Q = rows [0..31], K = rows [32..63], V = rows [64..95]
    for (int t = 0; t < seq_len; t++) {
        const float *in_t = &s_x_ln1[t * N_EMBD];
        float *k_t = &s_K[t * N_EMBD];
        float *v_t = &s_V[t * N_EMBD];

        // K projection
        matvec(k_t, in_t, &layer0_qkv_weight[N_EMBD * N_EMBD], &layer0_qkv_bias[N_EMBD], N_EMBD, N_EMBD);
        // V projection
        matvec(v_t, in_t, &layer0_qkv_weight[2 * N_EMBD * N_EMBD], &layer0_qkv_bias[2 * N_EMBD], N_EMBD, N_EMBD);
    }

    // Compute Q for the last token only (position last_idx)
    matvec(s_Q, &s_x_ln1[last_idx * N_EMBD], layer0_qkv_weight, layer0_qkv_bias, N_EMBD, N_EMBD);

    // 2c. Multi-Head Attention for last token
    float scale = 1.0f / sqrtf((float)HEAD_DIM);

    for (int h = 0; h < N_HEAD; h++) {
        int h_offset = h * HEAD_DIM;
        const float *q_h = &s_Q[h_offset];

        // Dot-product Q with past K
        float max_score = -1e9f;
        for (int t = 0; t < seq_len; t++) {
            const float *k_h = &s_K[t * N_EMBD + h_offset];
            float dot = 0.0f;
            for (int d = 0; d < HEAD_DIM; d++) {
                dot += q_h[d] * k_h[d];
            }
            float sc = dot * scale;
            s_scores[t] = sc;
            if (sc > max_score) max_score = sc;
        }

        // Softmax
        float sum_exp = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            s_scores[t] = expf(s_scores[t] - max_score);
            sum_exp += s_scores[t];
        }
        float inv_sum = 1.0f / (sum_exp + 1e-8f);
        for (int t = 0; t < seq_len; t++) {
            s_scores[t] *= inv_sum;
        }

        // Weighted sum of V
        float *out_h = &s_attn_concat[h_offset];
        for (int d = 0; d < HEAD_DIM; d++) {
            float v_acc = 0.0f;
            for (int t = 0; t < seq_len; t++) {
                v_acc += s_scores[t] * s_V[t * N_EMBD + h_offset + d];
            }
            out_h[d] = v_acc;
        }
    }

    // 2d. Output projection
    matvec(s_attn_proj, s_attn_concat, layer0_proj_weight, layer0_proj_bias, N_EMBD, N_EMBD);

    // Residual add to last position of x
    float *x_last = &s_x[last_idx * N_EMBD];
    for (int i = 0; i < N_EMBD; i++) {
        x_last[i] += s_attn_proj[i];
    }

    // 2e. Pre-LayerNorm 2
    layernorm(s_x_ln2, x_last, layer0_ln2_gamma, layer0_ln2_beta, N_EMBD);

    // 2f. Feed-Forward MLP: fc1 (32 -> 128) + GELU
    matvec(s_hidden, s_x_ln2, layer0_fc1_weight, layer0_fc1_bias, N_EMBD, 4 * N_EMBD);
    for (int i = 0; i < 4 * N_EMBD; i++) {
        s_hidden[i] = gelu(s_hidden[i]);
    }

    // fc2 (128 -> 32)
    matvec(s_mlp_out, s_hidden, layer0_fc2_weight, layer0_fc2_bias, 4 * N_EMBD, N_EMBD);

    // Residual add
    for (int i = 0; i < N_EMBD; i++) {
        x_last[i] += s_mlp_out[i];
    }

    // 3. Final LayerNorm
    layernorm(s_x_final, x_last, ln_f_gamma, ln_f_beta, N_EMBD);

    // 4. LM Head: project to VOCAB_SIZE logits
    matvec(logits_out, s_x_final, lm_head_weight, lm_head_bias, N_EMBD, VOCAB_SIZE);
}

void micro_gpt_apply_repetition_penalty(float *logits, const int *tokens, int seq_len, float penalty)
{
    if (penalty <= 1.0f || seq_len <= 0) return;

    // Look back up to the last 16 tokens in context
    int start = (seq_len > 16) ? (seq_len - 16) : 0;
    for (int i = start; i < seq_len; i++) {
        int tok = tokens[i];
        if (tok >= 0 && tok < VOCAB_SIZE) {
            // Keskar et al. repetition penalty formula
            if (logits[tok] > 0.0f) {
                logits[tok] /= penalty;
            } else {
                logits[tok] *= penalty;
            }
        }
    }
}

int micro_gpt_sample(float *logits, float temperature)
{
    // Greedy argmax if temperature is tiny
    if (temperature <= 0.01f) {
        int best_id = 0;
        float best_val = logits[0];
        for (int i = 1; i < VOCAB_SIZE; i++) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best_id = i;
            }
        }
        return best_id;
    }

    // Temperature scaling & max subtraction for stability
    float max_logit = -1e9f;
    for (int i = 0; i < VOCAB_SIZE; i++) {
        logits[i] /= temperature;
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < VOCAB_SIZE; i++) {
        logits[i] = expf(logits[i] - max_logit);
        sum_exp += logits[i];
    }

    // ESP32 hardware true random float in [0, 1)
    uint32_t rand_val = esp_random();
    float r = (float)rand_val / 4294967296.0f;

    float cdf = 0.0f;
    for (int i = 0; i < VOCAB_SIZE; i++) {
        cdf += logits[i] / sum_exp;
        if (r <= cdf) {
            return i;
        }
    }

    return VOCAB_SIZE - 1;
}
