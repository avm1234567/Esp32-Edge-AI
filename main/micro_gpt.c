#include "micro_gpt.h"
#include "model_weights.h"
#include <math.h>
#include <string.h>

const char VOCABULARY[] = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

/* Each layer keeps keys and values for the current absolute-position window. */
static float keys[N_LAYER][BLOCK_SIZE][N_EMBD];
static float values[N_LAYER][BLOCK_SIZE][N_EMBD];
static int cached_tokens[BLOCK_SIZE];
static int cached_count;
static float x[N_EMBD], norm[N_EMBD], qkv[3 * N_EMBD];
static float attention[N_EMBD], projected[N_EMBD], hidden[4 * N_EMBD];
static float scores[BLOCK_SIZE];

void micro_gpt_init(void) { cached_count = 0; }

int micro_gpt_tokenize(char c)
{
    for (int i = 0; i < VOCAB_SIZE; i++) if (VOCABULARY[i] == c) return i;
    return -1;
}

char micro_gpt_detokenize(int token)
{
    return token >= 0 && token < VOCAB_SIZE ? VOCABULARY[token] : '?';
}

static void matvec(float *out, const float *in, const float *weight,
                   const float *bias, int inputs, int outputs)
{
    for (int row = 0; row < outputs; row++) {
        float sum = bias[row];
        for (int col = 0; col < inputs; col++) sum += weight[row * inputs + col] * in[col];
        out[row] = sum;
    }
}

static void layernorm(float *out, const float *in, const float *gamma, const float *beta)
{
    float mean = 0, variance = 0;
    for (int i = 0; i < N_EMBD; i++) mean += in[i];
    mean /= N_EMBD;
    for (int i = 0; i < N_EMBD; i++) { float d = in[i] - mean; variance += d*d; }
    float scale = 1.0f / sqrtf(variance / N_EMBD + 1e-5f);
    for (int i = 0; i < N_EMBD; i++) out[i] = (in[i] - mean) * scale * gamma[i] + beta[i];
}

void micro_gpt_forward(const int *tokens, int count, float *logits)
{
    /* Caller supplies 1..BLOCK_SIZE valid token IDs. Cache only identical prefixes.
       A shifted window changes learned position embeddings, so it is rebuilt. */
    int start = 0;
    while (start < count && start < cached_count && tokens[start] == cached_tokens[start]) start++;
    if (start == count) start--; /* Recreate the final hidden state for repeat calls. */
    for (int position = start; position < count; position++) {
        for (int d = 0; d < N_EMBD; d++)
            x[d] = wte_weight[tokens[position] * N_EMBD + d] + wpe_weight[position * N_EMBD + d];
        for (int layer = 0; layer < N_LAYER; layer++) {
            const int vector = layer * N_EMBD;
            const int matrix = layer * N_EMBD * N_EMBD;
            layernorm(norm, x, layer_ln1_gamma + vector, layer_ln1_beta + vector);
            matvec(qkv, norm, layer_qkv_weight + 3 * matrix, layer_qkv_bias + 3 * vector, N_EMBD, 3*N_EMBD);
            memcpy(keys[layer][position], qkv + N_EMBD, sizeof(float)*N_EMBD);
            memcpy(values[layer][position], qkv + 2*N_EMBD, sizeof(float)*N_EMBD);
            for (int head = 0; head < N_HEAD; head++) {
                int offset = head * HEAD_DIM;
                float maximum = -INFINITY;
                for (int t = 0; t <= position; t++) {
                    float dot = 0;
                    for (int d = 0; d < HEAD_DIM; d++) dot += qkv[offset+d] * keys[layer][t][offset+d];
                    scores[t] = dot / sqrtf((float)HEAD_DIM);
                    if (scores[t] > maximum) maximum = scores[t];
                }
                float sum = 0;
                for (int t = 0; t <= position; t++) { scores[t] = expf(scores[t]-maximum); sum += scores[t]; }
                for (int d = 0; d < HEAD_DIM; d++) {
                    float weighted = 0;
                    for (int t = 0; t <= position; t++) weighted += scores[t] * values[layer][t][offset+d];
                    attention[offset+d] = weighted / sum;
                }
            }
            matvec(projected, attention, layer_proj_weight + matrix, layer_proj_bias + vector, N_EMBD, N_EMBD);
            for (int d = 0; d < N_EMBD; d++) x[d] += projected[d];
            layernorm(norm, x, layer_ln2_gamma + vector, layer_ln2_beta + vector);
            matvec(hidden, norm, layer_fc1_weight + 4*matrix, layer_fc1_bias + 4*vector, N_EMBD, 4*N_EMBD);
            for (int d = 0; d < 4*N_EMBD; d++) {
                float a = hidden[d];
                hidden[d] = 0.5f*a*(1.0f+tanhf(0.7978845608f*(a+0.044715f*a*a*a)));
            }
            matvec(projected, hidden, layer_fc2_weight + 4*matrix, layer_fc2_bias + vector, 4*N_EMBD, N_EMBD);
            for (int d = 0; d < N_EMBD; d++) x[d] += projected[d];
        }
        cached_tokens[position] = tokens[position];
    }
    cached_count = count;
    layernorm(norm, x, ln_f_gamma, ln_f_beta);
    matvec(logits, norm, lm_head_weight, lm_head_bias, N_EMBD, VOCAB_SIZE);
}

int micro_gpt_sample(const float *logits, float temperature, int top_k, float random_unit)
{
    int order[VOCAB_SIZE];
    for (int i = 0; i < VOCAB_SIZE; i++) {
        int j = i;
        while (j > 0 && logits[order[j-1]] < logits[i]) { order[j] = order[j-1]; j--; }
        order[j] = i;
    }
    if (temperature <= 0.01f) return order[0];
    if (top_k <= 0 || top_k > VOCAB_SIZE) top_k = VOCAB_SIZE;
    float probabilities[VOCAB_SIZE], total = 0;
    for (int i = 0; i < top_k; i++) {
        probabilities[i] = expf((logits[order[i]]-logits[order[0]])/temperature);
        total += probabilities[i];
    }
    float threshold = random_unit * total, cumulative = 0;
    for (int i = 0; i < top_k; i++) {
        cumulative += probabilities[i];
        if (threshold < cumulative) return order[i];
    }
    return order[top_k-1];
}
