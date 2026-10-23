#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include "esp_random.h"
#include "micro_gpt.h"

#define MAX_PROMPT_LEN BLOCK_SIZE
#define MAX_GEN_TOKENS 128
#define SAMPLING_TEMPERATURE 0.80f
#define SAMPLING_TOP_K 20

void app_main(void)
{
    micro_gpt_init();

    printf("\n");
    printf("========================================================\n");
    printf("     ESP32-S3 Edge AI: Micro-Transformer Online!       \n");
    printf("  Architecture: %d Heads | %d Embd | %d Context Window  \n", N_HEAD, N_EMBD, BLOCK_SIZE);
    printf("  Weights: Shakespeare (%d params, %d layers)\n", MODEL_PARAMETERS, N_LAYER);
    printf("========================================================\n");

    printf("[USB Serial Ready] Type a prompt and press Enter:\n\n> ");
    fflush(stdout);

    char prompt[MAX_PROMPT_LEN + 1];
    int token_history[MAX_PROMPT_LEN + MAX_GEN_TOKENS + 1];
    float logits[VOCAB_SIZE];
    int prompt_len = 0;

    while (1) {
        int character = getchar();

        if (character != EOF) {
            if (character == '\r' || character == '\n') {
                if (prompt_len > 0) {
                    prompt[prompt_len] = '\0';
                    printf("\n\n[1] Received prompt: \"%s\"\n", prompt);

                    // 1. Tokenize prompt
                    int num_tokens = 0;
                    int unsupported = 0;
                    printf("[2] Prompt Tokens: ");
                    for (int i = 0; i < prompt_len; i++) {
                        int id = micro_gpt_tokenize(prompt[i]);
                        if (id >= 0) {
                            token_history[num_tokens++] = id;
                            printf("%d ", id);
                        } else {
                            unsupported = 1;
                        }
                    }
                    printf("\n");

                    if (unsupported || num_tokens == 0) {
                        printf("Prompt contains unsupported characters; use the Shakespeare vocabulary.\n> ");
                        prompt_len = 0;
                        fflush(stdout);
                        continue;
                    }

                    // 2. Autoregressive text generation loop
                    printf("[3] AI Generating (T=%.1f): \n", SAMPLING_TEMPERATURE);
                    printf("----------------------------------------\n");
                    printf("%s", prompt);
                    fflush(stdout);

                    micro_gpt_init();
                    int64_t start_time = esp_timer_get_time();
                    int generated_count = 0;

                    for (int step = 0; step < MAX_GEN_TOKENS; step++) {
                        int ctx_len = num_tokens > BLOCK_SIZE ? BLOCK_SIZE : num_tokens;
                        const int *ctx_tokens = &token_history[num_tokens - ctx_len];

                        micro_gpt_forward(ctx_tokens, ctx_len, logits);

                        float random_unit = (esp_random() >> 8) * (1.0f / 16777216.0f);
                        int next_tok = micro_gpt_sample(logits, SAMPLING_TEMPERATURE, SAMPLING_TOP_K, random_unit);
                        char out_char = micro_gpt_detokenize(next_tok);
                        putchar(out_char);
                        fflush(stdout);

                        if (num_tokens < (int)(sizeof(token_history) / sizeof(token_history[0]) - 1)) {
                            token_history[num_tokens++] = next_tok;
                        } else {
                            memmove(&token_history[0], &token_history[1], (num_tokens - 1) * sizeof(int));
                            token_history[num_tokens - 1] = next_tok;
                        }
                        generated_count++;

                        vTaskDelay(1);
                    }

                    int64_t end_time = esp_timer_get_time();
                    float elapsed_ms = (float)(end_time - start_time) / 1000.0f;
                    float tok_per_sec = (float)generated_count / (elapsed_ms / 1000.0f);

                    printf("\n----------------------------------------\n");
                    printf("[Stats] %d tokens in %.1f ms (%.1f tokens/sec)\n\n",
                           generated_count, elapsed_ms, tok_per_sec);

                    prompt_len = 0;
                    printf("> ");
                    fflush(stdout);
                }
            } else if (character == '\b' || character == 127) {
                if (prompt_len > 0) {
                    prompt_len--;
                    printf("\b \b");
                    fflush(stdout);
                }
            } else if (prompt_len < MAX_PROMPT_LEN) {
                prompt[prompt_len++] = character;
                putchar(character);
                fflush(stdout);
            } else {
                printf("\n[Prompt full: press Enter to run]\n");
                fflush(stdout);
            }
        } else {
            vTaskDelay(pdMS_TO_TICKS(10));
        }
    }
}
