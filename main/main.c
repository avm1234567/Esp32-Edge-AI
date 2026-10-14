#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

void app_main(void)
{
    const char vocabulary[] =
        "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    char prompt[65];
    int tokens[64];
    int length = 0;

    printf("ESP32-S3 ready. Enter a prompt (up to 64 characters):\n");

    while (1) {
        int character = getchar();

        if (character != EOF) {
            if (character == '\r' || character == '\n') {
                if (length > 0) {
                    prompt[length] = '\0';
                    printf("\nReceived prompt: %s\n", prompt);

                    printf("Token IDs (-1 = unsupported): ");
                    for (int i = 0; i < length; i++) {
                        tokens[i] = -1;
                        for (int j = 0; vocabulary[j] != '\0'; j++) {
                            if (prompt[i] == vocabulary[j]) {
                                tokens[i] = j;
                                break;
                            }
                        }
                        printf("%d ", tokens[i]);
                    }
                    printf("\n");
                    length = 0;
                }
            } else if (length < 64) {
                prompt[length] = character;
                length++;
                putchar(character);
            } else {
                printf("\nPrompt full: press Enter to submit.\n");
            }
            fflush(stdout);
        } else {
            vTaskDelay(pdMS_TO_TICKS(10));
        }
    }
}
