idf_component_register(
    SRCS "main.c" "micro_gpt.c"
    INCLUDE_DIRS "."
    PRIV_REQUIRES freertos esp_hw_support esp_timer
)
