idf_component_register(
    SRCS "main.c" "micro_gpt.c" "wifi_server.c"
    INCLUDE_DIRS "."
    PRIV_REQUIRES freertos esp_hw_support esp_timer esp_wifi esp_netif esp_event nvs_flash esp_http_server log
)
