#ifndef WIFI_SERVER_H
#define WIFI_SERVER_H

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initialize NVS, Wi-Fi SoftAP ("ESP32-Edge-AI"), and starts the Web Server on 192.168.4.1
esp_err_t wifi_server_init(void);

#ifdef __cplusplus
}
#endif

#endif // WIFI_SERVER_H
