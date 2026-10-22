#include "wifi_server.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "esp_wifi.h"
#include "esp_netif.h"
#include "esp_event.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "esp_http_server.h"
#include "esp_timer.h"
#include "micro_gpt.h"

static const char *TAG = "wifi_server";

#define WIFI_AP_SSID      "ESP32-Edge-AI"
#define WIFI_AP_PASS      "" // Open AP for seamless connection
#define WIFI_AP_CHANNEL   1
#define WIFI_AP_MAX_CONN  4

#define WEB_MAX_PROMPT_LEN 64
#define WEB_MAX_GEN_TOKENS 50
#define WEB_DEFAULT_TEMP   0.50f

static httpd_handle_t s_server = NULL;

static const char INDEX_HTML[] =
"<!DOCTYPE html>"
"<html lang='en'>"
"<head>"
"<meta charset='UTF-8'>"
"<meta name='viewport' content='width=device-width,initial-scale=1.0'>"
"<title>ESP32-C3 Edge AI</title>"
"<style>"
":root{--bg:#090d16;--card:#131c2e;--card-border:#1f2d47;--primary:#38bdf8;--text:#f1f5f9;--subtext:#94a3b8;--user-bubble:#0284c7;--ai-bubble:#1e293b;}"
"*{box-sizing:border-box;margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif}"
"body{background:var(--bg);color:var(--text);display:flex;flex-direction:column;height:100vh;overflow:hidden}"
"header{background:var(--card);border-bottom:1px solid var(--card-border);padding:14px 18px;display:flex;justify-content:space-between;align-items:center;box-shadow:0 4px 12px rgba(0,0,0,0.3)}"
".logo{display:flex;align-items:center;gap:10px}"
".logo-icon{width:32px;height:32px;background:linear-gradient(135deg,#38bdf8,#818cf8);border-radius:8px;display:flex;align-items:center;justify-content:center;font-weight:900;color:#fff;font-size:16px}"
".title{font-size:16px;font-weight:700;letter-spacing:-0.3px}"
".status{display:flex;align-items:center;gap:6px;font-size:12px;color:#4ade80;background:rgba(74,222,128,0.1);padding:4px 10px;border-radius:20px;border:1px solid rgba(74,222,128,0.2)}"
".status-dot{width:7px;height:7px;background:#4ade80;border-radius:50%;animation:pulse 2s infinite}"
"@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}"
".banner{background:linear-gradient(90deg,rgba(56,189,248,0.08),rgba(129,140,248,0.08));padding:8px 16px;font-size:11px;color:var(--subtext);text-align:center;border-bottom:1px solid var(--card-border)}"
"#chat{flex:1;overflow-y:auto;padding:18px;display:flex;flex-direction:column;gap:14px}"
".msg{display:flex;flex-direction:column;max-width:85%;animation:fadeIn 0.25s ease-out}"
"@keyframes fadeIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}"
".msg.user{align-self:flex-end;align-items:flex-end}"
".msg.ai{align-self:flex-start;align-items:flex-start}"
".bubble{padding:10px 14px;border-radius:14px;font-size:14px;line-height:1.5;white-space:pre-wrap;word-break:break-word}"
".user .bubble{background:var(--user-bubble);color:#fff;border-bottom-right-radius:2px}"
".ai .bubble{background:var(--ai-bubble);color:var(--text);border:1px solid var(--card-border);border-bottom-left-radius:2px}"
".stats{font-size:10px;color:var(--subtext);margin-top:4px;display:flex;gap:8px}"
".temp-bar{display:flex;align-items:center;gap:6px;padding:6px 14px;background:#0d1424;border-top:1px solid var(--card-border);font-size:11px;color:var(--subtext)}"
".t-btn{background:#1e293b;color:var(--subtext);border:1px solid var(--card-border);padding:3px 10px;border-radius:12px;font-size:11px;cursor:pointer;transition:all 0.15s}"
".t-btn.active{background:var(--primary);color:#090d16;font-weight:700;border-color:var(--primary)}"
".pill-box{display:flex;gap:6px;overflow-x:auto;padding:8px 14px;background:var(--card);border-top:1px solid var(--card-border)}"
".pill{background:#1e293b;color:var(--primary);border:1px solid var(--card-border);padding:5px 12px;border-radius:16px;font-size:12px;white-space:nowrap;cursor:pointer;transition:all 0.15s}"
".pill:hover{background:#38bdf8;color:#0f172a}"
"footer{background:var(--card);padding:12px 14px;display:flex;gap:8px;border-top:1px solid var(--card-border)}"
"#prompt-input{flex:1;background:#090d16;border:1px solid var(--card-border);color:var(--text);padding:12px 16px;border-radius:24px;font-size:14px;outline:none;transition:border 0.2s}"
"#prompt-input:focus{border-color:var(--primary)}"
"#send-btn{background:var(--primary);color:#090d16;border:none;border-radius:24px;padding:0 20px;font-weight:700;font-size:14px;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:opacity 0.2s}"
"#send-btn:disabled{opacity:0.4;cursor:not-allowed}"
"</style>"
"</head>"
"<body>"
"<header>"
"<div class='logo'>"
"<div class='logo-icon'>AI</div>"
"<div>"
"<div class='title'>ESP32-C3 Edge AI</div>"
"<div style='font-size:11px;color:var(--subtext);'>Micro-Transformer &bull; RISC-V 160MHz</div>"
"</div>"
"</div>"
"<div class='status'><div class='status-dot'></div>Live AP</div>"
"</header>"
"<div class='banner'>Offline On-Device Inference &bull; 18,017 Parameters &bull; 100% Local</div>"
"<div id='chat'>"
"<div class='msg ai'>"
"<div class='bubble'>Hello! I am a 1-Layer Transformer LLM running bare-metal on the ESP32-C3.\n\nType a prompt or tap a suggestion below!</div>"
"</div>"
"</div>"
"<div class='temp-bar'>"
"<span>Temp:</span>"
"<button class='t-btn' onclick='setTemp(0.35,this)'>0.35 Clean</button>"
"<button class='t-btn active' onclick='setTemp(0.50,this)'>0.50 Balanced</button>"
"<button class='t-btn' onclick='setTemp(0.75,this)'>0.75 Creative</button>"
"</div>"
"<div class='pill-box'>"
"<div class='pill' onclick='usePill(\"King:\")'>King:</div>"
"<div class='pill' onclick='usePill(\"ROMEO:\")'>ROMEO:</div>"
"<div class='pill' onclick='usePill(\"First Citizen:\")'>First Citizen:</div>"
"<div class='pill' onclick='usePill(\"To be or \")'>To be or</div>"
"<div class='pill' onclick='usePill(\"hello \")'>hello</div>"
"</div>"
"<footer>"
"<input type='text' id='prompt-input' placeholder='Type a prompt (e.g. King:)...' maxlength='64' onkeydown='if(event.key===\"Enter\")sendPrompt()'>"
"<button id='send-btn' onclick='sendPrompt()'>Send</button>"
"</footer>"
"<script>"
"const chat=document.getElementById('chat');"
"const input=document.getElementById('prompt-input');"
"const sendBtn=document.getElementById('send-btn');"
"let currentTemp=0.50;"
"function setTemp(v,btn){currentTemp=v;document.querySelectorAll('.t-btn').forEach(b=>b.classList.remove('active'));btn.classList.add('active');}"
"function usePill(t){input.value=t;sendPrompt();}"
"async function sendPrompt(){"
"const text=input.value.trim();"
"if(!text)return;"
"input.value='';"
"sendBtn.disabled=true;"
"const userMsg=document.createElement('div');"
"userMsg.className='msg user';"
"userMsg.innerHTML='<div class=\"bubble\">'+escapeHtml(text)+'</div>';"
"chat.appendChild(userMsg);"
"const aiMsg=document.createElement('div');"
"aiMsg.className='msg ai';"
"aiMsg.innerHTML='<div class=\"bubble\"><span class=\"stream-target\"></span></div><div class=\"stats\">Generating on RISC-V core (T='+currentTemp+')...</div>';"
"chat.appendChild(aiMsg);"
"chat.scrollTop=chat.scrollHeight;"
"const streamSpan=aiMsg.querySelector('.stream-target');"
"const statsDiv=aiMsg.querySelector('.stats');"
"const t0=performance.now();"
"let tokenCount=0;"
"try{"
"const res=await fetch('/api/generate?temp='+currentTemp+'&prompt='+encodeURIComponent(text));"
"const reader=res.body.getReader();"
"const dec=new TextDecoder();"
"while(true){"
"const{done,value}=await reader.read();"
"if(done)break;"
"const chunk=dec.decode(value,{stream:true});"
"streamSpan.textContent+=chunk;"
"tokenCount+=chunk.length;"
"chat.scrollTop=chat.scrollHeight;"
"}"
"const elapsedMs=(performance.now()-t0);"
"const tokPerSec=(tokenCount/(elapsedMs/1000)).toFixed(1);"
"statsDiv.innerHTML='⚡ '+tokenCount+' tokens in '+(elapsedMs/1000).toFixed(1)+'s ('+tokPerSec+' tok/s &bull; T='+currentTemp+')';"
"}catch(e){"
"statsDiv.textContent='Error generating response';"
"}"
"sendBtn.disabled=false;"
"chat.scrollTop=chat.scrollHeight;"
"}"
"function escapeHtml(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}"
"</script>"
"</body>"
"</html>";

static esp_err_t index_handler(httpd_req_t *req)
{
    httpd_resp_set_type(req, "text/html");
    httpd_resp_set_hdr(req, "Content-Encoding", "identity");
    return httpd_resp_send(req, INDEX_HTML, strlen(INDEX_HTML));
}

static void url_decode(char *dst, const char *src, size_t dst_size)
{
    size_t d = 0;
    while (*src && d + 1 < dst_size) {
        if (*src == '%' && src[1] && src[2]) {
            int val = 0;
            if (sscanf(src + 1, "%2x", &val) == 1) {
                dst[d++] = (char)val;
                src += 3;
                continue;
            }
        } else if (*src == '+') {
            dst[d++] = ' ';
            src++;
            continue;
        }
        dst[d++] = *src++;
    }
    dst[d] = '\0';
}

static esp_err_t generate_handler(httpd_req_t *req)
{
    char query[160] = {0};
    char raw_prompt[WEB_MAX_PROMPT_LEN * 3 + 1] = {0};
    char prompt[WEB_MAX_PROMPT_LEN + 1] = {0};
    float temperature = WEB_DEFAULT_TEMP;
    char temp_str[16] = {0};

    if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK) {
        httpd_query_key_value(query, "prompt", raw_prompt, sizeof(raw_prompt));
        url_decode(prompt, raw_prompt, sizeof(prompt));
        if (httpd_query_key_value(query, "temp", temp_str, sizeof(temp_str)) == ESP_OK) {
            float t = (float)atof(temp_str);
            if (t >= 0.05f && t <= 2.0f) {
                temperature = t;
            }
        }
    }

    if (strlen(prompt) == 0) {
        strncpy(prompt, "King:", sizeof(prompt) - 1);
    }

    ESP_LOGI(TAG, "Web request: prompt=\"%s\", temp=%.2f", prompt, temperature);

    httpd_resp_set_type(req, "text/plain; charset=utf-8");

    // Acquire Transformer lock for thread safety with UART
    micro_gpt_lock();

    int token_history[WEB_MAX_PROMPT_LEN + WEB_MAX_GEN_TOKENS + 1];
    int num_tokens = 0;

    for (size_t i = 0; i < strlen(prompt); i++) {
        int id = micro_gpt_tokenize(prompt[i]);
        if (id >= 0) {
            token_history[num_tokens++] = id;
        }
    }

    if (num_tokens == 0) {
        token_history[num_tokens++] = 0; // fallback newline
    }

    float logits[VOCAB_SIZE];

    for (int step = 0; step < WEB_MAX_GEN_TOKENS; step++) {
        int ctx_len = num_tokens > BLOCK_SIZE ? BLOCK_SIZE : num_tokens;
        const int *ctx_tokens = &token_history[num_tokens - ctx_len];

        micro_gpt_forward(ctx_tokens, ctx_len, logits);
        micro_gpt_apply_repetition_penalty(logits, ctx_tokens, ctx_len, 1.25f);

        int next_tok = micro_gpt_sample(logits, temperature);
        char out_char = micro_gpt_detokenize(next_tok);

        // Stream chunk directly to browser
        httpd_resp_send_chunk(req, &out_char, 1);

        if (num_tokens < (int)(sizeof(token_history) / sizeof(token_history[0]) - 1)) {
            token_history[num_tokens++] = next_tok;
        } else {
            memmove(&token_history[0], &token_history[1], (num_tokens - 1) * sizeof(int));
            token_history[num_tokens - 1] = next_tok;
        }

        // Yield to FreeRTOS so IDLE task and Wi-Fi network stack can run
        if ((step % 4) == 0) {
            vTaskDelay(1);
        }
    }

    micro_gpt_unlock();

    // Terminate chunked transfer
    httpd_resp_send_chunk(req, NULL, 0);
    return ESP_OK;
}

static const httpd_uri_t uri_index = {
    .uri       = "/",
    .method    = HTTP_GET,
    .handler   = index_handler,
    .user_ctx  = NULL
};

static const httpd_uri_t uri_generate = {
    .uri       = "/api/generate",
    .method    = HTTP_GET,
    .handler   = generate_handler,
    .user_ctx  = NULL
};

esp_err_t wifi_server_init(void)
{
    // 1. Initialize NVS (needed by Wi-Fi)
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    // 2. Initialize TCP/IP and Event Loop
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_ap();

    // 3. Configure Wi-Fi SoftAP
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    wifi_config_t wifi_config = {
        .ap = {
            .ssid = WIFI_AP_SSID,
            .ssid_len = strlen(WIFI_AP_SSID),
            .channel = WIFI_AP_CHANNEL,
            .password = WIFI_AP_PASS,
            .max_connection = WIFI_AP_MAX_CONN,
            .authmode = WIFI_AUTH_OPEN,
        },
    };

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "Wi-Fi SoftAP started! SSID: %s | URL: http://192.168.4.1", WIFI_AP_SSID);

    // 4. Start HTTP Web Server
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.max_uri_handlers = 8;
    config.stack_size = 8192; // 8 KB stack for web handler

    if (httpd_start(&s_server, &config) == ESP_OK) {
        httpd_register_uri_handler(s_server, &uri_index);
        httpd_register_uri_handler(s_server, &uri_generate);
        ESP_LOGI(TAG, "HTTP Web Server started successfully on port 80!");
        return ESP_OK;
    }

    ESP_LOGE(TAG, "Failed to start HTTP server!");
    return ESP_FAIL;
}
