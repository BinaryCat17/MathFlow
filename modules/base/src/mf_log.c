#include <mathflow/base/mf_log.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

#define MAX_SINKS 8
#define LOG_BUFFER_SIZE 2048

typedef struct {
    mf_log_sink_fn func;
    void* user_data;
    mf_log_level level;
} mf_log_sink_entry;

static struct {
    mf_log_sink_entry sinks[MAX_SINKS];
    int sink_count;
    mf_mutex_t mutex;
    bool initialized;
} g_logger = {0};

// Default global level (initially INFO until configured)
mf_log_level g_mf_log_global_level = MF_LOG_LEVEL_INFO;

// --- Default Console Sink ---

static const char* level_colors[] = {
    "\x1b[35m", // FATAL (Magenta)
    "\x1b[31m", // ERROR (Red)
    "\x1b[33m", // WARN  (Yellow)
    "\x1b[32m", // INFO  (Green)
    "\x1b[36m", // DEBUG (Cyan)
    "\x1b[90m"  // TRACE (Gray)
};

static const char* level_names[] = {
    "FATAL", "ERROR", "WARN ", "INFO ", "DEBUG", "TRACE"
};

static void console_sink(void* user_data, mf_log_level level, const char* file, int line, const char* message) {
    (void)user_data;
    
    // Get time
    time_t now = time(NULL);
    struct tm* t = localtime(&now);
    char time_buf[16];
    if (t) {
        strftime(time_buf, sizeof(time_buf), "%H:%M:%S", t);
    } else {
        strcpy(time_buf, "00:00:00");
    }

    const char* color = (level <= MF_LOG_LEVEL_TRACE) ? level_colors[level] : "\x1b[0m";
    const char* reset = "\x1b[0m";
    const char* name = (level <= MF_LOG_LEVEL_TRACE) ? level_names[level] : "UNK";

    // Format: [TIME] [LEVEL] Message (File:Line)
    // We use fprintf directly to stdout/stderr
    FILE* stream = (level <= MF_LOG_LEVEL_ERROR) ? stderr : stdout;
    
    fprintf(stream, "%s %s[%s]%s %s \x1b[90m(%s:%d)%s\n", 
            time_buf, color, name, reset, message, file, line, reset);
            
    // Flush to ensure we see logs immediately if crash happens
    fflush(stream);
}

// --- Implementation ---

static void update_global_level(void) {
    mf_log_level max = MF_LOG_LEVEL_FATAL;
    for (int i = 0; i < g_logger.sink_count; ++i) {
        if (g_logger.sinks[i].level > max) {
            max = g_logger.sinks[i].level;
        }
    }
    g_mf_log_global_level = max;
}

void mf_log_init(void) {
    if (g_logger.initialized) return; 
    
    mf_mutex_init(&g_logger.mutex);
    g_logger.sink_count = 0;
    g_logger.initialized = true;

    // Add default console sink
    mf_log_add_sink(console_sink, NULL, MF_LOG_LEVEL_INFO);
}

void mf_log_shutdown(void) {
    if (!g_logger.initialized) return;
    mf_mutex_destroy(&g_logger.mutex);
    g_logger.initialized = false;
}

void mf_log_add_sink(mf_log_sink_fn sink_fn, void* user_data, mf_log_level level) {
    if (!g_logger.initialized) mf_log_init();

    mf_mutex_lock(&g_logger.mutex);
    if (g_logger.sink_count < MAX_SINKS) {
        g_logger.sinks[g_logger.sink_count].func = sink_fn;
        g_logger.sinks[g_logger.sink_count].user_data = user_data;
        g_logger.sinks[g_logger.sink_count].level = level;
        g_logger.sink_count++;
        
        // Recalculate global level
        update_global_level();
    }
    mf_mutex_unlock(&g_logger.mutex);
}

void mf_log_set_global_level(mf_log_level level) {
    if (!g_logger.initialized) mf_log_init();
    
    mf_mutex_lock(&g_logger.mutex);
    for (int i = 0; i < g_logger.sink_count; ++i) {
        g_logger.sinks[i].level = level;
    }
    g_mf_log_global_level = level;
    mf_mutex_unlock(&g_logger.mutex);
}

void mf_log_message(mf_log_level level, const char* file, int line, const char* fmt, ...) {
    if (!g_logger.initialized) mf_log_init();

    // 1. Format the message into a stack buffer
    char buffer[LOG_BUFFER_SIZE];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);

    // 2. Dispatch to sinks (Thread-Safe)
    mf_mutex_lock(&g_logger.mutex);
    for (int i = 0; i < g_logger.sink_count; ++i) {
        if (level <= g_logger.sinks[i].level) {
            g_logger.sinks[i].func(g_logger.sinks[i].user_data, level, file, line, buffer);
        }
    }
    mf_mutex_unlock(&g_logger.mutex);
}
