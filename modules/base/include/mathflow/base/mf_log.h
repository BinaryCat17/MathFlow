#ifndef MF_LOG_H
#define MF_LOG_H

#include <mathflow/base/mf_platform.h>
#include <stdarg.h>

// --- Log Levels ---

typedef enum {
    MF_LOG_LEVEL_FATAL = 0,
    MF_LOG_LEVEL_ERROR = 1,
    MF_LOG_LEVEL_WARN  = 2,
    MF_LOG_LEVEL_INFO  = 3,
    MF_LOG_LEVEL_DEBUG = 4,
    MF_LOG_LEVEL_TRACE = 5
} mf_log_level;

// --- Sink Interface ---

/*
 * Callback function for log output.
 * user_data: Custom pointer passed during registration (e.g. FILE* or WindowHandle).
 * level: The severity of the message.
 * file/line: Source location.
 * message: The formatted log message (null-terminated).
 */
typedef void (*mf_log_sink_fn)(void* user_data, mf_log_level level, const char* file, int line, const char* message);

// --- Public API ---

/**
 * Initializes the logging system. 
 * Not strictly required if zero-initialization is sufficient, but good for explicit setup.
 */
void mf_log_init(void);

/**
 * Clean up resources (mutexes).
 */
void mf_log_shutdown(void);

/**
 * Register a new log sink.
 * sink_fn: The callback to invoke.
 * user_data: Context pointer passed to the callback.
 * level: The maximum log level this sink cares about.
 */
void mf_log_add_sink(mf_log_sink_fn sink_fn, void* user_data, mf_log_level level);

/**
 * Helper: Sets the global "gatekeeper" level manually (rarely needed, usually auto-managed).
 */
void mf_log_set_global_level(mf_log_level level);

/**
 * Internal function called by macros. Do not call directly.
 */
void mf_log_message(mf_log_level level, const char* file, int line, const char* fmt, ...);

// --- Internal Global State for Zero-Cost checks ---

extern mf_log_level g_mf_log_global_level;

// --- Macros ---

// The "do { ... } while(0)" idiom ensures the macro behaves like a single statement.
// We check the global level *before* evaluating arguments to avoid formatting overhead.

#define MF_LOG_FATAL(fmt, ...) do { \
    if (g_mf_log_global_level >= MF_LOG_LEVEL_FATAL) \
        mf_log_message(MF_LOG_LEVEL_FATAL, __FILE__, __LINE__, fmt, ##__VA_ARGS__); \
} while(0)

#define MF_LOG_ERROR(fmt, ...) do { \
    if (g_mf_log_global_level >= MF_LOG_LEVEL_ERROR) \
        mf_log_message(MF_LOG_LEVEL_ERROR, __FILE__, __LINE__, fmt, ##__VA_ARGS__); \
} while(0)

#define MF_LOG_WARN(fmt, ...) do { \
    if (g_mf_log_global_level >= MF_LOG_LEVEL_WARN) \
        mf_log_message(MF_LOG_LEVEL_WARN, __FILE__, __LINE__, fmt, ##__VA_ARGS__); \
} while(0)

#define MF_LOG_INFO(fmt, ...) do { \
    if (g_mf_log_global_level >= MF_LOG_LEVEL_INFO) \
        mf_log_message(MF_LOG_LEVEL_INFO, __FILE__, __LINE__, fmt, ##__VA_ARGS__); \
} while(0)

#define MF_LOG_DEBUG(fmt, ...) do { \
    if (g_mf_log_global_level >= MF_LOG_LEVEL_DEBUG) \
        mf_log_message(MF_LOG_LEVEL_DEBUG, __FILE__, __LINE__, fmt, ##__VA_ARGS__); \
} while(0)

#define MF_LOG_TRACE(fmt, ...) do { \
    if (g_mf_log_global_level >= MF_LOG_LEVEL_TRACE) \
        mf_log_message(MF_LOG_LEVEL_TRACE, __FILE__, __LINE__, fmt, ##__VA_ARGS__); \
} while(0)

#endif // MF_LOG_H
