#ifndef MF_UTILS_H
#define MF_UTILS_H

#include <mathflow/base/mf_memory.h>
#include <stdbool.h>

// --- Hashing ---

u32 mf_fnv1a_hash(const char* str);

// --- String / Path Utils ---

// Duplicates string into arena
char* mf_arena_strdup(mf_arena* arena, const char* str);

// Sprintf into arena
char* mf_arena_sprintf(mf_arena* arena, const char* fmt, ...);

// Extract directory from path (e.g. "a/b/c.json" -> "a/b")
char* mf_path_get_dir(const char* path, mf_arena* arena);

// Extract extension from path (e.g. "a.json" -> "json")
const char* mf_path_get_ext(const char* path);

// Join directory and file (handling separators)
char* mf_path_join(const char* dir, const char* file, mf_arena* arena);

// Parse provider string (e.g. "host.index.0") into builtin ID and axis
void mf_provider_parse(const char* provider, u16* out_builtin_id, u8* out_builtin_axis);

// Check if file exists
bool mf_file_exists(const char* path);

// Read entire file into arena memory (null-terminated)
char* mf_file_read(const char* path, mf_arena* arena);

// Read entire file as binary (no arena, caller must free)
void* mf_file_read_bin(const char* path, size_t* out_size);

// Decodes a UTF-8 string into UTF-32 codepoints.
// Returns the number of codepoints produced.
// If out_buffer is NULL, only calculates the count.
size_t mf_utf8_to_utf32(const char* utf8, u32* out_buffer, size_t max_out);

// --- String Map (Key -> U32) ---

typedef struct {
    const char* key;
    u32 value;
    void* ptr_value;
} mf_map_entry;

typedef struct {
    mf_map_entry* entries;
    size_t capacity;
    size_t count;
} mf_str_map;

void mf_map_init(mf_str_map* map, size_t capacity, mf_arena* arena);
void mf_map_put(mf_str_map* map, const char* key, u32 value);
void mf_map_put_ptr(mf_str_map* map, const char* key, void* ptr);
bool mf_map_get(mf_str_map* map, const char* key, u32* out_val);
bool mf_map_get_ptr(mf_str_map* map, const char* key, void** out_ptr);

#endif // MF_UTILS_H
