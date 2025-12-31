#include <mathflow/base/mf_utils.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>

// --- Hashing ---

u32 mf_fnv1a_hash(const char* str) {
    u32 hash = 2166136261u;
    while (*str) {
        hash ^= (u8)*str++;
        hash *= 16777619u;
    }
    return hash;
}

// --- String / Path Utils ---

char* mf_arena_strdup(mf_arena* arena, const char* str) {
    if (!str) return NULL;
    size_t len = strlen(str);
    char* copy = MF_ARENA_PUSH(arena, char, len + 1);
    strcpy(copy, str);
    return copy;
}

char* mf_arena_sprintf(mf_arena* arena, const char* fmt, ...) {
    va_list args;
    
    // 1. Calculate length
    va_start(args, fmt);
    int len = vsnprintf(NULL, 0, fmt, args);
    va_end(args);
    
    if (len < 0) return NULL;
    
    // 2. Allocate
    char* buf = MF_ARENA_PUSH(arena, char, len + 1);
    
    // 3. Print
    va_start(args, fmt);
    vsnprintf(buf, len + 1, fmt, args);
    va_end(args);
    
    return buf;
}

char* mf_path_get_dir(const char* path, mf_arena* arena) {
    // Find last slash
    const char* last_slash = strrchr(path, '/');
#ifdef _WIN32
    const char* last_bslash = strrchr(path, '\\');
    if (last_bslash > last_slash) last_slash = last_bslash;
#endif

    if (!last_slash) return mf_arena_strdup(arena, ".");

    size_t len = last_slash - path;
    char* dir = MF_ARENA_PUSH(arena, char, len + 1);
    memcpy(dir, path, len);
    dir[len] = '\0';
    return dir;
}

char* mf_path_join(const char* dir, const char* file, mf_arena* arena) {
    if (!dir || !file) return NULL;
    
    // If file is absolute, return file
    if (file[0] == '/' || file[0] == '\\' || (strlen(file) > 2 && file[1] == ':')) {
        return mf_arena_strdup(arena, file);
    }

    size_t len1 = strlen(dir);
    
    // Check if dir already has trailing slash
    bool slash = (len1 > 0 && (dir[len1-1] == '/' || dir[len1-1] == '\\'));
    
    if (slash) {
        return mf_arena_sprintf(arena, "%s%s", dir, file);
    } else {
        return mf_arena_sprintf(arena, "%s/%s", dir, file);
    }
}

char* mf_file_read(const char* path, mf_arena* arena) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    // Safety check
    if (len < 0) { fclose(f); return NULL; }
    
    char* buf = MF_ARENA_PUSH(arena, char, len + 1);
    if (fread(buf, 1, len, f) != (size_t)len) {
        // If read failed, strictly we can't rewind allocation in linear arena easily 
        // without save/restore, but for now just return NULL.
        fclose(f);
        return NULL;
    }
    buf[len] = 0;
    fclose(f);
    return buf;
}

size_t mf_utf8_to_utf32(const char* utf8, u32* out_buffer, size_t max_out) {
    size_t count = 0;
    const unsigned char* p = (const unsigned char*)utf8;
    while (*p) {
        u32 codepoint = 0;
        int len = 0;
        
        if ((*p & 0x80) == 0) {
            codepoint = *p;
            len = 1;
        } else if ((*p & 0xE0) == 0xC0) {
            codepoint = (*p & 0x1F) << 6;
            codepoint |= (p[1] & 0x3F);
            len = 2;
        } else if ((*p & 0xF0) == 0xE0) {
            codepoint = (*p & 0x0F) << 12;
            codepoint |= (p[1] & 0x3F) << 6;
            codepoint |= (p[2] & 0x3F);
            len = 3;
        } else if ((*p & 0xF8) == 0xF0) {
            codepoint = (*p & 0x07) << 18;
            codepoint |= (p[1] & 0x3F) << 12;
            codepoint |= (p[2] & 0x3F) << 6;
            codepoint |= (p[3] & 0x3F);
            len = 4;
        } else {
            p++; 
            continue;
        }
        
        if (out_buffer && count < max_out) {
            out_buffer[count] = codepoint;
        }
        count++;
        p += len;
    }
    return count;
}

// --- String Map ---

void mf_map_init(mf_str_map* map, size_t capacity, mf_arena* arena) {
    map->capacity = capacity;
    map->count = 0;
    map->entries = MF_ARENA_PUSH(arena, mf_map_entry, capacity);
    memset(map->entries, 0, sizeof(mf_map_entry) * capacity);
}

void mf_map_put(mf_str_map* map, const char* key, u32 value) {
    if (map->count >= map->capacity / 2) return; 

    u32 hash = mf_fnv1a_hash(key);
    size_t idx = hash % map->capacity;

    while (map->entries[idx].key != NULL) {
        if (strcmp(map->entries[idx].key, key) == 0) {
            map->entries[idx].value = value;
            return;
        }
        idx = (idx + 1) % map->capacity;
    }

    map->entries[idx].key = key;
    map->entries[idx].value = value;
    map->count++;
}

void mf_map_put_ptr(mf_str_map* map, const char* key, void* ptr) {
    if (map->count >= map->capacity / 2) return; 

    u32 hash = mf_fnv1a_hash(key);
    size_t idx = hash % map->capacity;

    while (map->entries[idx].key != NULL) {
        if (strcmp(map->entries[idx].key, key) == 0) {
            map->entries[idx].ptr_value = ptr;
            return;
        }
        idx = (idx + 1) % map->capacity;
    }

    map->entries[idx].key = key;
    map->entries[idx].ptr_value = ptr;
    map->count++;
}

bool mf_map_get(mf_str_map* map, const char* key, u32* out_val) {
    if (map->capacity == 0) return false;
    u32 hash = mf_fnv1a_hash(key);
    size_t idx = hash % map->capacity;

    while (map->entries[idx].key != NULL) {
        if (strcmp(map->entries[idx].key, key) == 0) {
            if (out_val) *out_val = map->entries[idx].value;
            return true;
        }
        idx = (idx + 1) % map->capacity;
    }
    return false;
}

bool mf_map_get_ptr(mf_str_map* map, const char* key, void** out_ptr) {
    if (map->capacity == 0) return false;
    u32 hash = mf_fnv1a_hash(key);
    size_t idx = hash % map->capacity;

    while (map->entries[idx].key != NULL) {
        if (strcmp(map->entries[idx].key, key) == 0) {
            if (out_ptr) *out_ptr = map->entries[idx].ptr_value;
            return true;
        }
        idx = (idx + 1) % map->capacity;
    }
    return false;
}
