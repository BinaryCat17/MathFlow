#include <mathflow/base/mf_json.h>
#include <mathflow/base/mf_utils.h>
#include <mathflow/base/mf_log.h>
#include <string.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>

// --- Lexer ---

typedef enum {
    TOK_EOF,
    TOK_LBRACE, TOK_RBRACE,
    TOK_LBRACKET, TOK_RBRACKET,
    TOK_COLON, TOK_COMMA,
    TOK_STRING, TOK_NUMBER,
    TOK_TRUE, TOK_FALSE, TOK_NULL
} mf_token_type;

typedef struct {
    mf_token_type type;
    const char* start;
    size_t length;
    mf_json_loc loc;
} mf_token;

typedef struct {
    const char* source;
    const char* cursor;
    mf_json_loc loc;
} mf_lexer;

static void lex_init(mf_lexer* l, const char* source) {
    l->source = source;
    l->cursor = source;
    l->loc.line = 1;
    l->loc.column = 1;
}

static void skip_whitespace(mf_lexer* l) {
    while (*l->cursor) {
        char c = *l->cursor;
        if (isspace(c)) {
            if (c == '\n') {
                l->loc.line++;
                l->loc.column = 1;
            } else {
                l->loc.column++;
            }
            l->cursor++;
        } else if (c == '/' && l->cursor[1] == '/') {
            while (*l->cursor && *l->cursor != '\n') {
                l->cursor++;
                l->loc.column++;
            }
        } else {
            break;
        }
    }
}

static mf_token next_token(mf_lexer* l) {
    skip_whitespace(l);
    
    mf_token t;
    t.start = l->cursor;
    t.loc = l->loc;
    t.length = 1;

    char c = *l->cursor;
    if (!c) {
        t.type = TOK_EOF;
        return t;
    }

    l->cursor++;
    l->loc.column++;

    switch (c) {
        case '{': t.type = TOK_LBRACE; return t;
        case '}': t.type = TOK_RBRACE; return t;
        case '[': t.type = TOK_LBRACKET; return t;
        case ']': t.type = TOK_RBRACKET; return t;
        case ':': t.type = TOK_COLON; return t;
        case ',': t.type = TOK_COMMA; return t;
        case '"':
            t.type = TOK_STRING;
            t.start = l->cursor;
            while (*l->cursor && *l->cursor != '"') {
                if (*l->cursor == '\\' && l->cursor[1]) {
                    l->cursor += 2;
                    l->loc.column += 2;
                } else {
                    l->cursor++;
                    l->loc.column++;
                }
            }
            t.length = l->cursor - t.start;
            if (*l->cursor == '"') {
                l->cursor++;
                l->loc.column++;
            }
            return t;
    }

    if (isdigit(c) || c == '-') {
        t.type = TOK_NUMBER;
        t.start = l->cursor - 1;
        while (*l->cursor && (isdigit(*l->cursor) || *l->cursor == '.' || *l->cursor == 'e' || *l->cursor == 'E' || *l->cursor == '+' || *l->cursor == '-')) {
            l->cursor++;
            l->loc.column++;
        }
        t.length = l->cursor - t.start;
        return t;
    }

    if (isalpha(c)) {
        t.start = l->cursor - 1;
        while (*l->cursor && isalpha(*l->cursor)) {
            l->cursor++;
            l->loc.column++;
        }
        t.length = l->cursor - t.start;
        if (strncmp(t.start, "true", 4) == 0 && t.length == 4) t.type = TOK_TRUE;
        else if (strncmp(t.start, "false", 5) == 0 && t.length == 5) t.type = TOK_FALSE;
        else if (strncmp(t.start, "null", 4) == 0 && t.length == 4) t.type = TOK_NULL;
        else t.type = TOK_EOF; // Error
        return t;
    }

    t.type = TOK_EOF;
    return t;
}

// --- Parser ---

typedef struct {
    mf_lexer lexer;
    mf_token peek;
    mf_arena* arena;
} mf_parser;

static void advance(mf_parser* p) {
    p->peek = next_token(&p->lexer);
}

static mf_json_value* parse_value(mf_parser* p);

static mf_json_value* parse_object(mf_parser* p) {
    mf_json_value* v = MF_ARENA_PUSH(p->arena, mf_json_value, 1);
    v->type = MF_JSON_VAL_OBJECT;
    v->loc = p->peek.loc;
    
    advance(p); // {
    
    typedef struct Field {
        const char* key;
        mf_json_value* val;
        struct Field* next;
    } Field;
    
    Field* head = NULL;
    Field* tail = NULL;
    size_t count = 0;

    while (p->peek.type != TOK_RBRACE && p->peek.type != TOK_EOF) {
        if (p->peek.type != TOK_STRING) {
            MF_LOG_ERROR("Expected string key at %u:%u", p->peek.loc.line, p->peek.loc.column);
            return v;
        }
        
        char* key = mf_arena_alloc((mf_allocator*)p->arena, p->peek.length + 1);
        memcpy(key, p->peek.start, p->peek.length);
        key[p->peek.length] = '\0';
        advance(p);
        
        if (p->peek.type != TOK_COLON) {
            MF_LOG_ERROR("Expected ':' after key at %u:%u", p->peek.loc.line, p->peek.loc.column);
            return v;
        }
        advance(p);
        
        mf_json_value* val = parse_value(p);
        
        Field* f = MF_ARENA_PUSH(p->arena, Field, 1);
        f->key = key;
        f->val = val;
        f->next = NULL;
        
        if (tail) tail->next = f; else head = f;
        tail = f;
        count++;
        
        if (p->peek.type == TOK_COMMA) advance(p);
        else if (p->peek.type != TOK_RBRACE) {
            MF_LOG_ERROR("Expected ',' or '}' at %u:%u", p->peek.loc.line, p->peek.loc.column);
            break;
        }
    }
    
    if (p->peek.type == TOK_RBRACE) advance(p);
    
    v->as.object.count = count;
    v->as.object.keys = MF_ARENA_PUSH(p->arena, const char*, count);
    v->as.object.values = MF_ARENA_PUSH(p->arena, mf_json_value, count);
    
    size_t i = 0;
    for (Field* f = head; f; f = f->next) {
        v->as.object.keys[i] = f->key;
        v->as.object.values[i] = *f->val;
        i++;
    }
    
    return v;
}

static mf_json_value* parse_array(mf_parser* p) {
    mf_json_value* v = MF_ARENA_PUSH(p->arena, mf_json_value, 1);
    v->type = MF_JSON_VAL_ARRAY;
    v->loc = p->peek.loc;
    
    advance(p); // [
    
    typedef struct Item {
        mf_json_value* val;
        struct Item* next;
    } Item;
    
    Item* head = NULL;
    Item* tail = NULL;
    size_t count = 0;
    
    while (p->peek.type != TOK_RBRACKET && p->peek.type != TOK_EOF) {
        mf_json_value* val = parse_value(p);
        
        Item* it = MF_ARENA_PUSH(p->arena, Item, 1);
        it->val = val;
        it->next = NULL;
        
        if (tail) tail->next = it; else head = it;
        tail = it;
        count++;
        
        if (p->peek.type == TOK_COMMA) advance(p);
        else if (p->peek.type != TOK_RBRACKET) {
            MF_LOG_ERROR("Expected ',' or ']' at %u:%u", p->peek.loc.line, p->peek.loc.column);
            break;
        }
    }
    
    if (p->peek.type == TOK_RBRACKET) advance(p);
    
    v->as.array.count = count;
    v->as.array.items = MF_ARENA_PUSH(p->arena, mf_json_value, count);
    
    size_t i = 0;
    for (Item* it = head; it; it = it->next) {
        v->as.array.items[i++] = *it->val;
    }
    
    return v;
}

static mf_json_value* parse_value(mf_parser* p) {
    mf_json_value* v = NULL;
    switch (p->peek.type) {
        case TOK_LBRACE: return parse_object(p);
        case TOK_LBRACKET: return parse_array(p);
        case TOK_STRING:
            v = MF_ARENA_PUSH(p->arena, mf_json_value, 1);
            v->type = MF_JSON_VAL_STRING;
            v->loc = p->peek.loc;
            char* s = mf_arena_alloc((mf_allocator*)p->arena, p->peek.length + 1);
            memcpy(s, p->peek.start, p->peek.length);
            s[p->peek.length] = '\0';
            v->as.s = s;
            advance(p);
            return v;
        case TOK_NUMBER:
            v = MF_ARENA_PUSH(p->arena, mf_json_value, 1);
            v->type = MF_JSON_VAL_NUMBER;
            v->loc = p->peek.loc;
            char buf[64];
            size_t len = p->peek.length < 63 ? p->peek.length : 63;
            memcpy(buf, p->peek.start, len);
            buf[len] = '\0';
            v->as.n = atof(buf);
            advance(p);
            return v;
        case TOK_TRUE:
            v = MF_ARENA_PUSH(p->arena, mf_json_value, 1);
            v->type = MF_JSON_VAL_BOOL;
            v->loc = p->peek.loc;
            v->as.b = true;
            advance(p);
            return v;
        case TOK_FALSE:
            v = MF_ARENA_PUSH(p->arena, mf_json_value, 1);
            v->type = MF_JSON_VAL_BOOL;
            v->loc = p->peek.loc;
            v->as.b = false;
            advance(p);
            return v;
        case TOK_NULL:
            v = MF_ARENA_PUSH(p->arena, mf_json_value, 1);
            v->type = MF_JSON_VAL_NULL;
            v->loc = p->peek.loc;
            advance(p);
            return v;
        default:
            MF_LOG_ERROR("Unexpected token at %u:%u", p->peek.loc.line, p->peek.loc.column);
            advance(p);
            return NULL;
    }
}

// --- API Implementation ---

const mf_json_value* mf_json_get_field(const mf_json_value* obj, const char* key) {
    if (!obj || obj->type != MF_JSON_VAL_OBJECT) return NULL;
    for (size_t i = 0; i < obj->as.object.count; ++i) {
        if (strcmp(obj->as.object.keys[i], key) == 0) {
            return &obj->as.object.values[i];
        }
    }
    return NULL;
}

const char* mf_json_get_string(const mf_json_value* val) {
    if (val && val->type == MF_JSON_VAL_STRING) return val->as.s;
    return NULL;
}

mf_json_value* mf_json_parse(const char* json_str, mf_arena* arena) {
    mf_parser p;
    lex_init(&p.lexer, json_str);
    p.arena = arena;
    advance(&p);
    return parse_value(&p);
}

mf_ast_graph* mf_json_parse_graph(const char* json_str, mf_arena* arena) {
    mf_json_value* root_val = mf_json_parse(json_str, arena);
    if (!root_val || root_val->type != MF_JSON_VAL_OBJECT) return NULL;
    
    mf_ast_graph* graph = MF_ARENA_PUSH(arena, mf_ast_graph, 1);
    memset(graph, 0, sizeof(mf_ast_graph));
    
    const mf_json_value* nodes_val = mf_json_get_field(root_val, "nodes");
    if (nodes_val && nodes_val->type == MF_JSON_VAL_ARRAY) {
        graph->node_count = nodes_val->as.array.count;
        graph->nodes = MF_ARENA_PUSH(arena, mf_ast_node, graph->node_count);
        for (size_t i = 0; i < graph->node_count; ++i) {
            const mf_json_value* n_val = &nodes_val->as.array.items[i];
            mf_ast_node* node = &graph->nodes[i];
            node->loc = n_val->loc;
            
            const mf_json_value* id_val = mf_json_get_field(n_val, "id");
            node->id = (id_val && id_val->type == MF_JSON_VAL_STRING) ? id_val->as.s : "unknown";
            
            const mf_json_value* type_val = mf_json_get_field(n_val, "type");
            node->type = (type_val && type_val->type == MF_JSON_VAL_STRING) ? type_val->as.s : "unknown";
            
            const mf_json_value* data_val = mf_json_get_field(n_val, "data");
            node->data = (mf_json_value*)data_val;
        }
    }
    
    const mf_json_value* links_val = mf_json_get_field(root_val, "links");
    if (links_val && links_val->type == MF_JSON_VAL_ARRAY) {
        graph->link_count = links_val->as.array.count;
        graph->links = MF_ARENA_PUSH(arena, mf_ast_link, graph->link_count);
        for (size_t i = 0; i < graph->link_count; ++i) {
            const mf_json_value* l_val = &links_val->as.array.items[i];
            mf_ast_link* link = &graph->links[i];
            link->loc = l_val->loc;
            
            link->src = mf_json_get_string(mf_json_get_field(l_val, "src"));
            if (!link->src) link->src = "unknown";
            
            link->dst = mf_json_get_string(mf_json_get_field(l_val, "dst"));
            if (!link->dst) link->dst = "unknown";
            
            link->src_port = mf_json_get_string(mf_json_get_field(l_val, "src_port"));
            link->dst_port = mf_json_get_string(mf_json_get_field(l_val, "dst_port"));
        }
    }

    const mf_json_value* imports_val = mf_json_get_field(root_val, "imports");
    if (imports_val && imports_val->type == MF_JSON_VAL_ARRAY) {
        graph->import_count = imports_val->as.array.count;
        graph->imports = MF_ARENA_PUSH(arena, const char*, graph->import_count);
        for (size_t i = 0; i < graph->import_count; ++i) {
            const mf_json_value* imp = &imports_val->as.array.items[i];
            graph->imports[i] = (imp->type == MF_JSON_VAL_STRING) ? imp->as.s : "";
        }
    }
    
    return graph;
}
            