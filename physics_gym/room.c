/* room.c: read room.json (written by room.py) into the physics' room table (body.c room_set).
 * A small reader for that file's shape only: the "boxes" array of objects with numeric arrays c, h, q and
 * numbers mu, soft; other keys are ignored. strtod reads Python's round-trip decimals exactly. */
#include "body.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *field(const char *o, const char *end, const char *key) {   /* the value after "key": inside [o, end) */
    char pat[32]; snprintf(pat, sizeof pat, "\"%s\"", key);
    for (const char *p = strstr(o, pat); p && p < end; p = strstr(p + 1, pat)) { p += strlen(pat); while (*p == ' ' || *p == ':') p++; return p; }
    return NULL;
}
static int nums(const char *p, double *v, int n) {
    if (!p) return 0;
    if (*p == '[') p++;
    for (int i = 0; i < n; i++) { char *e; v[i] = strtod(p, &e); if (e == p) return 0; p = e; while (*p == ' ' || *p == ',') p++; }
    return 1;
}
int room_load(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) return -1;
    fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
    char *s = malloc(len + 1); if (fread(s, 1, len, f) != (size_t)len) { fclose(f); free(s); return -1; } s[len] = 0; fclose(f);
    const char *p = strstr(s, "\"boxes\""); if (!p) { free(s); return -1; }
    static double v[ROOM_MAX * ROOM_REC]; int n = 0;
    for (p = strchr(p, '['); p && (p = strchr(p + 1, '{')) && n < ROOM_MAX; ) {
        const char *end = strchr(p, '}'); if (!end) break;
        double *b = v + ROOM_REC * n;
        if (!nums(field(p, end, "c"), b, 3) || !nums(field(p, end, "h"), b + 3, 3) || !nums(field(p, end, "q"), b + 6, 4)
            || !nums(field(p, end, "mu"), b + 10, 1) || !nums(field(p, end, "soft"), b + 11, 1)) { free(s); return -1; }
        if (!nums(field(p, end, "grip"), b + 12, 1)) b[12] = 0;   /* room.json v1 has no grip labels */
        n++; p = end;
    }
    free(s);
    return room_set(v, n);
}
/* a room bank (room.py --bank, tools/merge_rooms.py, tools/jump_course.mjs: {"rooms": [{"boxes": [...]}, ...]}) or a single
 * room.json, as rooms 0, 1, ... of the physics' bank (each world collides with its own, World.room; the nav grids
 * built); returns how many rooms, or -1 */
int room_load_bank(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) return -1;
    fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
    char *s = malloc(len + 1); if (fread(s, 1, len, f) != (size_t)len) { fclose(f); free(s); return -1; } s[len] = 0; fclose(f);
    static double v[ROOM_MAX * ROOM_REC]; int r = 0;
    for (const char *p = strstr(s, "\"boxes\""); p && r < ROOM_BANK; p = strstr(p, "\"boxes\"")) {
        p = strchr(p, '['); if (!p) break; int n = 0;
        for (p++; n < ROOM_MAX; ) {   /* the objects of this boxes array, up to its closing bracket (theirs are inside them) */
            const char *o = strchr(p, '{'), *close = strchr(p, ']');
            if (!o || (close && close < o)) { p = close ? close + 1 : p; break; }
            const char *end = strchr(o, '}'); if (!end) break;
            double *b = v + ROOM_REC * n;
            if (!nums(field(o, end, "c"), b, 3) || !nums(field(o, end, "h"), b + 3, 3) || !nums(field(o, end, "q"), b + 6, 4)
                || !nums(field(o, end, "mu"), b + 10, 1) || !nums(field(o, end, "soft"), b + 11, 1)) { free(s); return -1; }
            if (!nums(field(o, end, "grip"), b + 12, 1)) b[12] = 0;
            n++; p = end + 1;
        }
        if (room_set_bank(r, v, n) < 0) { free(s); return -1; }
        r++;
    }
    free(s);
    return r ? r : -1;
}
