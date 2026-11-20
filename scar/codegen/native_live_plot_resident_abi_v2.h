#ifndef SCAR_NATIVE_LIVE_PLOT_RESIDENT_ABI_V2_H
#define SCAR_NATIVE_LIVE_PLOT_RESIDENT_ABI_V2_H

#include "scar_abi_v1.h"

#ifdef __cplusplus
extern "C" {
#endif

/* An owner snapshots the qualified layout, font and immutable X/Y roots.
 * The Z descriptor is borrowed through one synchronous render call.
 * All operations on an owner use the thread that prepared it. Destroy
 * before unloading the library. Arguments remain owned by the caller. */
typedef int (*scar_live_plot_prepare_fn_v2)(const scar_array_v1* x, const scar_array_v1* y,
    const char* layout, const char* font, void** context, char* error, uint64_t capacity);
typedef int (*scar_live_plot_render_fn_v2)(void* context, const scar_array_v1* z,
    double time, const char* output, char* error, uint64_t capacity);
typedef void (*scar_live_plot_destroy_fn_v2)(void* context);

SCAR_EXPORT_V1 int scar_live_plot_prepare_v2(const scar_array_v1* x, const scar_array_v1* y,
    const char* layout, const char* font, void** context, char* error, uint64_t capacity);
SCAR_EXPORT_V1 int scar_live_plot_render_v2(void* context, const scar_array_v1* z,
    double time, const char* output, char* error, uint64_t capacity);
SCAR_EXPORT_V1 void scar_live_plot_destroy_v2(void* context);

#ifdef __cplusplus
}
#endif
#endif
