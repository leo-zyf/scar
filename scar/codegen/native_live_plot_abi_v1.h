#ifndef SCAR_NATIVE_LIVE_PLOT_ABI_V1_H
#define SCAR_NATIVE_LIVE_PLOT_ABI_V1_H

#include "scar_abi_v1.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Synchronous. Roots and path strings are borrowed until return. On success
 * the target contains a complete new PNG. On failure no new target is kept. */
typedef int (*scar_live_plot_fn_v1)(const scar_array_v1* x_root, const scar_array_v1* y_root,
    const scar_array_v1* z_root, double time, const char* layout_path, const char* font_path,
    const char* output_png, char* error, uint64_t error_capacity);

SCAR_EXPORT_V1 int scar_live_plot_v1(const scar_array_v1* x_root, const scar_array_v1* y_root,
    const scar_array_v1* z_root, double time, const char* layout_path, const char* font_path,
    const char* output_png, char* error, uint64_t error_capacity);

#ifdef __cplusplus
}
#endif
#endif
