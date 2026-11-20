#ifndef SCAR_CONTROLLER_EFFECT_ABI_V1_H
#define SCAR_CONTROLLER_EFFECT_ABI_V1_H

#include "scar_abi_v1.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum scar_controller_value_kind_v1 {
    SCAR_CONTROLLER_NONE_V1 = 0,
    SCAR_CONTROLLER_INT64_V1 = 1,
    SCAR_CONTROLLER_FLOAT64_V1 = 2,
    SCAR_CONTROLLER_BOOL_V1 = 3,
    SCAR_CONTROLLER_UTF8_V1 = 4
} scar_controller_value_kind_v1;

typedef struct scar_controller_value_v1 {
    uint32_t kind;
    uint32_t reserved;
    int64_t integer;
    double floating;
    const char* text;
    uint64_t text_size;
} scar_controller_value_v1;

/* Dispatch is synchronous. Argument text and roots are borrowed read-only
 * for this call. The provider keeps no references after return. */
typedef int (*scar_controller_effect_v1)(void* context, uint32_t ordinal,
    const scar_controller_value_v1* arguments, uint64_t argument_count,
    const scar_array_v1* roots, uint64_t root_count);

typedef int (*scar_controller_update_v1)(void* context, uint32_t ordinal,
    const scar_controller_value_v1* arguments, uint64_t argument_count,
    const scar_controller_value_v1* current_slots, uint64_t slot_count,
    const scar_array_v1* roots, uint64_t root_count, scar_controller_value_v1* result);

typedef struct scar_controller_effect_table_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    const char* controller_sha256;
    uint64_t effect_count;
    void* context;
    scar_controller_effect_v1 dispatch;
} scar_controller_effect_table_v1;

typedef const scar_controller_effect_table_v1* (*scar_controller_effect_getter_v1)(void);
SCAR_EXPORT_V1 const scar_controller_effect_table_v1* scar_get_controller_effects_v1(void);

#ifdef __cplusplus
SCAR_EXPORT_V1 int scar_controller_run_v1(scar_controller_value_v1* slots, uint64_t slot_count,
    const scar_array_v1* roots, uint64_t root_count, void* context, scar_controller_update_v1 update,
    scar_controller_effect_v1 effect, uint64_t max_iterations) noexcept;
SCAR_EXPORT_V1 const char* scar_controller_plan_sha256_v1(void) noexcept;
}
#endif
#endif
