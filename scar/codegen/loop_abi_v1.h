#ifndef SCAR_LOOP_ABI_V1_H
#define SCAR_LOOP_ABI_V1_H
#include "scar_abi_v1.h"

#ifdef __cplusplus
extern "C" {
#endif

/* This is a loop ABI, not a single body operator. The exact positive count,
 * source loop, strategy and bound body plan are distinct versioned inputs. */
typedef int32_t (*scar_loop_run_fn_v1)(const scar_array_v1 *state,
                                      uint64_t trip_count,
                                      scar_array_v1 *output);

typedef struct scar_loop_table_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint64_t trip_count;
    scar_loop_run_fn_v1 run;
    const char *loop_sha256;
    const char *strategy_sha256;
    const char *execution_plan_sha256;
    const char *contract_sha256;
} scar_loop_table_v1;

enum scar_loop_status_v1 {
    SCAR_LOOP_TRIP_COUNT_GUARD_V1 = 5,
    SCAR_LOOP_PAYLOAD_GUARD_V1 = 6
};

SCAR_EXPORT_V1 const scar_loop_table_v1 *scar_get_loop_table_v1(void);
#ifdef __cplusplus
}
#endif
#endif
