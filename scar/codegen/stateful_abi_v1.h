#ifndef SCAR_STATEFUL_ABI_V1_H
#define SCAR_STATEFUL_ABI_V1_H
#include "scar_abi_v1.h"

typedef struct scar_scalar_v1 {
    double value;
    int64_t integer;
} scar_scalar_v1;

/* Exact version-pinned NumPy native ufunc loop; the caller owns its capsule. */
typedef int (*scar_provider_loop_v1)(void *, char **, const intptr_t *,
                                    const intptr_t *, void *);
typedef struct scar_provider_v1 {
    scar_provider_loop_v1 loop;
    void *context;
    void *auxdata;
} scar_provider_v1;

typedef int32_t (*scar_stateful_run_v1)(const scar_array_v1 *, uint64_t,
    const scar_scalar_v1 *, uint64_t, const scar_provider_v1 *, uint64_t);
typedef struct scar_stateful_table_v1 {
    uint32_t abi_version, struct_size;
    uint64_t array_count, scalar_count, provider_count;
    scar_stateful_run_v1 run;
    const char *region_sha256, *plan_sha256, *contract_sha256;
} scar_stateful_table_v1;

extern "C" SCAR_EXPORT_V1 const scar_stateful_table_v1 *scar_get_stateful_table_v1();
#endif
