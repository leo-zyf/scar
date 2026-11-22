"""Structural simplification for the standalone functional ATen file ABI."""

from hashlib import sha256

from scar.ir.v2.schemas import canonical_json
from scar.optimization.aten_live_values import tensor_releases


# Exact schemas within the statically specialized contiguous CPU FP32 ABI.
# These calls depend on explicit Tensor/scalar operands, create fresh results,
# and do not mutate input storage, publish results, or consume RNG state.
# Native recipe normalization separately verifies the installed dispatcher
# declaration, its CPU route, and the absence of alias/write annotations.
FUNCTIONAL_EXPORT_SCHEMAS = {
    'aten.mm.default': 'aten::mm(Tensor self, Tensor mat2) -> Tensor',
    'aten.matmul.default': 'aten::matmul(Tensor self, Tensor other) -> Tensor',
    'aten.add.Tensor': 'aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor',
    'aten.sub.Tensor': 'aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor',
    'aten.mul.Tensor': 'aten::mul.Tensor(Tensor self, Tensor other) -> Tensor',
    'aten.relu.default': 'aten::relu(Tensor self) -> Tensor',
    'aten.sigmoid.default': 'aten::sigmoid(Tensor self) -> Tensor',
    'aten.tanh.default': 'aten::tanh(Tensor self) -> Tensor',
}
_EXACT_SCHEMAS = frozenset(FUNCTIONAL_EXPORT_SCHEMAS.values())


def eliminate_dead_export_ops(recipe):
    """Return a reduced normalized export and its replayable structural receipt.

    The caller must have statically specialized and validated the source
    geometry, or be replaying an archive of that same qualified recipe. This
    law applies to normal completion of the standalone file entry. Library or
    resource failures and process-local floating-point status are outside that
    entry contract. Unknown schemas may observe hidden state, so a recipe with
    any unknown effect retains its whole body.
    """
    tensor_releases(recipe)  # Check the closed ordered use/def boundary.
    operations = recipe['operations']
    missing_effects = [op['id'] for op in operations if op['schema'] not in _EXACT_SCHEMAS]
    removed = []
    retained = operations
    if not missing_effects:
        live = set(recipe['outputs'])
        for op in reversed(operations):
            if op['id'] in live:
                live.update(arg['value'] for arg in op['arguments'] if 'value' in arg)
        removed = [op['id'] for op in operations if op['id'] not in live]
        retained = [op for op in operations if op['id'] in live]
    reduced = dict(recipe, operations=retained)
    decision = ('RETAIN_UNQUALIFIED_EFFECTS' if missing_effects else
                'ELIMINATE_DEAD_OPERATIONS' if removed else 'RETAIN_ALL_RESULTS_DEMANDED')
    return reduced, dict(
        schema='scar.aten-export-structural-rewrite', schema_version=1,
        rule='BACKWARD_LIVENESS_FROM_FILE_RETURN_PORTS', decision=decision,
        source_recipe_sha256=sha256(canonical_json(recipe).encode()).hexdigest(),
        lowered_recipe_sha256=sha256(canonical_json(reduced).encode()).hexdigest(),
        original_operation_count=len(operations), lowered_operation_count=len(retained),
        removed_operations=removed, unqualified_effects=missing_effects,
        scope='NORMAL_COMPLETION_STANDALONE_CPU_FLOAT32_FILE_ABI',
        preserved=['ORDERED_INPUT_PORTS', 'ORDERED_RETURN_PORTS', 'RETAINED_CALL_ORDER', 'RETURNED_VALUE_IDENTITIES'],
        excluded=['PROVIDER_FAILURES', 'RESOURCE_EXHAUSTION', 'PROCESS_LOCAL_FLOATING_POINT_STATUS'],
    )
