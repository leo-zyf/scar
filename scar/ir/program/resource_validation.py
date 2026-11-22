"""Resource validation."""

from __future__ import annotations

from scar.ir.program.ids import OperationID, UseID, ValueID, RegionID
from scar.ir.program.model import (
    ReferenceValue,
    CompletionResult,
    DataValue,
    LocalBindingValue,
    OperationResult,
    ProtocolValue,
    StateVersionValue,
    AccessMode,
    ProgramGraph,
    TerminatorKind,
    UseRole,
)
from scar.ir.program.resources import CreatedResource, RequirementRelation, ResourceUse
from scar.ir.v2._validation import cycle_errors
from scar.ir.program.control import PendingPayload
from scar.ir.program.providers import (
    ProviderArtifactKind,
    ProviderElementKind,
    ProviderRegionRelation,
)
from scar.ir.program.resumption import ResumeInput
from scar.ir.program.state_addresses import (
    AddressCoverage,
    AddressEffect,
    DomainOwner,
    ExistingAccess,
    InvokeStateOperand,
    ReferenceOwner,
    ResourceOwner,
    StateAddressRefinement,
    UnknownOwner,
    WholeOwner,
    address_effect_uses,
)


# Validate static resource interfaces, not dynamic realization or readiness.
def resource_errors(graph, operation_region, uses):
    errors = []

    def error(code, path, message):
        errors.append((code, path, message))

    definitions = {row.id: row for row in graph.resources}
    requirements = {row.id: row for row in graph.resource_requirements}
    operations = {row.id for row in graph.operations}
    regions = {row.id: row for row in graph.regions}
    locations = {row.id for row in graph.locations}
    values = {row.id: row for row in graph.values}

    for resource in graph.resources:
        origin = resource.origin
        if isinstance(origin, CreatedResource):
            if origin.operation not in operations:
                error("resource_origin", resource.id, "creation operation does not exist")
        elif origin.region not in regions:
            error("resource_origin", resource.id, "external resource region does not exist")

    same_parent = {key: key for key in requirements}

    def representative(key):
        while same_parent[key] != key:
            same_parent[key] = same_parent[same_parent[key]]
            key = same_parent[key]
        return key

    valid_relations = []
    for requirement in graph.resource_requirements:
        owners = operations if isinstance(requirement.owner, OperationID) else regions
        if requirement.owner not in owners:
            error("resource_requirement_owner", requirement.id, "requirement owner does not exist")
        for relation in requirement.relations:
            other = requirements.get(relation.other)
            if other is None:
                error("resource_constraint", requirement.id, "related requirement does not exist")
                continue
            if requirement.owner != other.owner:
                error("resource_constraint_scope", requirement.id,
                      "related requirements need the same owner/activation scope")
                continue
            valid_relations.append((requirement, relation))
            if relation.relation is RequirementRelation.SAME_RESOURCE:
                same_parent[representative(requirement.id)] = representative(other.id)

    component_types = {}
    for requirement in graph.resource_requirements:
        root = representative(requirement.id)
        kind_role = requirement.kind, requirement.role
        if root in component_types and component_types[root] != kind_role:
            error("resource_constraint", requirement.id, "same-resource kinds/roles disagree")
        component_types[root] = kind_role
    child_edges = []
    for requirement, relation in valid_relations:
        left, right = representative(requirement.id), representative(relation.other)
        if relation.relation is not RequirementRelation.SAME_RESOURCE and left == right:
            error("resource_constraint", requirement.id, "same-resource closure contradicts relation")
        if relation.relation is RequirementRelation.CHILD_RESOURCE:
            child_edges.append((left, right))
    for message in cycle_errors(child_edges, "strict resource containment"):
        error("resource_constraint", "resource_requirements", message)

    def region_contains(ancestor, child):
        visited = set()
        while child in regions and child not in visited:
            if child == ancestor:
                return True
            visited.add(child)
            child = regions[child].parent
        return False

    used_keys = set()
    for index, use in enumerate(graph.resource_uses):
        path = f"resource_uses[{index}]"
        key = use.operation, use.requirement
        if key in used_keys:
            error("resource_use", path, "one operation/requirement has conflicting bindings")
        used_keys.add(key)
        if use.operation not in operations:
            error("resource_use", path, "operation does not exist")
        requirement = requirements.get(use.requirement)
        if requirement is None:
            error("resource_use", path, "requirement does not exist")
        elif isinstance(requirement.owner, OperationID):
            if requirement.owner != use.operation:
                error("resource_use_owner", path, "operation uses another operation's requirement")
        elif not region_contains(requirement.owner, operation_region.get(use.operation)):
            error("resource_use_owner", path, "operation is outside the owning region")
        if use.definition is not None:
            definition = definitions.get(use.definition)
            if definition is None:
                error("resource_target", path, "definition does not exist")
            elif requirement is not None and (definition.kind is not requirement.kind
                                              or definition.role is not requirement.role):
                error("resource_target", path, "definition does not meet kind/role requirement")
        if use.handle is not None:
            occurrence = uses.get(use.handle)
            if (occurrence is None or occurrence[0] != use.operation
                    or not isinstance(values.get(occurrence[1].value), ReferenceValue)):
                error("resource_handle", path, "handle is not this operation's reference Use")

    for index, binding in enumerate(graph.resource_locations):
        if binding.location not in locations or binding.resource not in definitions:
            error("resource_location", f"resource_locations[{index}]",
                  "resource state binding needs existing location and definition")
    return tuple(errors)


# Check declared external provenance against the canonical core.
#
# No artifact is opened and no provider claim is promoted to a semantic proof.
# Actual producers, use owners and state flow are always read from ProgramGraph.
def provider_errors(graph, operation_region, uses):
    if not (graph.provider_envelopes or graph.provider_regions or graph.provider_bindings):
        return ()
    errors = []

    def error(code, path, message):
        errors.append((code, str(path), message))

    envelopes = {row.id: row for row in graph.provider_envelopes}
    regions = {row.id for row in graph.regions}
    values = {row.id: row for row in graph.values}
    owners = {}
    region_coordinates = set()
    definitions = set()
    for index, row in enumerate(graph.provider_regions):
        path = f"provider_regions[{index}]"
        envelope = envelopes.get(row.provider)
        if envelope is None:
            error("provider_reference", path, "provider envelope does not exist")
        if row.region not in regions:
            error("provider_region", path, "core region does not exist")
        if row.relation is ProviderRegionRelation.DEFINES_REPRESENTATION:
            coordinate = row.provider, row.external_region
            if coordinate in region_coordinates:
                error("provider_region_coordinate", path, "external definition region is already represented")
            region_coordinates.add(coordinate)
            if envelope is not None and envelope.kind is not ProviderArtifactKind.DEFINITION_GRAPH:
                error("provider_region_kind", path, "only a definition graph can own a representation")
            if row.region in owners:
                error("provider_representation_owner", path, "core region already has a representation owner")
            owners[row.region] = row.provider
            definitions.add((row.provider, row.region))
        elif envelope is not None and envelope.kind is not ProviderArtifactKind.EXECUTION_EVIDENCE:
            error("provider_region_kind", path, "observation scope requires execution evidence")

    value_classes = {
        ProviderElementKind.DATA_VALUE: DataValue,
        ProviderElementKind.REFERENCE_VALUE: ReferenceValue,
        ProviderElementKind.STATE_VALUE: StateVersionValue,
        ProviderElementKind.BINDING_VALUE: LocalBindingValue,
        ProviderElementKind.PROTOCOL_VALUE: ProtocolValue,
    }
    external_coordinates = set()
    canonical_targets = set()
    operation_bindings = {}
    blocks = {block.id: block for region in graph.regions for block in region.blocks}
    edge_producers = {}
    incoming = {block: [] for block in blocks}
    for operation in graph.operations:
        if operation.terminator is not None:
            for edge in operation.terminator.successors:
                incoming[edge.block].append((operation.id, edge))
    for block, edges in incoming.items():
        # Multiple real edges make this a phi, even when one incoming edge
        # projects a payload. Never choose a producer by record iteration order.
        if len(edges) != 1:
            continue
        operation, edge = edges[0]
        for actual, formal in zip(edge.arguments, blocks[block].arguments):
            if isinstance(actual, (CompletionResult, PendingPayload, ResumeInput)):
                # This checks the external binding's boundary-operation owner.
                # The canonical value remains a BlockArgument, including for
                # externally supplied resume state/payload; no producer or
                # runtime evidence is added to ProgramGraph.
                edge_producers[formal] = operation
    for index, row in enumerate(graph.provider_bindings):
        path = f"provider_bindings[{index}]"
        envelope = envelopes.get(row.provider)
        if envelope is None:
            error("provider_reference", path, "provider envelope does not exist")
        elif envelope.kind is not ProviderArtifactKind.DEFINITION_GRAPH:
            error("provider_binding_kind", path, "execution evidence cannot bind to a static definition")
        if (row.provider, row.region) not in definitions:
            error("provider_binding_scope", path, "binding needs this provider's explicit representation ownership")

        kind = row.external.kind
        group = "VALUE" if kind in value_classes else kind.value
        coordinate = row.provider, row.region, group, row.external.key, row.external.path
        if coordinate in external_coordinates:
            error("provider_endpoint_conflict", path, "external endpoint has more than one binding")
        external_coordinates.add(coordinate)
        target = row.provider, row.target
        if target in canonical_targets:
            error("provider_target_conflict", path, "canonical target folds multiple external endpoints")
        canonical_targets.add(target)

        actual_region = None
        if type(row.target) is OperationID:
            actual_region = operation_region.get(row.target)
            operation_bindings[(row.provider, row.target)] = row
        elif type(row.target) is UseID:
            site = uses.get(row.target)
            if site is not None:
                actual_region = operation_region.get(site[0])
        elif type(row.target) is ValueID:
            value = values.get(row.target)
            if value is not None:
                actual_region = value.region
                if not isinstance(value, value_classes[kind]):
                    error("provider_value_kind", path, "external value category disagrees with the canonical value")
        if actual_region is None:
            error("provider_target", path, "canonical target does not exist")
        elif actual_region != row.region:
            error("provider_target_region", path, "canonical target belongs to another region")

    for index, row in enumerate(graph.provider_bindings):
        path = f"provider_bindings[{index}]"
        producer = None
        if type(row.target) is UseID:
            site = uses.get(row.target)
            if site is not None:
                producer = site[0]  # Here this is the use's owning operation.
        elif type(row.target) is ValueID:
            value = values.get(row.target)
            if value is not None and isinstance(value.origin, OperationResult):
                producer = value.origin.operation
            else:
                producer = edge_producers.get(row.target)
            if producer is not None and not row.external.path:
                error("provider_result_projection", path, "produced result needs an explicit external result path")
        if producer is not None:
            binding = operation_bindings.get((row.provider, producer))
            if (binding is None or binding.region != row.region
                    or binding.external.key != row.external.key):
                error("provider_binding_owner", path,
                      "external node key does not match the canonical use owner or result producer")
    return tuple(errors)


# Validate address refinements against the graph's original state ports.
#
# This validator never adds state accesses or versions. It checks only typed
# ownership and consistency with the original coarse summaries/invoke ports;
# address metadata does not establish alias disjointness or runtime effects.
def _mode_bits(mode):
    if mode is AccessMode.READ:
        return 1
    if mode is AccessMode.WRITE:
        return 2
    if mode is AccessMode.READ_WRITE:
        return 3
    return 0


def _root_replacement(effect):
    return (
        type(effect) is AddressEffect
        and type(effect.owner) is DomainOwner
        and len(effect.path) == 1
        and type(effect.path[0]) is WholeOwner
        and effect.mode is AccessMode.WRITE
    )


def state_address_errors(graph: ProgramGraph) -> tuple[tuple[str, str, str], ...]:
    """Return structural address issues without changing canonical state SSA.

    `graph` is expected to be a ProgramGraph. Malformed/foreign references are
    reported as bounded issues rather than used as dictionary indexes.
    """
    issues: list[tuple[str, str, str]] = []

    def error(code, path, message):
        issues.append((code, str(path), message))

    if type(graph) is not ProgramGraph:
        return (("address_graph", "graph", "expected ProgramGraph"),)

    operations = {row.id: row for row in graph.operations}
    values = {row.id: row for row in graph.values}
    requirements = {row.id: row for row in graph.resource_requirements}
    regions = {row.id: row for row in graph.regions}
    resource_definition_ids = {row.id for row in graph.resources}
    resources_by_operation: dict[object, list[ResourceUse]] = {}
    for row in graph.resource_uses:
        resources_by_operation.setdefault(row.operation, []).append(row)
    operation_region = {}
    for region in graph.regions:
        for block in region.blocks:
            for operation_id in block.operations:
                operation_region[operation_id] = region.id

    def requirement_owns_operation(requirement, operation_id):
        owner = requirement.owner
        if type(owner) is OperationID:
            return owner == operation_id
        if type(owner) is not RegionID:
            return False
        current = operation_region.get(operation_id)
        visited = set()
        while current in regions and current not in visited:
            if current == owner:
                return True
            visited.add(current)
            current = regions[current].parent
        return False

    seen_sites = set()
    seen_carriers = set()
    for row_index, row in enumerate(graph.state_address_refinements):
        base_path = f"state_address_refinements[{row_index}]"
        if type(row) is not StateAddressRefinement:
            error("address_record", base_path, "expected exact StateAddressRefinement")
            continue
        try:
            row.__post_init__()
        except (TypeError, ValueError) as exc:
            error("address_record", base_path, str(exc))
            continue

        if row.site in seen_sites:
            error("address_duplicate_site", base_path,
                  "an original state-effect site may have only one refinement")
        seen_sites.add(row.site)

        operation_id = row.site.operation
        operation = operations.get(operation_id)
        if operation is None:
            error("address_operation", base_path, "site operation is not in this graph")
            continue
        own_uses = {use.id: use for use in operation.uses}
        is_invoke = (
            operation.terminator is not None
            and operation.terminator.kind is TerminatorKind.INVOKE
        )

        access = None
        state_location = None
        if type(row.site) is ExistingAccess:
            if is_invoke:
                error("address_site_kind", base_path,
                      "ExistingAccess cannot name an INVOKE operation")
                continue
            if row.site.index >= len(operation.accesses):
                error("address_access_index", base_path,
                      "access index is not present on the operation")
                continue
            access = operation.accesses[row.site.index]
            state_location = access.location
            before_use = own_uses.get(access.before)
            if before_use is None:
                error("address_access_before", base_path,
                      "selected access has no owned before Use")
            else:
                before_value = values.get(before_use.value)
                expected_role = (UseRole.STATE_ORDER if access.full_overwrite
                                 else UseRole.STATE_CONTENT)
                if before_use.role is not expected_role:
                    error("address_access_role", base_path,
                          "selected access before Use disagrees with its overwrite role")
                if not isinstance(before_value, StateVersionValue):
                    error("address_access_state", base_path,
                          "selected access before Use is not a StateVersionValue")
                elif before_value.location != access.location:
                    error("address_access_state", base_path,
                          "selected access before state has the wrong location")
            if access.mode not in (AccessMode.READ, AccessMode.WRITE, AccessMode.READ_WRITE):
                error("address_access_mode", base_path, "selected access has an invalid mode")
        elif type(row.site) is InvokeStateOperand:
            if not is_invoke:
                error("address_site_kind", base_path,
                      "InvokeStateOperand must name an INVOKE operation")
                continue
            state_use = own_uses.get(row.site.state_use)
            if state_use is None:
                error("address_invoke_use", base_path,
                      "state operand is not an owned Use")
                continue
            if row.site.state_use not in operation.invoke_arguments:
                error("address_invoke_argument", base_path,
                      "state operand is not an actual invoke_arguments Use")
            if state_use.role is not UseRole.STATE_CONTENT:
                error("address_invoke_role", base_path,
                      "INVOKE state operands require the core STATE_CONTENT role")
            state_value = values.get(state_use.value)
            if not isinstance(state_value, StateVersionValue):
                error("address_invoke_state", base_path,
                      "invoke state operand does not carry a StateVersionValue")
            else:
                state_location = state_value.location
        else:
            error("address_site_kind", base_path, "unsupported state address site variant")
            continue

        if state_location is not None:
            carrier_key = (operation_id, state_location)
            if carrier_key in seen_carriers:
                error("address_duplicate_carrier", base_path,
                      "an operation may have only one refinement per derived state carrier")
            seen_carriers.add(carrier_key)

        resource_uses = resources_by_operation.get(operation_id, ())
        reference_envelope = None
        if access is not None and access.reference is not None:
            reference_use = own_uses.get(access.reference)
            if reference_use is None or reference_use.role is not UseRole.REFERENCE:
                error("address_reference_envelope", base_path,
                      "access reference is not an owned REFERENCE Use")
            else:
                reference_envelope = values.get(reference_use.value)
                if not isinstance(reference_envelope, ReferenceValue):
                    error("address_reference_envelope", base_path,
                          "access reference Use does not carry a ReferenceValue")
                elif (reference_envelope.targets is not None
                      and state_location is not None
                      and state_location not in reference_envelope.targets):
                    error("address_reference_target", base_path,
                          "access reference excludes the selected state location")

        covered_modes = 0
        for effect_index, effect in enumerate(row.effects):
            effect_path = f"{base_path}.effects[{effect_index}]"
            if type(effect) is not AddressEffect:
                error("address_effect", effect_path, "expected exact AddressEffect")
                continue
            try:
                effect.__post_init__()
            except (TypeError, ValueError) as exc:
                error("address_effect", effect_path, str(exc))
                continue

            effect_uses = []
            owner = effect.owner
            if type(owner) is ReferenceOwner:
                owner_use = own_uses.get(owner.use)
                if owner_use is None:
                    error("address_owner_use", effect_path,
                          "reference owner is not an owned Use")
                elif owner_use.role is not UseRole.REFERENCE:
                    error("address_owner_role", effect_path,
                          "reference owner Use must have REFERENCE role")
                else:
                    owner_value = values.get(owner_use.value)
                    if not isinstance(owner_value, ReferenceValue):
                        error("address_owner_value", effect_path,
                              "reference owner Use does not carry a ReferenceValue")
                    else:
                        if (owner_value.targets is not None and state_location is not None
                                and state_location not in owner_value.targets):
                            error("address_owner_target", effect_path,
                                  "reference owner excludes the selected state location")
                        if (reference_envelope is not None
                                and owner_value.id != reference_envelope.id):
                            error("address_reference_envelope", effect_path,
                                  "reference owner is outside the access reference envelope")
                effect_uses.append(owner.use)
            elif type(owner) is ResourceOwner:
                requirement = requirements.get(owner.requirement)
                if requirement is None:
                    error("address_resource_requirement", effect_path,
                          "resource owner requirement is not in this graph")
                else:
                    found = [item for item in resource_uses
                             if item.requirement == owner.requirement]
                    if not found:
                        error("address_resource_use", effect_path,
                              "resource owner has no ResourceUse on this operation")
                    for item in found:
                        if item.definition is not None and item.definition not in resource_definition_ids:
                            error("address_resource_definition", effect_path,
                                  "ResourceUse names a missing resource definition")
                        if reference_envelope is not None and item.handle is None:
                            error("address_reference_envelope", effect_path,
                                  "resource owner has no handle inside the access reference envelope")
                        if item.handle is not None:
                            handle = own_uses.get(item.handle)
                            if handle is None or handle.role is not UseRole.REFERENCE:
                                error("address_resource_handle", effect_path,
                                      "ResourceUse handle is not an owned REFERENCE Use")
                            else:
                                effect_uses.append(item.handle)
                                if (reference_envelope is not None
                                        and handle.value != reference_envelope.id):
                                    error("address_reference_envelope", effect_path,
                                          "resource handle is outside the access reference envelope")
            elif type(owner) is UnknownOwner:
                effect_uses.extend(owner.operands)
                if reference_envelope is not None:
                    error("address_reference_envelope", effect_path,
                          "UnknownOwner cannot bypass an explicit access reference envelope")
            elif type(owner) is DomainOwner:
                if reference_envelope is not None:
                    error("address_reference_envelope", effect_path,
                          "DomainOwner cannot bypass an explicit access reference envelope")
            else:
                error("address_owner", effect_path, "unsupported address owner variant")

            try:
                effect_uses.extend(address_effect_uses(effect))
            except (TypeError, ValueError) as exc:
                error("address_selector", effect_path, str(exc))

            for use_id in effect_uses:
                use = own_uses.get(use_id)
                if use is None:
                    error("address_operand_use", effect_path,
                          f"address operand {use_id} is not owned by this operation")
                    continue
                if use.role not in (UseRole.DATA, UseRole.REFERENCE):
                    error("address_operand_role", effect_path,
                          "owner/key/index/slice operands must be DATA or REFERENCE Uses")
                if is_invoke and use_id not in operation.invoke_arguments:
                    error("address_invoke_argument", effect_path,
                          "address operand must be an actual invoke_arguments Use")

            if type(owner) is ResourceOwner:
                # A resource requirement is owned either directly by this
                # operation or by a lexical ancestor region, matching the
                # canonical ResourceUse ownership rule. No resource identity
                # is inferred from a label or unrelated ResourceUse.
                requirement = requirements.get(owner.requirement)
                if (requirement is not None
                        and not requirement_owns_operation(requirement, operation_id)):
                    error("address_resource_scope", effect_path,
                          "resource requirement belongs to another operation/region")

            effect_bits = _mode_bits(effect.mode)
            covered_modes |= effect_bits
            if access is not None and effect_bits & ~_mode_bits(access.mode):
                error("address_mode_exceeds_summary", effect_path,
                      "address effect mode exceeds its selected StateAccess mode")

        if access is not None and row.coverage is AddressCoverage.DECLARED_COMPLETE:
            if covered_modes != _mode_bits(access.mode):
                error("address_mode_incomplete", base_path,
                      "complete address effects do not cover the original access mode")

        root_replacement = (
            len(row.effects) == 1 and _root_replacement(row.effects[0])
            and row.coverage is AddressCoverage.DECLARED_COMPLETE
        )
        if access is not None and access.full_overwrite and not root_replacement:
            error("address_full_overwrite", base_path,
                  "a refined full overwrite must remain a complete DomainOwner whole-root WRITE")
    return tuple(issues)


__all__ = [
    'resource_errors',
    'provider_errors',
    'state_address_errors',
]
