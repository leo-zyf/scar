"""Project the original and proposed partition updates into the shared IR."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from scar.analysis.partition_copy import discover
from scar.ir.execution_recipe import artifact_reference
from scar.ir.v2.common import Completeness, EvidenceClaim, EvidenceKind, SourceReference
from scar.ir.v2.contracts import (ContractDefinition, ContractFacet, EffectPresence, EffectSet,
    EffectSummary, EffectTarget, EffectTargetKind, FacetRequirement)
from scar.ir.v2.ids import ContractID, EffectSummaryID, OperationDefinitionID, SourceAtomID, ValueSlotID
from scar.ir.v2.semantic import (OperationDefinition, OperationKind, SemanticEdge, SemanticEndpoint,
    SemanticGraph, SemanticNodeKind as Node, SemanticRelation as Edge, SourceAtom, ValueSlot)


SCHEMA = 'scar.partition-copy-plan'
OBLIGATIONS = (
    ('source_environment', 'RUNTIME_GUARD', 'Current code, builtin iteration and ndarray/ufunc providers.'),
    ('partition_geometry', 'RUNTIME_GUARD', 'Exhaustive basic selectors over disjoint output elements and disjoint operands.'),
    ('exclusive_window', 'EXTERNAL_ASSUMPTION', 'No concurrent observer, mutation, buffer escape or reentrant intervention within the window.'),
    ('numeric_contract', 'UNRESOLVED', 'Prove the same provider semantics for in-place and disjoint-output multiplication in the admitted dtype/layout/FP context.'),
    ('completion_contract', 'UNRESOLVED', 'Prove equivalent observable output state on exceptional exit or rule out such exits in the admitted window.'),
    ('entry_cost', 'UNRESOLVED', 'Include guards, dispatch, temporaries and memory traffic in the same entry scope.'),
)


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _graph(source, *, eliminate_copy):
    graph = SemanticGraph()
    key = _digest([source['path'], source['source_sha256'], source['code_id']])
    root = OperationDefinitionID(key + ':region')
    contract = ContractID(key + ':contract')
    atoms = {}
    for role, (first, start, last, end) in source['spans'].items():
        atom = SourceAtomID(key + ':' + role)
        reference = SourceReference(atom, source['path'], 'sha256:' + source['source_sha256'], first, last, start, end)
        graph.add_source_atom(SourceAtom(atom, reference, role))
        atoms[role] = atom
    evidence = EvidenceClaim(EvidenceKind.INFERRED, references=(source['code_id'],), scope='supplied-output branch')
    graph.add_definition(OperationDefinition(root, OperationKind.REGION, source['qualname'],
        code_id=source['code_id'], source_file=source['path'], contract=contract, metadata={
            'scope': 'supplied output, successful source prechecks', 'exception_effect': 'UNKNOWN',
            'allocation_branch_preserved': source['conditional_output_allocation'],
            'source_prechecks': source['entry_checks'], 'output_identity': 'original output argument',
        }))

    def edge(relation, left, right, kind=Node.OPERATION):
        identity = _digest([relation.value, left.wire, right.wire])
        graph.add_edge(SemanticEdge(identity, relation, SemanticEndpoint(Node.OPERATION, left),
                                    SemanticEndpoint(kind, right), evidence))

    def slot(role, direction, **metadata):
        value = ValueSlotID(key + ':' + role)
        graph.add_slot(ValueSlot(value, role, direction, owner=root, metadata=metadata))
        edge(Edge.CONTAINS, root, value, Node.VALUE_SLOT)
        return value

    original = slot('input content', 'input', storage='source')
    weights = slot('matrix', 'input', storage='weights')
    selectors = slot('partition selectors', 'input')
    output = slot('output object', 'input', storage='caller_output')
    result = slot('final output content', 'output', storage='caller_output')
    product = slot('diagonal product', 'temporary', region='partition[i]', storage='caller_output')
    accumulator = slot('off-diagonal input', 'temporary', region='partition[i]', storage='caller_output')
    updated = slot('off-diagonal result', 'temporary', region='partition[i]', storage='caller_output')
    completed = slot('final partition content', 'temporary', region='partition[i]', storage='caller_output')
    graph.definitions[root].input_slots = (original, weights, selectors, output)
    graph.definitions[root].output_slots = (result,)
    graph.add_contract(ContractDefinition(contract, 'Partition update', (result,), (
        FacetRequirement(ContractFacet.STATE, True, description='Same output object and visible writes.'),
        FacetRequirement(ContractFacet.NUMERICAL, True, description='Same values and arithmetic order.'),
        FacetRequirement(ContractFacet.EXCEPTION, True, description='Preserve observable exceptional completion.'),
        FacetRequirement(ContractFacet.ORDERING, True, description='Preserve relevant observation order.'),
    ), evidence=(evidence,)))
    edge(Edge.HAS_CONTRACT, root, contract, Node.CONTRACT)

    def operation(role, kind, label, parent, reads=(), writes=(), **metadata):
        identity = OperationDefinitionID(key + ':' + role)
        effect = EffectSummaryID(identity.value)
        targets = lambda values: EffectSet(tuple(EffectTarget(EffectTargetKind.VALUE_SLOT, value.wire)
            for value in values), Completeness.PARTIAL)
        graph.add_effect(EffectSummary(effect, reads=targets(reads), writes=targets(writes),
            ordering=EffectPresence.PRESENT, evidence=(evidence,)))
        graph.add_definition(OperationDefinition(identity, kind, label, parent_id=parent,
            source_atoms=(atoms[role],), effect_summary=effect, metadata=metadata))
        edge(Edge.CONTAINS, parent, identity)
        edge(Edge.HAS_SOURCE, identity, atoms[role], Node.SOURCE_ATOM)
        edge(Edge.HAS_EFFECT, identity, effect, Node.EFFECT)
        for value in reads:
            edge(Edge.CONSUMES, identity, value, Node.VALUE_SLOT)
        for value in writes:
            edge(Edge.PRODUCES, identity, value, Node.VALUE_SLOT)
        return identity

    prior = original
    if not eliminate_copy:
        prior = slot('copied input content', 'temporary', storage='caller_output', logical_origin=original.wire)
        copy = operation('copy', OperationKind.TRANSFER, 'Copy the complete input', root, (original, output), (prior,))
    outer = operation('outer', OperationKind.LOOP, 'For each output selector i', root, (selectors,), (result,),
                      iteration_order='enumeration', collect=completed.wire)
    diagonal = operation('diagonal', OperationKind.OPERATOR, 'Multiply partition i by coefficient (i,i)', outer,
        (prior, weights, selectors, output), (product,), provider='multiply',
        write_mode='direct_output' if eliminate_copy else 'inplace', region='partition[i]')
    inner = operation('inner', OperationKind.LOOP, 'For each input selector j distinct from i', outer,
        (selectors, product), (accumulator, completed), iteration_order='enumeration', predicate='i != j',
        carry={'initial': product.wire, 'next': updated.wire, 'iteration_distance': 1},
        result_selector='last update or initial product on zero updates')
    add = operation('off_diagonal', OperationKind.OPERATOR, 'Multiply input partition j and add to output partition i', inner,
        (original, weights, selectors, accumulator), (updated,), region='partition[i]', arithmetic=['multiply', 'inplace_add'])
    finish = operation('result', OperationKind.OPERATOR, 'Return the original output object', root,
                       (output, result), escape=output.wire)
    edge(Edge.CONSUMES, inner, updated, Node.VALUE_SLOT)
    edge(Edge.CONSUMES, outer, completed, Node.VALUE_SLOT)
    if not eliminate_copy:
        edge(Edge.CONTROLS, copy, outer)
    else:
        # The source coordinate for the removed action is kept in the original graph.
        del graph.source_atoms[atoms['copy']]
    for left, right in ((diagonal, inner), (add, inner), (outer, finish)):
        edge(Edge.CONTROLS, left, right)
    graph.assert_valid()
    return graph.to_dict()


def derive(source):
    """Rebuild the candidate and its actual successor graph from sealed source."""
    return dict(schema=SCHEMA, schema_version=1, source=source,
        before=_graph(source, eliminate_copy=False), after=_graph(source, eliminate_copy=True),
        prerequisites=[dict(id=key, status=status, detail=detail) for key, status, detail in OBLIGATIONS],
        condition='all output elements are covered exactly once by outer slices',
        resource_model=dict(removed_copy_bytes='input.nbytes',
            removed_logical_read_write_bytes='2 * input.nbytes',
            removed_arithmetic=0, estimated_duration_ns=None,
            physical_transfer_bytes=None, provider_temporary_bytes=None),
        legality='CONDITIONAL', decision='KEEP', execution_authorized=False)


def prepare(source_path, out):
    path = Path(out).resolve()
    candidates = [derive(row) for row in discover(source_path)]
    path.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, candidate in enumerate(candidates):
        files = {}
        for graph in ('before', 'after'):
            destination = path / f'{index}-{graph}.json'
            document = candidate.pop(graph)
            destination.write_text(json.dumps(document, indent=2, allow_nan=False) + '\n')
            files[graph] = artifact_reference(destination, document['schema'])
        rows.append({**candidate, 'graphs': files})
    result = dict(schema=SCHEMA + '-index', schema_version=1, source=str(Path(source_path).resolve()), candidates=rows)
    (path / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def check(path):
    from scar.ir.execution_recipe import read_artifact

    report = json.loads(Path(path).read_text())
    if (report['schema'], report['schema_version']) != (SCHEMA + '-index', 1):
        raise ValueError('unknown partition plan schema')
    expected = [derive(row) for row in discover(report['source'])]
    for actual, rederived in zip(report['candidates'], expected, strict=True):
        current = {name: value for name, value in actual.items() if name != 'graphs'}
        current.update({name: read_artifact(actual['graphs'][name]) for name in ('before', 'after')})
        if current != rederived:
            raise ValueError('partition plan differs from source replay')
    return len(expected)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar plan-partition-copy', description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--source')
    choice.add_argument('--check')
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    if args.check:
        if args.out:
            parser.error('--check reads a saved plan without --out')
        print(json.dumps(dict(verified_candidates=check(args.check))))
    else:
        if not args.out:
            parser.error('--source requires --out')
        result = prepare(args.source, args.out)
        print(json.dumps(dict(candidates=len(result['candidates']), decision='KEEP')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
