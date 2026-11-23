"""Stream per-source rule recipes and explicit frontiers; never run targets."""
import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import gzip
from hashlib import sha256
import io
import json
import multiprocessing
import os
from pathlib import Path
import time

from scar.analysis.frontend_bindings import _replay_frontend_unit
from scar.analysis.source_contexts import SourceContextView
from scar.analysis.source_expression_rules import SourceExpressionRules, SourceRuleError
from scar.analysis.source_statement_rules import SourceStatementRules
from scar.analysis.source_root_rules import SourceRootRules
from scar.analysis.source_productions import ProductionSourceIndex, ProductionError
from scar.analysis.source_core_interfaces import CORE_INSTRUCTIONS
from scar.artifacts import _durable_bytes, _durable_mkdir, _sync_directory, file_reference
from scar.analysis.source_materials import _matches, _source_envelope
from scar.analysis.static.query import _Fragments
from scar.ir.program import source_productions as P
from scar.ir.program.ids import LocationID
from scar.ir.program.model import ValueKind, ValueType
from scar.ir.program.protocol_signatures import ExceptionABI
from scar.ir.record_codec import decode, encode


_PARENTS = ('source_roles', 'source_contexts', 'source_declarations', 'source_bindings',
            'source_symbols', 'compiler_topology', 'compiler_symbols')
_FRONTIERS = {'RULE_FRONTIER', 'PURPOSE_FRONTIER', 'BINDING_INTERFACE_FRONTIER', 'SUSPENSION_CONTEXT_FRONTIER',
              'STATEMENT_CONTEXT_FRONTIER'}
_WORLD = ValueType(ValueKind.STATE, location=LocationID('source-python-world'))
_RULES = {
    'expressions': ('expr', 'expression', P.ProductionPurpose.VALUE, SourceExpressionRules),
    'statements': ('stmt', 'statement', P.ProductionPurpose.STATEMENT, SourceStatementRules),
    'roots': ('mod', 'root', P.ProductionPurpose.ROOT, SourceRootRules),
}
_LIMITS = dict(nodes=1_000_000, tokens=2_000_000, steps=100_000, regions=100_000, visits=1_000_000)


def _json(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False) + '\n').encode('ascii')


def _implementation():
    package = Path(__file__).resolve().parents[1]
    names = ('artifacts.py', 'analysis/source_expression_materials.py', 'analysis/source_expression_rules.py',
        'analysis/frontend_bindings.py',
        'analysis/static/lowering/source_displays.py', 'analysis/static/lowering/source_string.py',
        'analysis/static/lowering/source_comprehension.py',
        'analysis/await_protocol.py', 'ir/program/delegation_signatures.py',
        'analysis/source_statement_interfaces.py',
        'analysis/source_deferred_interfaces.py',
        'analysis/source_statement_rules.py', 'analysis/statement_bindings.py',
        'analysis/source_statement_exceptions.py',
        'analysis/source_control.py',
        'analysis/source_root_rules.py',
        'analysis/source_scope_annotation_interfaces.py',
        'analysis/annotation_state.py',
        'analysis/source_compilation_contexts.py',
        'analysis/source_core_interfaces.py',
        'analysis/source_syntax.py', 'analysis/source_materials.py',
        'analysis/source_productions.py', 'analysis/program_types.py',
        'ir/program/source_productions.py', 'ir/record_codec.py')
    return {name: sha256((package / name).read_bytes()).hexdigest() for name in names}


def _basis(unit):
    key = unit['key']
    if type(key) is not str or not key or Path(key).name != key or key in ('.', '..'):
        raise ValueError('source unit identity must be one safe path component')
    names = ('source', 'source_bytes', 'source_tokens', 'source_syntax', *_PARENTS)
    return dict(unit=unit['key'], path=unit['path'], namespace_root=unit['namespace_root'],
                artifacts={name: unit['artifacts'][name] for name in names})


def _source_index(model, unit, limits, pins):
    if _implementation() != pins:
        raise ValueError('source rule implementation changed between parent and worker')
    reader = _Fragments(Path(model))
    source = reader.read(unit['artifacts']['source'])
    size = _source_envelope(unit, source)
    f0, f1, f2 = (reader.read(unit['artifacts'][name])
                  for name in ('source_bytes', 'source_tokens', 'source_syntax'))
    if not _matches(unit, source, f0, size):
        raise ValueError('source stage requires matching original F0 bytes/text')
    for stage, material, budget, ceiling in (('F1', f1, 'max_tokens', limits['tokens']),
                                               ('F2', f2, 'max_nodes', limits['nodes'])):
        if type(material[budget]) is not int or not 0 <= material[budget] <= ceiling:
            raise ValueError(f'{stage} replay budget exceeds the requested {budget} limit')
    frontend, syntax = _replay_frontend_unit(f0, f1, f2,
        *(reader.read(unit['artifacts'][name]) for name in _PARENTS))
    contexts = SourceContextView(syntax, frontend.roles, frontend.contexts)
    categories = {row.name: row.bases for row in syntax.profile.ast_schema if not row.abstract}
    return ProductionSourceIndex(f0, contexts, frontend), categories


def _shard(unit, source, categories, domain, out, prior, materials, limits, keep_catalog, prerequisite_seconds):
    started = time.monotonic()
    contexts = source.context
    syntax = contexts.syntax
    family, _, default_purpose, rule_type = _RULES[domain]
    rules = rule_type(source, world=_WORLD, exception_abi=ExceptionABI.EXCEPTION_WORLD)
    catalog = rules.catalog()
    parent_seconds = prerequisite_seconds + time.monotonic() - started
    relative = 'units/' + unit['key'] + '/' + domain + '.jsonl.gz'
    digest, decoded_bytes = sha256(), 0
    built, frontier, child_obligations, descriptors, suppressed = (Counter() for _ in range(5))
    condition_queue=deque();condition_seen=set();condition_built=Counter();request_bodies=Counter();stringized_values=0
    compilation_counts=Counter();compilation_edge_kinds=Counter();core_signatures=Counter()
    expressions, parent_expressions = 0, 0
    stream_path = Path(out) / relative

    def account(plan):
        nonlocal stringized_values
        flow=plan.compilation_flow
        compilation_counts.update(compilation_context_ports=len(flow.ports),
            compilation_context_edges=len(flow.edges),compilation_fact_uses=len(flow.uses))
        compilation_edge_kinds.update(edge.kind.value for edge in flow.edges)
        for step in plan.steps:
            instruction=step.instruction
            if type(instruction) in CORE_INSTRUCTIONS:
                core_signatures[type(instruction).__name__]+=1
            if type(instruction) is P.EvaluateChild:
                selected=rules.source.roles.locate(instruction.field)[1]
                conditioned=instruction.purpose is P.ProductionPurpose.CONDITION
                child_obligations[('CONDITION:' if conditioned else '')+syntax.nodes[selected.reference].constructor]+=1
                if conditioned and selected.reference not in condition_seen:
                    condition_seen.add(selected.reference);condition_queue.append(selected.reference)
            elif type(instruction) is P.BuildFunctionAnnotations:
                request_bodies['FUNCTION_ANNOTATIONS']+=1
            elif type(instruction) is P.BuildLazyExpression:
                request_bodies['LAZY_EXPRESSION']+=1
            elif type(instruction) is P.DefineAnnotationString:
                stringized_values+=1
            elif type(instruction) is P.BuildScopeAnnotations:
                request_bodies['MODULE_VARIABLE_ANNOTATIONS']+=1
            elif type(instruction) is P.BuildClassAnnotations:
                request_bodies['CLASS_VARIABLE_ANNOTATIONS']+=1

    with ExitStack() as stack:
        if prior is None:
            _durable_mkdir(stream_path.parent)
            destination = stack.enter_context(stream_path.open('xb'))
            stream = stack.enter_context(gzip.GzipFile(filename='', mode='wb', fileobj=destination, compresslevel=3, mtime=0))
        else:
            descriptor = prior['artifact']
            path = Path(materials) / relative
            if descriptor['path'] != relative or path.resolve() != path:
                raise ValueError('source expression shard has an unsafe identity/path')
            payload = path.read_bytes()
            if len(payload) != descriptor['bytes'] or sha256(payload).hexdigest() != descriptor['sha256']:
                raise ValueError('source expression compressed receipt changed')
            stream = stack.enter_context(gzip.GzipFile(fileobj=io.BytesIO(payload), mode='rb'))
        for occurrence, node in enumerate(syntax.nodes):
            is_expression = family in categories[node.constructor]
            owner = rules.descriptor_owner(occurrence)
            if not is_expression and owner is None:
                continue
            expressions += is_expression
            purpose = default_purpose
            if node.constructor in ('Name', 'Attribute', 'Subscript', 'List', 'Tuple'):
                purpose = {'Load': P.ProductionPurpose.VALUE, 'Store': P.ProductionPurpose.STORE,
                           'Del': P.ProductionPurpose.DELETE}[rules.source.roles.context(occurrence)]
            row = dict(occurrence=occurrence, constructor=node.constructor, purpose=purpose.value)
            if owner is not None:
                row.update(status='PARENT_SOURCE_DESCRIPTOR', purpose=P.ProductionPurpose.DESCRIPTOR.value,
                    owner=encode(owner), fields=[encode(P.SlotRef(occurrence, 'FIELD', i)) for i in range(len(node.fields))])
                descriptors[node.constructor] += 1
                parent_expressions += is_expression
            elif contexts.material.regions[contexts.entry_context(occurrence).region].eligibility == 'SUPPRESSED':
                context = contexts.entry_context(occurrence)
                region = contexts.material.regions[context.region]
                row.update(status='SOURCE_NOT_SCHEDULED', context=encode(context),
                    suppression=dict(reason=region.suppression_reason, origin=encode(region.suppression_origin)))
                suppressed[node.constructor] += 1
            else:
                try:
                    plan = rules.build(occurrence, purpose, max_steps=limits['steps'], max_regions=limits['regions'], max_visits=limits['visits'])
                    row.update(status='LOCAL_SOURCE_RULE', plan=encode(plan, record_tags=P.PRODUCTION_RECORD_TAGS))
                    built[node.constructor] += 1
                    account(plan)
                except SourceRuleError as error:
                    if error.code not in _FRONTIERS:
                        raise
                    row.update(status='RULE_FRONTIER', error=dict(code=error.code, path=error.path, detail=error.detail))
                    frontier[node.constructor + ':' + error.code] += 1
            raw = _json(row)
            digest.update(raw)
            decoded_bytes += len(raw)
            if prior is None:
                stream.write(raw)
            elif stream.readline(len(raw) + 1) != raw:
                raise ValueError('stored source rule/frontier differs from fresh original F2/F3 replay')
        # Context variants form a separate keyed domain after the original
        # occurrence partition. Expand only demanded conditions, once per key.
        condition_rules=SourceExpressionRules(rules.source,world=_WORLD,exception_abi=ExceptionABI.EXCEPTION_WORLD) if condition_queue else None
        while condition_queue:
            occurrence=condition_queue.popleft()
            plan=condition_rules.build(occurrence,P.ProductionPurpose.CONDITION,max_steps=limits['steps'],max_regions=limits['regions'],max_visits=limits['visits'])
            condition_built[syntax.nodes[occurrence].constructor]+=1
            account(plan)
            row=dict(occurrence=occurrence,constructor=syntax.nodes[occurrence].constructor,
                purpose=P.ProductionPurpose.CONDITION.value,status='LOCAL_CONTEXT_SOURCE_RULE',
                plan=encode(plan,record_tags=P.PRODUCTION_RECORD_TAGS))
            raw=_json(row);digest.update(raw);decoded_bytes+=len(raw)
            if prior is None:stream.write(raw)
            elif stream.readline(len(raw)+1)!=raw:
                raise ValueError('stored condition variant differs from fresh original F2/F3 replay')
        if prior is not None and stream.read(1):
            raise ValueError('extra source expression records after the original occurrence domain')
    if expressions != sum(built.values()) + sum(frontier.values()) + sum(suppressed.values()) + parent_expressions:
        raise ValueError('expression dispositions do not partition the original expression domain')
    if prior is None:
        with stream_path.open('rb') as stream:
            os.fsync(stream.fileno())
        _sync_directory(stream_path.parent)
        payload = stream_path.read_bytes()
        descriptor = dict(path=relative, bytes=len(payload), sha256=sha256(payload).hexdigest(),
            decoded_bytes=decoded_bytes, decoded_sha256=digest.hexdigest())
    elif descriptor['decoded_bytes'] != decoded_bytes or descriptor['decoded_sha256'] != digest.hexdigest():
        raise ValueError('decoded source expression receipt changed')
    result = dict(unit=unit['key'], path=unit['path'], input=encode(rules.source.input), artifact=descriptor,
        built=dict(sorted(built.items())), frontiers=dict(sorted(frontier.items())),
        parent_descriptors=dict(sorted(descriptors.items())), **{domain: expressions},
        suppressed_sources=dict(sorted(suppressed.items())), parent_owned_expressions=parent_expressions,
        child_interface_obligations=dict(sorted(child_obligations.items())),
        condition_recipes=sum(condition_built.values()),condition_constructors=dict(sorted(condition_built.items())),
        evaluation_request_templates=dict(request_bodies),stringized_annotation_values=stringized_values,
        compilation_contexts=dict(compilation_counts),compilation_edge_kinds=dict(sorted(compilation_edge_kinds.items())),
        core_signature_checks=dict(sorted(core_signatures.items())),
        parent_seconds=parent_seconds, seconds=prerequisite_seconds+time.monotonic()-started,
        catalog_sha256=sha256(_json(catalog)).hexdigest())
    if keep_catalog:
        result['_catalog'] = catalog
    return result


def _derive_unit(task):
    """Load the original F0–F3 inputs once for this pass and source unit."""
    model, unit, domains, limits, pins, keep_catalog = task
    label = 'SCAR source prerequisites: ' + unit['path']
    try:
        started = time.monotonic()
        source, categories = _source_index(model, unit, limits, pins)
        prerequisite_seconds = time.monotonic() - started
        rows = {}
        for domain, out, prior, materials in domains:
            label = 'SCAR ' + domain + ' source: ' + unit['path']
            rows[domain] = _shard(unit, source, categories, domain, out, prior, materials,
                                  limits, keep_catalog, prerequisite_seconds if not rows else 0.0)
        return rows, os.getpid(), prerequisite_seconds
    except ProductionError as error:
        error.add_note(label)
        raise
    except Exception as error:
        raise RuntimeError(label + '\n' + type(error).__qualname__ + ': ' + str(error)) from error


def _read_previous(domain, store, out, basis, pins, limits):
    if store.resolve() != store or out.is_relative_to(store) or store.is_relative_to(out):
        raise ValueError('fresh verification output must be independent of source materials')
    raw = (store / 'result.json').read_bytes()
    previous = json.loads(raw)
    schema = 'scar.source-' + _RULES[domain][1] + '-materials'
    if (previous.get('schema') != schema or previous.get('schema_version') != 4
            or previous.get('domain') != domain or previous.get('implementation_sha256') != pins
            or previous.get('input_basis') != basis or previous.get('limits') != limits):
        raise ValueError('previous rule material input/implementation/limit domain differs')
    prior = {row['unit']: row for row in previous['units']}
    if len(prior) != len(basis) or set(prior) != {unit['unit'] for unit in basis}:
        raise ValueError('previous source units do not match the selected rule domain')
    return previous, raw, prior


def _domain_report(domain, rows, basis, pins, model_sha, model_units, limits, jobs, store, previous, seconds):
    singular = _RULES[domain][1]
    catalog = rows[0].pop('_catalog')
    catalog_bytes = _json(catalog)
    if any(r['catalog_sha256'] != sha256(catalog_bytes).hexdigest() for r in rows):
        raise ValueError('source workers do not share the original rule/schema domain')
    if store is not None and (store / 'catalog.json').read_bytes() != catalog_bytes:
        raise ValueError('source rule catalog changed')
    counts = dict(units=len(rows), local_recipes=sum(sum(r['built'].values()) for r in rows),
        source_core_signature_checks=sum(sum(r['core_signature_checks'].values()) for r in rows),
        **{name:sum(r['compilation_contexts'].get(name,0) for r in rows) for name in
            ('compilation_context_ports','compilation_context_edges','compilation_fact_uses')},
        evaluation_request_templates=sum(sum(r['evaluation_request_templates'].values()) for r in rows),
        stringized_annotation_values=sum(r['stringized_annotation_values'] for r in rows),
        condition_recipes=sum(r['condition_recipes'] for r in rows),
        rule_frontiers=sum(sum(r['frontiers'].values()) for r in rows),
        **{singular+'_occurrences':sum(r[domain] for r in rows)},
        parent_descriptors=sum(sum(r['parent_descriptors'].values()) for r in rows),
        **{'parent_owned_'+domain:
            sum(r['parent_owned_expressions'] for r in rows)},
        suppressed_source_occurrences=sum(sum(r['suppressed_sources'].values()) for r in rows),
        child_interface_obligations=sum(sum(r['child_interface_obligations'].values()) for r in rows))
    if store is not None and previous['counts'] != counts:
        raise ValueError('stored source rule summary differs from fresh counters')
    report = dict(schema='scar.source-' + singular + '-materials', schema_version=4, domain=domain,
        status='LOCAL_RULES_WITH_FRONTIERS' if counts['rule_frontiers'] else 'LOCAL_RULES_VERIFIED',
        mode='FRESH_VERIFICATION' if store is not None else 'COLLECTION',
        input_basis=basis, implementation_sha256=pins, model_manifest_sha256=model_sha,
        selected_units=len(rows), model_units=model_units, limits=limits, jobs=jobs,
        catalog_sha256=sha256(catalog_bytes).hexdigest(), counts=counts, units=rows,
        seconds=seconds, local_source_rules_replayed=True,
        child_interfaces_proven=False, effects_proven=False, state_parameter_origin_proven=False,
        compilation_context_interfaces_checked=True, native_context_propagation_proven=False,
        core_signature_source_counts_checked=True,
        canonical_graph_generated=False, whole_program_graph_complete=False,
        target_executed=False, transformation_authorized=False)
    return report, catalog_bytes


def _collect_rule_shards(model, outputs, *, sources, materials, jobs, limits):
    if type(jobs) is not int or not 1 <= jobs <= 32 or any(type(v) is not int or v < 0 for v in limits.values()):
        raise ValueError('invalid source rule jobs/budgets')
    root = Path(model).absolute()
    outputs = {name: Path(path).absolute() for name, path in outputs.items()}
    materials = {name: Path(path).absolute() for name, path in materials.items()}
    if root.resolve() != root or any(out.resolve() != out or out.exists() or out.is_relative_to(root)
                                    or root.is_relative_to(out) for out in outputs.values()):
        raise ValueError('source rules require a canonical model and fresh independent outputs')
    manifest_bytes = (root / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get('schema') != 'scar.static-program' or manifest.get('schema_version') != 1:
        raise ValueError('expected an original static model')
    selected = {str(Path(path).absolute()) for path in sources}
    units = sorted((unit for unit in manifest['units'] if not selected or unit['path'] in selected),
                   key=lambda unit: unit['key'])
    if not units or selected and selected != {unit['path'] for unit in units} or len({unit['key'] for unit in units}) != len(units):
        raise ValueError('source selection must identify nonempty original source units exactly')
    basis = [_basis(unit) for unit in units]
    pins = _implementation()
    parents = {domain: _read_previous(domain, store, outputs[domain], basis, pins, limits)
               for domain, store in materials.items()}
    for out in outputs.values():
        _durable_mkdir(out)

    started = time.monotonic()
    tasks = []
    for index, unit in enumerate(units):
        domains = [(domain, out, parents[domain][2][unit['key']] if domain in parents else None,
                    materials.get(domain)) for domain, out in outputs.items()]
        tasks.append((root, unit, domains, limits, pins, index == 0))
    workers = min(jobs, len(tasks))
    if workers == 1:
        processed = list(map(_derive_unit, tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            processed = list(pool.map(_derive_unit, tasks))
    if (root / 'manifest.json').read_bytes() != manifest_bytes or _implementation() != pins:
        raise ValueError('model or implementation changed while processing source rules')
    if any((materials[domain] / 'result.json').read_bytes() != previous[1]
           for domain, previous in parents.items()):
        raise ValueError('source rule material manifest changed during fresh verification')

    model_sha = sha256(manifest_bytes).hexdigest()
    reports = {}
    for domain, out in outputs.items():
        rows = [shards[domain] for shards, _, _ in processed]
        report, catalog = _domain_report(domain, rows, basis, pins, model_sha, len(manifest['units']),
                                         limits, jobs, materials.get(domain),
                                         parents[domain][0] if domain in parents else None,
                                         time.monotonic() - started)
        _durable_bytes(out / 'catalog.json', catalog, replace=False)
        _durable_bytes(out / 'result.json', _json(report), replace=False)
        reports[domain] = report
    observation = dict(units=len(units), workers=workers, active_workers=len({pid for _, pid, _ in processed}),
                       prerequisite_loads=len(processed),
                       prerequisite_worker_seconds=sum(seconds for _, _, seconds in processed))
    return reports, observation


def collect_source_rules(model, out, *, domain, sources=(), materials=None, jobs=1,
                         max_nodes=1_000_000, max_tokens=2_000_000,
                         max_steps=100_000, max_regions=100_000, max_visits=1_000_000):
    if type(domain) is not str or domain not in _RULES:
        raise ValueError('expected the expression, statement or root source-rule domain')
    limits = dict(nodes=max_nodes, tokens=max_tokens, steps=max_steps, regions=max_regions, visits=max_visits)
    stores = {} if materials is None else {domain: materials}
    reports, _ = _collect_rule_shards(model, {domain: out}, sources=sources, materials=stores,
                                     jobs=jobs, limits=limits)
    return reports[domain]


def batch_main(argv):
    parser = argparse.ArgumentParser(description='Collect or independently replay all F4 source rule domains.')
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--source', action='append', default=[])
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args(argv)
    started = time.monotonic()
    mode = 'FRESH_VERIFICATION' if args.verify else 'COLLECTION'
    outputs = {domain: args.out / (domain + '-verified' if args.verify else domain) for domain in _RULES}
    inputs = {domain: args.out / domain for domain in _RULES} if args.verify else {}
    reports, observation = _collect_rule_shards(args.model, outputs, sources=args.source, materials=inputs,
                                                jobs=args.jobs, limits=_LIMITS)
    complete = all(report['status'] == 'LOCAL_RULES_VERIFIED' for report in reports.values())
    result = dict(schema='scar.source-rule-batch', schema_version=1, mode=mode,
                  status='LOCAL_RULES_VERIFIED' if complete else 'LOCAL_RULES_WITH_FRONTIERS',
                  domains={domain: file_reference(folder / 'result.json', reports[domain]['schema'])
                           for domain, folder in outputs.items()},
                  target_executed=False, seconds=time.monotonic() - started, **observation)
    report = args.out / ('rule-verification.json' if args.verify else 'rule-collection.json')
    _durable_bytes(report, _json(result), replace=False)
    print(json.dumps(result, indent=2))
    return 0


def collect_source_expressions(model, out, **options):
    return collect_source_rules(model, out, domain='expressions', **options)


def collect_source_statements(model, out, **options):
    return collect_source_rules(model, out, domain='statements', **options)


def collect_source_roots(model,out,**options):
    return collect_source_rules(model,out,domain='roots',**options)


def main(argv=None, *, domain='expressions'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--source', action='append', default=[])
    parser.add_argument('--materials', help='freshly replay and compare a prior material store')
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--max-nodes', type=int, default=1_000_000)
    parser.add_argument('--max-tokens', type=int, default=2_000_000)
    parser.add_argument('--max-steps', type=int, default=100_000)
    parser.add_argument('--max-regions', type=int, default=100_000)
    parser.add_argument('--max-visits', type=int, default=1_000_000)
    args = parser.parse_args(argv)
    report = collect_source_rules(args.model, args.out, domain=domain, sources=args.source, materials=args.materials,
        jobs=args.jobs, max_nodes=args.max_nodes, max_tokens=args.max_tokens,
        max_steps=args.max_steps, max_regions=args.max_regions, max_visits=args.max_visits)
    print(json.dumps({key: report[key] for key in ('status', 'mode', 'counts', 'seconds')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
