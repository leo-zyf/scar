"""Import GCC's early SSA CFG into the shared program graph.

The compiler-specific records remain evidence. Unknown effects and unsupported
control transfers are visible in the import report and gate rewrites.
"""

from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import json
from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory
from typing import Callable

from scar.analysis.target_graph import build_target_graph
from scar.artifacts import file_reference
from scar.ir.program.codec import SCHEMA as PROGRAM_SCHEMA, dumps
from scar.ir.program.ids import BlockID, LocationID, OperationID, RegionID, UseID, ValueID
from scar.ir.program.model import (
    AccessMode, Behavior, Block, BlockArgument, DataValue, ExitKind, ExitPort, LiteralOrigin,
    Operation, OperationResult, ProgramGraph, Region, StateAccess, StateLocation,
    StateVersionValue, Successor, Terminator, TerminatorKind, Use, UseRole,
    ValueKind, ValueType,
)
from scar.ir.program.providers import ProviderArtifact
from scar.ir.program.validation import require_valid

from ..contracts import GraphBundle, NativeInputs, native_bound_targets, verify_source_lock
from .clang import capture_unit, gnu_cfi_header, gnu_cfi_overlay


_FUNCTION = re.compile(r'^;; Function (.*?) \(([^,]*), funcdef_no=.*$', re.M)
_BLOCK = re.compile(r'^;;   basic block (\d+), loop depth (\d+).*$', re.M)
_JUMP = re.compile(r'goto <bb (\d+)>')
_SOURCE = re.compile(r'\[(?:[^\[\]\n]*:)?\d+:\d+(?: discrim \d+)?\] ?')
_QUOTED = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')
_SSA = re.compile(r'(?<![\w.])(?:[\w.$]*_\d+|\.MEM_\d+)(?:\(D\))?(?![\w.])')
_PHI = re.compile(r'# gimple_phi <([^,]+), (.*)>')
_PHI_ARG = re.compile(r'([^, ]+)\((\d+)\)')
_VUSE = re.compile(r'# VUSE <(\.MEM_\d+(?:\(D\))?)>')
_VDEF = re.compile(r'# (\.MEM_\d+) = VDEF <(\.MEM_\d+(?:\(D\))?)>')
_STMT = re.compile(r'^(gimple_\w+)\s+<(.*)>$')
_NAME = re.compile(r'^[\w.$]+$')
_VALUE_TYPE = 'native.ssa'
_CASE = re.compile(
    r'(?:default|case ([+-]?\d+)(?: \.\.\. ([+-]?\d+))?): (<[\w.$]+>|[\w.$]+)(?: \[[^\]]*\])?')
_PREDICATES = ('gimple_cond', 'native_switch_range')


def _inputs(raw):
    return tuple(dict.fromkeys(_SSA.findall(_QUOTED.sub('', raw))))


def _read_block(number, depth, raw):
    operations, phi, jumps, outgoing = [], {}, [], []
    memory = None
    in_successors = successors_captured = False
    for line in raw.splitlines():
        if line.startswith(';;    succ:'):
            in_successors = successors_captured = True
            line = line.partition(':')[2]
        elif in_successors and line.startswith(';;                '):
            line = line[2:]
        else:
            in_successors = False
        if in_successors:
            outgoing.extend(re.findall(r'(?<!\w)(\d+|EXIT)(?!\w)', line))
            continue
        code = _SOURCE.sub('', line).strip()
        if match := _PHI.fullmatch(code):
            name, bindings = match.groups()
            args = {predecessor: value for value, predecessor in _PHI_ARG.findall(bindings)}
            phi[name] = args
        elif match := _VDEF.fullmatch(code):
            after, before = match.groups()
            memory = before, after
        elif match := _VUSE.fullmatch(code):
            memory = match[1], None
        elif match := _STMT.fullmatch(code):
            opcode, args = match.groups()
            parts = args.split(', ', 3)
            result, call = None, None
            if opcode == 'gimple_assign' and len(parts) >= 3:
                result = parts[1] if _SSA.fullmatch(parts[1]) else None
                operands = args.split(', ', 2)[2] if result else args
            elif opcode == 'gimple_call' and len(parts) >= 2:
                call = parts[0]
                result = parts[1] if _SSA.fullmatch(parts[1]) else None
                target = call if _SSA.fullmatch(call) or call.startswith(('OBJ_TYPE_REF', '*')) else ''
                operands = ', '.join((target, *parts[2:]))
            elif opcode == 'gimple_return' and re.fullmatch(r'[+-]?\d+', args):
                result, operands = f'return:{number}', ''
            else:
                operands = args
            operations.append(dict(opcode=opcode, text=code, source=line.strip(),
                                   result=result, call=call, inputs=_inputs(operands), memory=memory))
            memory = None
        elif match := _JUMP.search(code):
            jumps.append(match[1])
    return dict(number=number, depth=int(depth), phi=phi, operations=operations,
                successors=outgoing, jumps=jumps,
                no_return=bool(successors_captured and not outgoing and operations
                               and operations[-1]['opcode'] == 'gimple_call'))


def read_ssa(raw, unit):
    functions = list(_FUNCTION.finditer(raw))
    for index, match in enumerate(functions):
        body = raw[match.end():functions[index + 1].start() if index + 1 < len(functions) else len(raw)]
        blocks = list(_BLOCK.finditer(body))
        if not blocks:
            continue
        version = sha256(body.encode()).hexdigest()
        yield dict(unit=unit, name=match[1], symbol=match[2], version=version, blocks=[
            _read_block(block[1], block[2], body[block.end():blocks[i + 1].start() if i + 1 < len(blocks) else len(body)])
            for i, block in enumerate(blocks)])


def capture_ssa(manifest, unit, destination):
    root = Path(manifest['source_root'])
    language = unit.get('language', manifest['language'])
    compiler = unit.get('compiler') or _SSA_COMPILERS[language]
    source = root / unit['path']
    destination = destination.absolute()
    with TemporaryDirectory(prefix='scar-native-ssa-') as scratch:
        command = [compiler, *unit['flags'], *unit.get('gcc_flags', []),
                   f'-fdump-tree-ssa-raw-blocks-lineno-vops={destination}',
                   f'-fdump-ipa-cgraph={destination.with_suffix(".cgraph")}',
                   '-c', str(source), '-o', str(Path(scratch) / 'unit.o')]
        result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=120)
        if result.returncode or not destination.is_file():
            raise ValueError(f'{unit["path"]}: {result.stderr}')
    return list(read_ssa(destination.read_text(errors='replace'), unit['path']))


def _function_identity(function):
    identity = (function['unit'], function['symbol'], function.get('version'))
    return sha256(json.dumps(identity).encode()).hexdigest()[:24]


def _expand_switches(blocks):
    if not any(op['opcode'] == 'gimple_switch' for block in blocks for op in block['operations']):
        return blocks
    labels = {op['text'].removeprefix('gimple_label <').removesuffix('>'): block['number']
              for block in blocks for op in block['operations'] if op['opcode'] == 'gimple_label'}
    expanded = []
    predecessors = defaultdict(list)
    for block in blocks:
        statements = block['operations']
        if not statements or statements[-1]['opcode'] != 'gimple_switch':
            expanded.append(block)
            continue
        switch = statements[-1]
        args = _STMT.fullmatch(switch['text'])[2].split(',')
        selector = args[0].strip()
        cases, default = [], None
        for argument in args[1:]:
            match = _CASE.fullmatch(argument.strip())
            if match is None or match[3] not in labels:
                raise ValueError('unbound integer switch case target')
            lower, upper, label = match.groups()
            target = labels[label]
            if lower is None:
                if default is not None:
                    raise ValueError('multiple switch default labels')
                default = target
            else:
                limits = int(lower), int(upper or lower)
                if limits[0] > limits[1] or any(a <= limits[1] and limits[0] <= b for (a, b), _ in cases):
                    raise ValueError('overlapping or reversed switch ranges')
                cases.append((limits, target))
        targets = {target for _, target in cases} | {default}
        if (not _SSA.fullmatch(selector) or switch['memory'] or default is None
                or targets != set(block['successors'])):
            raise ValueError('switch selector or cases differ from the compiler CFG')
        number = block['number']
        if not cases:
            expanded.append(dict(block, operations=[*statements[:-1], dict(switch, opcode='native_switch_default')]))
            continue
        tests = [number, *(f'{number}.case.{i}' for i in range(1, len(cases)))]
        for index, ((limits, target), test, otherwise) in enumerate(
                zip(cases, tests, [*tests[1:], default])):
            predicate = dict(switch, opcode='native_switch_range', inputs=(selector,), limits=limits)
            edges = [target, otherwise]
            expanded.append(dict(block, number=test, phi=block['phi'] if index == 0 else {},
                                 operations=[*statements[:-1], predicate] if index == 0 else [predicate],
                                 successors=edges, jumps=edges))
            predecessors[target, number].append(test)
        predecessors[default, number].append(tests[-1])
    return [dict(block, phi={
        name: {replacement: value for prior, value in alternatives.items()
               for replacement in predecessors.get((block['number'], prior), (prior,))}
        for name, alternatives in block['phi'].items()}) for block in expanded]


def _lower_function(function):
    identity = _function_identity(function)
    region = RegionID(identity)
    location = LocationID(identity + ':memory')
    exception_blocks = {row['number'] for row in function['blocks']
                        if any(op['opcode'] == 'gimple_resx' for op in row['operations'])}
    source_blocks = _expand_switches([
        dict(row, successors=[s for s in row['successors'] if s not in exception_blocks])
        for row in function['blocks'] if row['number'] not in exception_blocks])
    if not source_blocks:
        raise ValueError('no normal completion blocks')
    entry = source_blocks[0]['number']
    blocks = {item['number']: BlockID(f'{identity}:{item["number"]}') for item in source_blocks}
    definitions = {}
    values, operations, output_blocks, anchors = [], [], [], []
    supplied = defaultdict(list)

    def define(name, origin, kind=_VALUE_TYPE):
        value = ValueID(f'{identity}:{name}')
        created = (StateVersionValue(value, region, origin, location) if name.startswith('.MEM_')
                   else DataValue(value, region, origin, kind))
        if name in definitions:
            raise ValueError(f'duplicate definition of {name}')
        definitions[name] = value
        values.append(created)
        return value

    phi_ports = {row['number']: row['phi'] for row in source_blocks}
    defaults = set()
    for row in source_blocks:
        number = row['number']
        for name, alternatives in row['phi'].items():
            supplied[number].append(define(name, BlockArgument(blocks[number], len(supplied[number]))))
            defaults.update(value for value in alternatives.values() if value.endswith('(D)'))
        for index, op in enumerate(row['operations']):
            oid = OperationID(f'{identity}:{number}:{index}')
            defaults.update(name for name in op['inputs'] if name.endswith('(D)'))
            if op['result']:
                define(op['result'], OperationResult(oid, 0))
            if op['memory']:
                before, after = op['memory']
                if before.endswith('(D)'):
                    defaults.add(before)
                if after:
                    define(after, OperationResult(oid, int(bool(op['result']))))
            if op['opcode'] in _PREDICATES:
                define(f'cond:{number}', OperationResult(oid, 0), 'builtins.bool')
            for bound, value in enumerate(op.get('limits', ())):
                define(f'switch:{number}:{bound}', LiteralOrigin(value), 'builtins.int')
    for name in sorted(defaults):
        supplied[entry].append(define(name, BlockArgument(blocks[entry], len(supplied[entry]))))
    has_state = any(isinstance(value, StateVersionValue) for value in values)
    state_inputs = {}
    if has_state:
        for number, phi in phi_ports.items():
            incoming = [key for key in phi if key.startswith('.MEM_')]
            if number == entry:
                incoming += [key for key in defaults if key.startswith('.MEM_')]
            if len(incoming) > 1:
                raise ValueError('multiple virtual memory inputs')
            if not incoming:
                name = f'.MEM_in:{number}'
                supplied[number].append(define(name, BlockArgument(blocks[number], len(supplied[number]))))
                incoming.append(name)
            state_inputs[number] = incoming[0]

    exit_values = set()
    for row in source_blocks:
        number = row['number']
        oids = []
        condition = None
        returns = None
        state = state_inputs.get(number)

        def use(name, role=UseRole.DATA):
            if name not in definitions:
                raise ValueError(f'unresolved SSA definition {name}')
            link = Use(UseID(f'{oid.value}:{len(uses)}'), definitions[name], role)
            uses.append(link)
            return link.id

        for index, op in enumerate(row['operations']):
            oid = OperationID(f'{identity}:{number}:{index}')
            uses = []
            operands = [use(name) for name in op['inputs']]
            opcode = op['opcode']
            behaviors = [Behavior.COMPUTE] if opcode in ('gimple_assign', 'native_switch_range') else [Behavior.CONTROL]
            accesses, results = [], []
            if op['result']:
                results.append(definitions[op['result']])
            if op['memory']:
                _, after = op['memory']
                before_use = use(state, UseRole.STATE_CONTENT)
                mode = AccessMode.READ_WRITE if after else AccessMode.READ
                next_version = definitions[after] if after else None
                accesses.append(StateAccess(location, before_use, mode, next_version))
                behaviors.append(Behavior.READ)
                if next_version:
                    behaviors.append(Behavior.WRITE)
                    results.append(next_version)
                    state = after
            if opcode in _PREDICATES:
                condition = f'cond:{number}'
                results.append(definitions[condition])
            if opcode == 'native_switch_range':
                for bound in range(len(op['limits'])):
                    use(f'switch:{number}:{bound}')
            if opcode == 'gimple_return':
                if len(operands) > 1:
                    raise ValueError('unsupported return operand')
                returns = (op['result'],) if op['result'] else op['inputs']
                if not returns and not op['text'].endswith('<NULL>'):
                    raise ValueError('unsupported non-SSA return')
            if opcode in ('gimple_switch', 'gimple_resx', 'gimple_eh_dispatch', 'gimple_asm'):
                raise ValueError(f'unlowered control operation {opcode}')
            operations.append(Operation(oid, opcode, tuple(behaviors), tuple(uses), tuple(results),
                                        tuple(accesses), effects_complete=False))
            oids.append(oid)
            anchors.append(dict(operation=oid.wire, source=op['source'], call=op['call'],
                                unit=function['unit'], symbol=function['symbol'], region=region.wire))

        oid = OperationID(f'{identity}:{number}:end')
        uses = []
        edges = row['successors']
        local = [edge for edge in edges if edge != 'EXIT']
        if condition and len(local) == 2:
            if len(row['jumps']) != 2 or set(row['jumps']) != set(local):
                raise ValueError('unbound true/false targets')
            local = row['jumps']

        def successor(target):
            parameters = []
            for phi, sources in phi_ports[target].items():
                if phi.startswith('.MEM_'):
                    parameters.append(use(state, UseRole.STATE_CONTENT))
                    continue
                value = sources.get(number)
                if value is None:
                    raise ValueError(f'unresolved phi predecessor for {phi}')
                parameters.append(use(value))
            if has_state and state_inputs[target] not in phi_ports[target]:
                parameters.append(use(state, UseRole.STATE_CONTENT))
            return Successor(blocks[target], tuple(parameters))

        if len(local) == 1 and condition is None and returns is None and local == edges:
            term = Terminator(TerminatorKind.JUMP, tuple(map(successor, local)))
        elif len(local) == 2 and condition and 'EXIT' not in edges:
            term = Terminator(TerminatorKind.BRANCH, tuple(map(successor, local)), use(condition))
        elif edges == ['EXIT'] and returns is not None:
            returned = tuple(use(name) for name in returns)
            exit_values.add(len(returned))
            if has_state:
                returned += (use(state, UseRole.STATE_CONTENT),)
            term = Terminator(TerminatorKind.RETURN, exit='return', values=returned)
        elif not edges and row.get('no_return'):
            term = Terminator(TerminatorKind.NO_RETURN)
        else:
            raise ValueError(f'unlowered CFG terminator with successors {edges}')
        operations.append(Operation(oid, term.kind.value, (Behavior.CONTROL,), tuple(uses), terminator=term))
        output_blocks.append(Block(blocks[number], tuple(supplied[number]), (*oids, oid)))

    if len(exit_values) != 1:
        raise ValueError('inconsistent or open return interface')
    result_types = (ValueType(ValueKind.DATA, _VALUE_TYPE),) * exit_values.pop()
    if has_state:
        result_types += (ValueType(ValueKind.STATE, location=location),)
    ports = (ExitPort('return', ExitKind.NORMAL, result_types),)
    memory = (StateLocation(location, 'GCC virtual memory state'),) if has_state else ()
    graph = ProgramGraph(region, (Region(region, function['name'], blocks[entry], tuple(output_blocks), ports),),
                         tuple(operations), tuple(values), memory)
    require_valid(graph)
    return graph, anchors, sorted(exception_blocks)


def _write_graph(functions, out, entry):
    regions, operations, values, locations, anchors, frontiers = [], [], [], [], [], []
    excluded_control = []
    root = None
    model = None
    for function in functions:
        try:
            graph, mapping, exception_blocks = _lower_function(function)
        except (ValueError, KeyError) as error:
            frontiers.append(dict(unit=function['unit'], symbol=function['symbol'], reason=str(error)))
            continue
        function['region'] = graph.regions[0].id.wire
        if exception_blocks:
            excluded_control.append(dict(region=graph.root.wire, symbol=function['symbol'],
                                         blocks=exception_blocks, profile='NORMAL_COMPLETION'))
        if function['symbol'] == entry:
            if root is not None:
                raise ValueError(f'multiple native entry definitions for {entry}')
            root = graph.root
        regions.extend(graph.regions)
        operations.extend(graph.operations)
        values.extend(graph.values)
        locations.extend(graph.locations)
        anchors.extend(mapping)
    if root is not None:
        regions = [region if region.id == root else replace(region, parent=root) for region in regions]
        model = ProgramGraph(root, tuple(regions), tuple(operations), tuple(values), tuple(locations))
        out.write_text(dumps(model))
    return model, dict(entry=None if root is None else root.wire,
                regions=len(regions), operations=len(operations), values=len(values),
                locations=len(locations), anchors=anchors, frontiers=frontiers,
                excluded_control=excluded_control)


_IPA_HEADER = re.compile(r'^(.+)/(\d+) \((.*)\)(?: @.*)?$')
_IPA_REFERENCE = re.compile(r'/(\d+)(?=\s|$)')
_IPA_ADDRESS = re.compile(r'/(\d+) \(addr\)(?=\s|$)')
_SYMBOL = re.compile(r'^[\w.$]+$')
_SSA_TARGET = re.compile(r'^(?:[\w.$]*_\d+)(?:\(D\))?$')
_SITE = re.compile(r'^\[(?:(.+):)?(\d+):(\d+)(?: discrim \d+)?\]')
_ADDRESS_SITE = re.compile(r'gimple_assign <addr_expr, [^,]+, (\[[^\]]+\]) ')


def _ipa_calls(path):
    """Read the early table, ahead of inlining and compiler-created aliases."""
    table = Path(path).read_text(errors='replace').partition('Initial Symbol table:')[2]
    nodes = {}
    row = None
    for line in table.strip().partition('\n\n')[0].splitlines():
        if match := _IPA_HEADER.fullmatch(line):
            name, number, display = match.groups()
            row = nodes[number] = dict(name=name, display=display, calls=[], external=False,
                                       definition=False, addresses=[], function=False, public=False)
        elif row is not None:
            body = line.strip()
            if body.startswith('Calls:'):
                row['calls'] = _IPA_REFERENCE.findall(body)
            elif body.startswith('References:'):
                row['addresses'] = _IPA_ADDRESS.findall(body)
            elif body.startswith('Visibility:'):
                visibility = body.split()
                row['external'] = 'external' in visibility
                row['public'] = 'public' in visibility
            elif body.startswith('Type:'):
                kinds = body.split()
                row['definition'] = 'definition' in kinds
                row['function'] = 'function' in kinds
    definitions = defaultdict(list)
    for row in nodes.values():
        if row['definition']:
            definitions[row['name']].append(row)
    return nodes, definitions


def _elf_symbols(executable):
    symbols = defaultdict(set)
    if executable is not None:
        for flags in ([], ['-D']):
            result = subprocess.run(['nm', '-P', *flags, str(executable)], capture_output=True,
                                    text=True, timeout=30)
            for line in result.stdout.splitlines():
                fields = line.split()
                if len(fields) < 2 or len(fields[1]) != 1:
                    continue
                name, kind = fields[:2]
                name = name.partition('@')[0]
                if kind in ('T', 'W', 'I') and len(fields) >= 3:
                    symbols[name].add('DEFINITION')
                elif kind in ('t', 'i') and len(fields) >= 3:
                    symbols[name].add('LOCAL_DEFINITION')
                elif kind in ('U', 'w', 'v') and len(fields) == 2:
                    symbols[name].add('IMPORT')
    return symbols


def _compact(signature):
    return ''.join(signature.split())


def _clang_declarations(units):
    signatures, declarations, sites = defaultdict(set), {}, defaultdict(set)
    address_sites = defaultdict(set)
    for unit in units:
        if not isinstance(unit, dict):
            raise ValueError('expected Clang AST documents indexed by translation unit path')
        rows = unit['records']
        for record in rows:
            symbol = record.get('symbol')
            if not symbol or not record.get('usr'):
                continue
            declarations[record['usr']] = symbol
            qualified = [record['name']]
            parent = record.get('parent')
            while parent is not None:
                scope = rows[parent]
                if scope['kind'] in ('Namespace', 'ClassDecl', 'StructDecl', 'ClassTemplate') and scope['name']:
                    qualified.append(scope['name'])
                parent = scope.get('parent')
            result, suffix, parameters = record.get('type', '').partition('(')
            if suffix:
                display = f'{result} {"::".join(reversed(qualified))}({parameters}'
                signatures[(unit['path'], _compact(display))].add(symbol)
    for unit in units:
        for record in unit['records']:
            reference = record.get('reference') or {}
            symbol = reference.get('symbol') or declarations.get(reference.get('usr'))
            if record['kind'] == 'CallExpr' and symbol and reference.get('name'):
                sites[(unit['path'], record['path'], record['line'], reference['name'])].add(symbol)
            elif (record['kind'] == 'DeclRefExpr' and reference.get('kind') == 'FunctionDecl'
                  and symbol and _SYMBOL.fullmatch(symbol)):
                locations = [record]
                if record['parent'] is not None:
                    parent = unit['records'][record['parent']]
                    if parent['kind'] == 'UnaryOperator' and '(*' in parent['canonical_type']:
                        locations.append(parent)
                for location in locations:
                    key = (unit['path'], location['path'], location['line'], location['column'])
                    address_sites[key].add((reference['name'], symbol))
                # A header may lie outside the retained source root while the
                # compiler still records the exact referenced C declaration.
                result, suffix, parameters = record['canonical_type'].partition('(')
                if suffix and symbol == reference.get('name'):
                    signature = _compact(f'{result} {symbol}({parameters}')
                    signatures[(unit['path'], signature)].add(symbol)
    return signatures, sites, address_sites


def _spellings(node, source_names):
    yield node['name']
    yield node['display']
    for text in (node['display'], *source_names.get(node['name'], ())):
        function = text.partition('(')[0].strip()
        if function:
            yield function.split()[-1].rsplit('::', 1)[-1].split('<', 1)[0]


def _binding(unit, spelling, source, callers, nodes, names, signatures, sites):
    call_rows = {reference: nodes[reference] for caller in callers for reference in caller['calls']
                 if reference in nodes and spelling in _spellings(nodes[reference], names)}
    site_symbols = set()
    if match := _SITE.match(source or ''):
        path, line, _ = match.groups()
        location = path or unit
        if Path(location).is_absolute() and Path(unit).name == Path(location).name:
            location = unit
        site_symbols = sites.get((unit, location, int(line), spelling), set())
    candidates = defaultdict(set)
    unknown = False
    for node in call_rows.values():
        if _SYMBOL.fullmatch(node['name']):
            candidates[node['name']].add('gcc-ipa-call-edge')
        else:
            symbols = signatures.get((unit, _compact(node['display'])), ())
            unknown |= not bool(symbols)
            for symbol in symbols:
                candidates[symbol].update(('gcc-ipa-call-edge', 'clang-declaration-signature'))
    if site_symbols and candidates and site_symbols & candidates.keys():
        candidates = {symbol: evidence | {'clang-source-call'} for symbol, evidence in candidates.items()
                      if symbol in site_symbols}
    elif not call_rows and len(site_symbols) == 1:
        candidates = {symbol: {'clang-source-call'} for symbol in site_symbols}
    return candidates, unknown


def _callee(function):
    return dict(unit=function['unit'], name=function['name'], symbol=function['symbol'],
                region=function.get('region'))


def _linkage_binding(unit, linkage, origin, by_function, by_symbol, local_names, elf,
                     *, public_functions=None):
    binding = dict(linkage_symbol=linkage, origin=sorted(origin), elf=sorted(elf[linkage]))
    targets = by_symbol.get(linkage, ())
    same_unit = by_function.get((unit, linkage))
    if same_unit is not None and (targets == [same_unit] or linkage in local_names):
        targets = (same_unit,)
    elif public_functions is not None:
        targets = [target for target in targets if (target['unit'], linkage) in public_functions]
    if len(targets) == 1 and 'IMPORT' not in elf[linkage]:
        return dict(binding, status='BOUND', callee=_callee(targets[0]))
    if len(targets) > 1:
        return dict(binding, status='AMBIGUOUS', candidates=[_callee(target) for target in targets])
    if 'IMPORT' in elf[linkage]:
        return dict(binding, status='EXTERNAL')
    if linkage.startswith('__builtin_'):
        return dict(binding, status='BUILTIN')
    return dict(binding, status='UNBOUND_DIRECT')


def _address_target(operation, data, unit, source_root, sites, addressed):
    match = _ADDRESS_SITE.search(operation['source'])
    if match is None or (location := _SITE.fullmatch(match[1])) is None:
        return None
    path, line, column = location.groups()
    if path and Path(path).is_absolute():
        if source_root is None or not Path(path).is_relative_to(source_root):
            return None
        path = Path(path).relative_to(source_root).as_posix()
    key = unit, path or unit, int(line), int(column)
    targets = {symbol for name, symbol in sites.get(key, ())
               if symbol in addressed and data in (name, symbol)}
    if len(targets) == 1:
        return targets.pop()
    return None


def _pointer_definitions(function, callers, nodes, sites, source_root):
    """Index only pointer operations whose complete source and SSA are retained."""
    addressed = {nodes[key]['name'] for caller in callers for key in caller['addresses']
                 if key in nodes and nodes[key]['function']}
    if not addressed:
        return {}
    definitions = {}
    for block in function['blocks']:
        number = block['number']
        for name, inputs in block['phi'].items():
            if not name.startswith('.MEM_'):
                definitions[name] = dict(kind='PHI', block=number, predecessors=inputs)
        for index, op in enumerate(block['operations']):
            if op['opcode'] != 'gimple_assign' or op['memory'] or not op['result']:
                continue
            rhs = op['text'].removeprefix('gimple_assign <').removesuffix('>').split(', ')
            if len(rhs) != 5 or rhs[3:] != ['NULL', 'NULL']:
                continue
            operator, name, operand, *_ = rhs
            coordinate = dict(block=number, ordinal=index)
            if operator == 'ssa_name' and _SSA_TARGET.fullmatch(operand):
                definitions[name] = dict(kind='COPY', operand=operand, **coordinate)
            elif operator == 'addr_expr':
                target = _address_target(op, operand, function['unit'], source_root, sites, addressed)
                if target is not None:
                    definitions[name] = dict(kind='ADDRESS', linkage_symbol=target, **coordinate)
    return definitions


def _finite_pointer_flow(definitions, root):
    """Find all reaching address roots, including initialized phi cycles."""
    pending, reached, symbols, inputs_by_value = [root], set(), set(), defaultdict(set)
    initialized = []
    while pending:
        name = pending.pop()
        if name in reached:
            continue
        if (definition := definitions.get(name)) is None:
            return None
        reached.add(name)
        kind = definition['kind']
        if kind == 'ADDRESS':
            symbols.add(definition['linkage_symbol'])
            initialized.append(name)
            continue
        parents = ((definition['operand'],) if kind == 'COPY'
                   else tuple(definition['predecessors'].values()))
        if not parents:
            return None
        for parent in parents:
            inputs_by_value[parent].add(name)
        pending.extend(parents)
    proven = set(initialized)
    while initialized:
        for dependent in inputs_by_value[initialized.pop()] - proven:
            proven.add(dependent)
            initialized.append(dependent)
    if proven == reached:
        return sorted(symbols), dict(operand=root, definitions={name: definitions[name] for name in sorted(reached)})
    return None


def _open_call(row):
    if row['status'] == 'FINITE_TARGETS':
        return any(_open_call(target) for target in row['targets'])
    return (row['status'] in ('AMBIGUOUS', 'INDIRECT', 'UNBOUND_DIRECT')
            or row['status'] == 'BOUND' and row['callee']['region'] is None)


def link_calls(functions, anchors, *, entry='main', executable=None, clang_units=(), call_tables=(),
               source_root=None):
    """Produce source links, reachable functions and explicitly open call sites.

    A cross-unit edge needs compiler binding evidence. ELF symbol presence alone
    can confirm a linkage symbol, but cannot identify a source call expression.
    """
    by_symbol = defaultdict(list)
    by_function = {}
    names = defaultdict(set)
    for function in functions:
        key = function['unit'], function['symbol']
        if key in by_function:
            raise ValueError(f'duplicate compiler function identity: {key}')
        by_symbol[function['symbol']].append(function)
        by_function[key] = function
        names[function['symbol']].add(function['name'])
    tables = {unit: _ipa_calls(path) for unit, path in call_tables}
    signatures, sites, address_sites = _clang_declarations(tuple(clang_units))
    elf = _elf_symbols(executable)
    local_names = {unit: {node['name'] for node in nodes.values() if not node['external']}
                   for unit, (nodes, _) in tables.items()}
    public_functions = {(unit, node['name']) for unit, (nodes, _) in tables.items()
                        for node in nodes.values() if node['function'] and node['definition'] and node['public']}
    anchored_operation_ids = {row['operation'] for row in anchors if row.get('call')}
    calls = []
    called_by = defaultdict(set)
    for (unit, symbol), function in by_function.items():
        nodes, definitions = tables.get(unit, ({}, {}))
        callers = definitions.get(symbol, ())
        identity = _function_identity(function)
        pointer_definitions = None
        flows = {}
        local_definitions = local_names.get(unit, ())
        for block in function['blocks']:
            for index, operation in enumerate(block['operations']):
                if operation['opcode'] != 'gimple_call' or not operation.get('call'):
                    continue
                spelling = operation['call']
                oid = OperationID(f'{identity}:{block["number"]}:{index}').wire
                row = dict(caller=_callee(function), block=block['number'], ordinal=index,
                           operation=oid if oid in anchored_operation_ids else None,
                           source=operation.get('source'), spelling=spelling, callee=None,
                           linkage_symbol=None, origin=[])
                evidence, unknown = _binding(unit, spelling, operation.get('source'), callers,
                                             nodes, names, signatures, sites)
                if not evidence and not unknown and unit not in tables:
                    local = by_function.get((unit, spelling))
                    if local is not None and local['name'] == spelling and Path(unit).suffix == '.c':
                        evidence = {spelling: {'gcc-ssa-same-unit'}}
                if len(evidence) > 1:
                    row.update(status='AMBIGUOUS', candidates=sorted(evidence))
                elif len(evidence) == 1 and not unknown:
                    linkage, origin = next(iter(evidence.items()))
                    row.update(_linkage_binding(unit, linkage, origin, by_function, by_symbol,
                                                local_definitions, elf))
                elif spelling.startswith('__builtin_'):
                    row.update(status='BUILTIN', origin=['gcc-ssa-intrinsic'])
                elif not evidence and not unknown and (_SSA_TARGET.fullmatch(spelling)
                                                        or spelling.startswith('OBJ_TYPE_REF(')):
                    row.update(status='INDIRECT', origin=['gcc-ssa-indirect-operand'])
                    if row['operation'] and _SSA_TARGET.fullmatch(spelling):
                        if pointer_definitions is None:
                            pointer_definitions = _pointer_definitions(
                                function, callers, nodes, address_sites, source_root)
                        if spelling not in flows:
                            flows[spelling] = _finite_pointer_flow(pointer_definitions, spelling)
                        if flow := flows[spelling]:
                            symbols, row['target_flow'] = flow
                            origin = ('gcc-ssa-indirect-operand', 'gcc-ipa-address-reference',
                                      'clang-function-address-site')
                            targets = [_linkage_binding(unit, linkage, origin, by_function, by_symbol,
                                                        local_definitions, elf,
                                                        public_functions=public_functions) for linkage in symbols]
                            if len(targets) == 1 and targets[0]['status'] in ('BOUND', 'EXTERNAL'):
                                row.update(targets[0])
                            else:
                                row.update(status='FINITE_TARGETS', targets=targets, origin=sorted(origin))
                else:
                    row.update(status='UNBOUND_DIRECT')
                for target in native_bound_targets(row):
                    callee = target['callee']
                    called_by[(unit, symbol)].add((callee['unit'], callee['symbol']))
                calls.append(row)

    roots = by_symbol.get(entry, ())
    queue = deque([(roots[0]['unit'], entry)] if len(roots) == 1 else [])
    visited = set()
    while queue:
        key = queue.popleft()
        if key not in visited:
            visited.add(key)
            queue.extend(sorted(called_by[key] - visited))
    for row in calls:
        caller = row['caller']
        row['reachable'] = (caller['unit'], caller['symbol']) in visited
    frontiers = [row for row in calls if row['reachable'] and _open_call(row)]
    return dict(entry=_callee(roots[0]) if len(roots) == 1 else None,
                entry_candidates=[_callee(root) for root in roots],
                reachable=[_callee(by_function[key]) for key in sorted(visited)],
                calls=calls, frontiers=frontiers, elf_symbols_observed=bool(elf))


@dataclass(slots=True)
class CapturedUnits:
    """Normalized CFG records and separately referenced frontend evidence."""

    functions: list[dict] = field(default_factory=list)
    frontiers: list[dict] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    call_tables: list[tuple[str, Path]] = field(default_factory=list)
    declarations: list[dict] = field(default_factory=list)

    def extend(self, captured: 'CapturedUnits'):
        self.functions.extend(captured.functions)
        self.frontiers.extend(captured.frontiers)
        self.evidence.extend(captured.evidence)
        self.call_tables.extend(captured.call_tables)
        self.declarations.extend(captured.declarations)


@dataclass(frozen=True, slots=True)
class SourceAdapter:
    """A frontend for distinct languages with per-unit compiler settings."""

    name: str
    languages: tuple[str, ...]
    capture: Callable[[dict, list[dict], Path, int], CapturedUnits]
    version: int = 1

    def __post_init__(self):
        if (not self.languages or len(set(self.languages)) != len(self.languages)
                or type(self.version) is not int or self.version < 1):
            raise ValueError('frontend adapter requires distinct languages and a positive version')


def _capture_c_family(manifest, units, out, jobs):
    from shutil import which

    uses_fortran = any(unit.get('language', manifest['language']) == 'fortran' for unit in manifest['units'])
    cfi_header = gnu_cfi_header(manifest) if uses_fortran else None

    toolchains = {}
    for unit in units:
        language = unit.get('language', manifest['language'])
        compiler = unit.get('compiler') or _SSA_COMPILERS[language]
        if compiler in toolchains:
            continue
        executable = which(compiler)
        if executable is None and '/' in compiler:
            executable = str(Path(manifest['source_root']) / compiler)
        if executable is None:
            raise ValueError(f'SSA compiler unavailable: {compiler}')
        version = subprocess.check_output([executable, '--version'], cwd=manifest['source_root'],
                                          text=True, timeout=20).partition('\n')[0]
        toolchains[compiler] = dict(path=str(Path(executable).resolve()), version=version)

    def c_family(item):
        index, unit = item
        ssa = out / f'tu-{index}.ssa'
        functions = capture_ssa(manifest, unit, ssa)
        try:
            graph = capture_unit(manifest['source_root'], unit, cfi=cfi)
            (out / f'tu-{index}.clang.json').write_text(json.dumps(graph))
        except (ValueError, OSError) as error:
            return functions, ssa, None, dict(unit=unit['path'], reason=str(error))
        return functions, ssa, graph, None

    bundle = CapturedUnits()
    with gnu_cfi_overlay(cfi_header) as cfi, ThreadPoolExecutor(max_workers=jobs) as executor:
        for unit, (captured, ssa, ast, frontier) in zip(units, executor.map(c_family, enumerate(units))):
            language = unit.get('language', manifest['language'])
            compiler = unit.get('compiler') or _SSA_COMPILERS[language]
            calls = ssa.with_suffix('.cgraph')
            bundle.call_tables.append((unit['path'], calls))
            artifacts = dict(control=file_reference(ssa, 'gcc.ssa'),
                             calls=file_reference(calls, 'gcc.ipa-cgraph'))
            bundle.functions.extend(captured)
            if frontier is not None:
                bundle.frontiers.append(frontier)
            else:
                bundle.declarations.append(ast)
                artifacts['types'] = file_reference(ssa.with_suffix('.clang.json'), 'scar.clang-unit')
            bundle.evidence.append(dict(unit=unit['path'], toolchain=toolchains[compiler],
                                        gcc_flags=unit.get('gcc_flags', []), artifacts=artifacts))
    return bundle


def _capture_fortran(manifest, units, out, jobs):
    from .fortran import capture_fortran

    # Module files are built in source-list order for inter-unit USE statements.
    bundle = CapturedUnits()
    fortran = capture_fortran(dict(manifest, units=units), out)
    for unit in fortran['units']:
        bundle.functions.extend(read_ssa(Path(unit['raw']).read_text(), unit['path']))
        bundle.call_tables.append((unit['path'], Path(unit['call_table'])))
        bundle.evidence.append(dict(unit=unit['path'], compiler_version=fortran['version'], artifacts=dict(
            control=file_reference(unit['raw'], 'gcc.ssa'), types=file_reference(unit['semantics'], 'gfortran.semantics'),
            calls=file_reference(unit['call_table'], 'gcc.ipa-cgraph'))))
    (out / 'fortran.json').write_text(json.dumps(fortran))
    return bundle


_SSA_COMPILERS = {'c': 'gcc', 'c++': 'g++', 'fortran': 'gfortran'}
BUILTIN_ADAPTERS = (SourceAdapter('c-family', ('c', 'c++'), _capture_c_family),
                    SourceAdapter('gfortran', ('fortran',), _capture_fortran))


def capture_target(inputs: NativeInputs, out, jobs, *, adapters=BUILTIN_ADAPTERS):
    manifest = inputs.manifest
    providers = {adapter.name: adapter for adapter in adapters}
    if len(providers) != len(adapters) or any(not re.fullmatch(r'[\w-]+', name) for name in providers):
        raise ValueError('frontend adapter names must be unique path components')
    defaults = {}
    for adapter in adapters:
        for language in adapter.languages:
            defaults.setdefault(language, adapter)
    groups = defaultdict(list)
    for unit in manifest['units']:
        language = unit.get('language', manifest['language'])
        adapter = (providers.get(unit['source_adapter']) if 'source_adapter' in unit else defaults.get(language))
        if adapter is None or language not in adapter.languages:
            raise ValueError(f'no frontend adapter for {language}: {unit["path"]}')
        groups[adapter.name, language].append(unit)
    out.mkdir()
    result = CapturedUnits()
    for adapter in adapters:
        units = [unit for language in adapter.languages for unit in groups[adapter.name, language]]
        if units:
            directory = out if adapter in BUILTIN_ADAPTERS else out / adapter.name
            if directory != out:
                directory.mkdir()
            captured = adapter.capture(manifest, units, directory, jobs)
            expected = {unit['path']: unit for unit in units}
            reported = {row['unit'] for row in captured.evidence}
            if reported != expected.keys() or len(captured.evidence) != len(expected):
                raise ValueError(f'adapter {adapter.name} did not account for its compilation units')
            for row in captured.evidence:
                unit = expected[row['unit']]
                row.update(language=unit.get('language', manifest['language']), flags=unit['flags'],
                           standard=inputs.standards[unit['path']],
                           adapter=dict(name=adapter.name, version=adapter.version))
            result.extend(captured)
    return result


def build_graph(inputs: NativeInputs, directory, *, jobs, adapters=BUILTIN_ADAPTERS) -> GraphBundle:
    """Publish the program graph, platform facts and grounding evidence."""
    manifest, request = inputs.manifest, inputs.request
    hardware = directory / 'hardware'
    build_target_graph(hardware, paths=(manifest['source_root'], str(directory)))
    captured = capture_target(inputs, directory / 'frontend', jobs, adapters=adapters)
    verify_source_lock(manifest)
    program_path = directory / 'program.json'
    model, report = _write_graph(captured.functions, program_path, manifest['entry'])
    program = (file_reference(program_path, PROGRAM_SCHEMA) if report['entry'] is not None else None)
    target = file_reference(hardware / 'result.json', 'scar.target-hardware-graph')
    baseline = (manifest.get('baseline') or {}).get('argv') or [None]
    calls = dict(schema='scar.native-call-links', schema_version=1, **link_calls(
        captured.functions, report['anchors'], entry=manifest['entry'], executable=baseline[0],
        clang_units=captured.declarations, call_tables=captured.call_tables,
        source_root=Path(manifest['source_root'])))
    calls_path = directory / 'call-links.json'
    calls_path.write_text(json.dumps(calls) + '\n')
    call_receipt = file_reference(calls_path, 'scar.native-call-links')
    report.update(schema='scar.native-graph-evidence', schema_version=1, units=captured.evidence,
                  request=asdict(request), program=program, hardware=target, calls=call_receipt,
                  frontend_frontiers=captured.frontiers, call_frontiers=len(calls['frontiers']),
                  call_sites=len(calls['calls']),
                  call_links=sum(target['callee']['region'] is not None
                                 for row in calls['calls'] if row['operation'] is not None
                                 for target in native_bound_targets(row)),
                  reachable_regions=len(calls['reachable']))
    evidence_path = directory / ('graph-evidence.json' if program is not None else 'graph-report.json')
    (directory / 'compiler-cfg.json').write_text(json.dumps(captured.functions, indent=2, ensure_ascii=False) + '\n')
    evidence_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    evidence = file_reference(evidence_path, 'scar.native-graph-evidence')
    if program is not None:
        return GraphBundle.publish(directory, request=request, program=program, hardware=target,
                                   evidence=evidence, calls=call_receipt, entry_regions=(report['entry'],),
                                   model=model, report=report, call_document=calls)
    return GraphBundle(request, None if program is None else ProviderArtifact(**program),
                       ProviderArtifact(**target),
                       ProviderArtifact(**evidence),
                       ProviderArtifact(**call_receipt), report, model=model, call_document=calls)


def check_main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(prog='scar check-program',
                                     description='Validate the retained typed program graph.')
    parser.add_argument('graph')
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    from scar.ir.program.codec import _has_driver_capability, _has_no_body_capability, loads as load_program
    from scar.ir.program.model import LocalBindingValue, value_type
    from scar.ir.program.cleanup import CleanupRouteTerminator
    from scar.ir.program.resumption import SuspendTerminator
    from scar.ir.program.drivers import BodyDriveTerminator
    from scar.ir.record_codec import encode as encode_record
    from scar.ir.v2.common import ProofStatus

    source = Path(args.graph)
    report = {"schema": "scar.program-graph-check", "schema_version": 1,
              "input": str(source.resolve()), "scope": "declared graph structure and dependencies",
              "source_recovery": False, "transformation_authorized": False,
              "cleanup_structure_validated": False,
              "body_structure_validated": False,
              "body_driver_verified": False,
              "body_source_verified": False,
              "body_runtime_verified": False,
              "body_exclusivity_verified": False,
              "driver_structure_validated": False,
              "driver_lifecycle_verified": False,
              "body_publication_structure_validated": False,
              "no_body_structure_validated": False,
              "session_release_structure_validated": False,
              "conditional_lifecycle_facts_derived": False,
              "conditional_lifecycle_scopes": []}
    if args.out and Path(args.out).resolve() == source.resolve():
        report.update(status="INVALID", errors=["report path must differ from input graph"])
        print(json.dumps(report, ensure_ascii=False))
        return 2
    try:
        graph = load_program(source.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        report.update(status="INVALID", errors=[str(exc)])
    else:
        report.update(status="WELL_FORMED", errors=[], counts={
            "regions": len(graph.regions), "blocks": sum(len(r.blocks) for r in graph.regions),
            "operations": len(graph.operations), "values": len(graph.values),
            "locations": len(graph.locations), "uses": sum(len(op.uses) for op in graph.operations),
            "value_kinds": dict(Counter(
                "LOCAL_BINDING" if isinstance(value, LocalBindingValue)
                else value_type(value).kind.value for value in graph.values)),
            "declared_incomplete_effects": sum(not op.effects_complete for op in graph.operations),
        })
        if any((graph.resources, graph.resource_requirements, graph.resource_uses,
                graph.resource_locations)):
            report["counts"].update(
                resources=len(graph.resources),
                resource_requirements=len(graph.resource_requirements),
                resource_uses=len(graph.resource_uses),
                resource_locations=len(graph.resource_locations))
            report.update(resource_realization_verified=False, readiness_verified=False)
        if graph.environments:
            report["counts"].update(
                environments=len(graph.environments),
                environment_members=sum(len(item.members) for item in graph.environments),
                function_entries=len(graph.function_entries),
                python_parameters=sum(len(item.parameters) for item in graph.function_entries))
            report.update(environment_realization_verified=False,
                          environment_extraction_verified=False,
                          python_signature_binding_verified=False,
                          runtime_callable_verified=False)
        if graph.control_scopes:
            report["counts"].update(
                control_scopes=len(graph.control_scopes), control_ports=len(graph.control_ports),
                exception_packets=len(graph.exception_packets),
                pending_completions=len(graph.pending_completions))
            report.update(lexical_source_verified=False, cleanup_unwinding_verified=False,
                          exception_context_verified=False,
                          pending_runtime_realization_verified=False)
        if graph.handled_context_domains:
            report["counts"].update(
                cleanup_interfaces=len(graph.cleanup_interfaces),
                cleanup_actions=sum(len(rows) for rows in (
                    graph.cleanup_enters, graph.cleanup_resumes,
                    graph.cleanup_replacements, graph.handled_context_reads)),
                cleanup_routes=sum(isinstance(op.terminator, CleanupRouteTerminator)
                                   for op in graph.operations))
            report.update(cleanup_structure_validated=True,
                          cleanup_structure_scope="declared static interfaces, CFG routes and state flow",
                          cleanup_unwinding_verified=False,
                          exception_context_verified=False,
                          pending_runtime_realization_verified=False)
        if graph.provider_envelopes:
            report["counts"].update(
                provider_envelopes=len(graph.provider_envelopes),
                provider_regions=len(graph.provider_regions),
                provider_bindings=len(graph.provider_bindings))
            report.update(provider_artifact_replay_verified=False,
                          provider_conditions_verified=False,
                          provider_source_correspondence_verified=False,
                          cross_representation_equivalence_verified=False,
                          provider_runtime_correspondence_verified=False)
        if graph.native_boundaries:
            report["counts"].update(native_boundaries=len(graph.native_boundaries),
                                    boundary_relations=len(graph.boundary_relations))
            report.update(native_signature_replay_verified=False,
                          native_tree_mapping_verified=False,
                          native_state_binding_verified=False,
                          boundary_relations_verified=False)
        if graph.resumable_bodies:
            report["counts"].update(
                resumable_bodies=len(graph.resumable_bodies),
                suspensions=sum(type(operation.terminator) is SuspendTerminator
                                for operation in graph.operations),
                frame_context_actions=(len(graph.frame_context_initials)
                                       + len(graph.frame_context_attaches)
                                       + len(graph.frame_context_detaches)))
            report["body_structure_validated"] = True
        if _has_driver_capability(graph):
            report["counts"].update(
                resumable_creates=len(graph.resumable_creates),
                raw_driver_requests=len(graph.raw_driver_requests),
                async_request_creates=len(graph.async_request_creates),
                request_association_reads=len(graph.request_association_reads),
                driver_interfaces=len(graph.driver_interfaces),
                final_body_gates=len(graph.final_body_gates),
                body_drives=len(graph.body_drives),
                body_drive_outcomes=sum(
                    len(operation.terminator.successors)
                    for operation in graph.operations
                    if type(operation.terminator) is BodyDriveTerminator))
            report.update(driver_structure_validated=True,
                          driver_structure_scope="declared static driver ports, CFG and current state flow",
                          driver_lifecycle_verified=False)
        if graph.driver_lifecycles:
            interfaces = {row.region: row for row in graph.driver_interfaces}
            operation_regions = {operation: region.id for region in graph.regions
                                 for block in region.blocks for operation in block.operations}
            tables = ("body_outcome_commits", "created_close_tests", "no_body_finishes",
                      "session_updates", "terminal_frame_releases")
            report["conditional_lifecycle_scopes"] = [
                {"region": scope.region.wire,
                 "rule_profile": interfaces[scope.region].rule.profile.value,
                 "implementation": encode_record(interfaces[scope.region].rule.implementation),
                 "applicability": ProofStatus.UNKNOWN.value,
                 "conditional_facts_derived": True,
                 "declared_capabilities": {
                     name: sum(operation_regions[row.operation] == scope.region
                               for row in getattr(graph, name)) for name in tables}}
                for scope in graph.driver_lifecycles]
            report["conditional_lifecycle_facts_derived"] = True
            report["counts"].update(
                driver_lifecycles=len(graph.driver_lifecycles),
                body_outcome_commits=len(graph.body_outcome_commits))
            report.update(body_publication_structure_validated=True,
                          body_publication_structure_scope="declared body outcome, owner and publication state links",
                          driver_lifecycle_verified=False)
        if graph.driver_lifecycles and _has_no_body_capability(graph):
            report["counts"].update(created_close_tests=len(graph.created_close_tests),
                                    no_body_finishes=len(graph.no_body_finishes))
            report.update(no_body_structure_validated=True,
                          no_body_structure_scope="declared predicate, real branch edges and optional no-body finish links",
                          driver_lifecycle_verified=False)
        if graph.session_updates or graph.terminal_frame_releases:
            report["counts"].update(session_updates=len(graph.session_updates),
                                    terminal_frame_releases=len(graph.terminal_frame_releases))
            report.update(session_release_structure_validated=True,
                          session_release_structure_scope="declared post states and conditional terminal owner history; no native applicability or resource-release guarantee",
                          conditional_lifecycle_facts_derived=True,
                          driver_lifecycle_verified=False)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report["status"] == "WELL_FORMED" else 2
