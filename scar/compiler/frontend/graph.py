"""Import GCC's early SSA CFG into the shared program graph.

The compiler-specific records remain evidence. Unknown effects and unsupported
control transfers are visible in the import report and gate rewrites.
"""

from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory

from scar.ir.program.codec import dumps
from scar.ir.program.ids import BlockID, LocationID, OperationID, RegionID, UseID, ValueID
from scar.ir.program.model import (
    AccessMode, Behavior, Block, BlockArgument, DataValue, ExitKind, ExitPort,
    Operation, OperationResult, ProgramGraph, Region, StateAccess, StateLocation,
    StateVersionValue, Successor, Terminator, TerminatorKind, Use, UseRole,
    ValueKind, ValueType,
)
from scar.ir.program.validation import require_valid

from .clang import capture_unit


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


def _inputs(raw):
    return tuple(dict.fromkeys(_SSA.findall(_QUOTED.sub('', raw))))


def _read_block(number, depth, raw):
    operations, phi, jumps, outgoing = [], {}, [], []
    memory = None
    in_successors = False
    for line in raw.splitlines():
        if line.startswith(';;    succ:'):
            in_successors = True
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
                successors=outgoing, jumps=jumps)


def read_ssa(raw, unit):
    functions = list(_FUNCTION.finditer(raw))
    for index, match in enumerate(functions):
        body = raw[match.end():functions[index + 1].start() if index + 1 < len(functions) else len(raw)]
        blocks = list(_BLOCK.finditer(body))
        if not blocks:
            continue
        yield dict(unit=unit, name=match[1], symbol=match[2], blocks=[
            _read_block(block[1], block[2], body[block.end():blocks[i + 1].start() if i + 1 < len(blocks) else len(body)])
            for i, block in enumerate(blocks)])


def capture_ssa(manifest, unit, destination):
    root = Path(manifest['source_root'])
    language = unit.get('language', manifest['language'])
    compiler = unit.get('compiler', {'c': 'gcc', 'fortran': 'gfortran', 'c++': 'g++'}[language])
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


def _lower_function(function):
    identity = sha256((function['unit'] + ':' + function['symbol']).encode()).hexdigest()[:24]
    region = RegionID(identity)
    location = LocationID(identity + ':memory')
    exception_blocks = {row['number'] for row in function['blocks']
                        if any(op['opcode'] == 'gimple_resx' for op in row['operations'])}
    source_blocks = [
        dict(row, successors=[s for s in row['successors'] if s not in exception_blocks])
        for row in function['blocks'] if row['number'] not in exception_blocks]
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
            if op['opcode'] == 'gimple_cond':
                define(f'cond:{number}', OperationResult(oid, 0), 'builtins.bool')
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
            behaviors = [Behavior.COMPUTE] if opcode == 'gimple_assign' else [Behavior.CONTROL]
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
            if opcode == 'gimple_cond':
                condition = f'cond:{number}'
                results.append(definitions[condition])
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
            anchors.append(dict(operation=oid.wire, source=op['source'], call=op['call']))

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


def write_graph(functions, out, entry='main'):
    regions, operations, values, locations, anchors, frontiers = [], [], [], [], [], []
    excluded_control = []
    root = None
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
        graph = ProgramGraph(root, tuple(regions), tuple(operations), tuple(values), tuple(locations))
        require_valid(graph)
        out.write_text(dumps(graph))
    return dict(entry=None if root is None else root.wire,
                regions=len(regions), operations=len(operations), values=len(values),
                locations=len(locations), anchors=anchors, frontiers=frontiers,
                excluded_control=excluded_control)


_IPA_HEADER = re.compile(r'^(.+)/(\d+) \((.*)\)(?: @.*)?$')
_IPA_REFERENCE = re.compile(r'/(\d+)(?=\s|$)')
_SYMBOL = re.compile(r'^[\w.$]+$')
_SSA_TARGET = re.compile(r'^(?:[\w.$]*_\d+)(?:\(D\))?$')
_SITE = re.compile(r'^\[(?:(.+):)?(\d+):(\d+)(?: discrim \d+)?\]')


def _ipa_calls(path):
    """Read the early table, ahead of inlining and compiler-created aliases."""
    table = Path(path).read_text(errors='replace').partition('Initial Symbol table:')[2]
    nodes = {}
    row = None
    for line in table.strip().partition('\n\n')[0].splitlines():
        if match := _IPA_HEADER.fullmatch(line):
            name, number, display = match.groups()
            row = nodes[number] = dict(name=name, display=display, calls=[], external=False,
                                       definition=False, indirect=False)
        elif row is not None:
            body = line.strip()
            if body.startswith('Calls:'):
                row['calls'] = _IPA_REFERENCE.findall(body)
            elif body.startswith('Visibility:'):
                row['external'] = 'external' in body.split()
            elif body.startswith('Type:'):
                row['definition'] = 'definition' in body.split()
            elif body.startswith('Indirect call'):
                row['indirect'] = True
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
            if record['kind'] != 'CallExpr':
                continue
            reference = record.get('reference') or {}
            symbol = reference.get('symbol') or declarations.get(reference.get('usr'))
            if symbol and reference.get('name'):
                sites[(unit['path'], record['path'], record['line'], reference['name'])].add(symbol)
    return signatures, sites


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


def link_calls(functions, anchors, *, entry='main', executable=None, clang_units=(), call_tables=()):
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
    signatures, sites = _clang_declarations(tuple(clang_units))
    elf = _elf_symbols(executable)
    anchored_operation_ids = {row['operation'] for row in anchors if row.get('call')}
    calls = []
    called_by = defaultdict(set)
    for (unit, symbol), function in by_function.items():
        nodes, definitions = tables.get(unit, ({}, {}))
        callers = definitions.get(symbol, ())
        identity = sha256((unit + ':' + symbol).encode()).hexdigest()[:24]
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
                    row.update(linkage_symbol=linkage, origin=sorted(origin), elf=sorted(elf[linkage]))
                    targets = by_symbol.get(linkage, ())
                    same_unit = by_function.get((unit, linkage))
                    table_target = [node for node in nodes.values() if node['name'] == linkage]
                    if same_unit is not None and (targets == [same_unit]
                                                  or any(not node['external'] for node in table_target)):
                        targets = (same_unit,)
                    if len(targets) == 1 and 'IMPORT' not in elf[linkage]:
                        target = targets[0]
                        row.update(status='BOUND', callee=_callee(target))
                        called_by[(unit, symbol)].add((target['unit'], target['symbol']))
                    elif len(targets) > 1:
                        row.update(status='AMBIGUOUS', candidates=[_callee(target) for target in targets])
                    elif 'IMPORT' in elf[linkage]:
                        row.update(status='EXTERNAL')
                    elif linkage.startswith('__builtin_'):
                        row.update(status='BUILTIN')
                    else:
                        row.update(status='UNBOUND_DIRECT')
                elif spelling.startswith('__builtin_'):
                    row.update(status='BUILTIN', origin=['gcc-ssa-intrinsic'])
                elif not evidence and not unknown and (_SSA_TARGET.fullmatch(spelling)
                                                        or spelling.startswith('OBJ_TYPE_REF(')):
                    row.update(status='INDIRECT', origin=['gcc-ssa-indirect-operand'])
                else:
                    row.update(status='UNBOUND_DIRECT')
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
    frontiers = [row for row in calls if row['reachable'] and (
        row['status'] in ('AMBIGUOUS', 'INDIRECT', 'UNBOUND_DIRECT')
        or row['status'] == 'BOUND' and not row['callee']['region'])]
    return dict(entry=_callee(roots[0]) if len(roots) == 1 else None,
                entry_candidates=[_callee(root) for root in roots],
                reachable=[_callee(by_function[key]) for key in sorted(visited)],
                calls=calls, frontiers=frontiers, elf_symbols_observed=bool(elf))


def capture_target(manifest, out, jobs):
    out.mkdir()
    profiles = {'c': [], 'c++': [], 'fortran': []}
    for unit in manifest['units']:
        profiles[unit.get('language', manifest['language'])].append(unit)

    def c_family(item):
        index, unit = item
        ssa = out / f'tu-{index}.ssa'
        functions = capture_ssa(manifest, unit, ssa)
        try:
            graph = capture_unit(manifest['source_root'], unit)
            (out / f'tu-{index}.clang.json').write_text(json.dumps(graph))
        except (ValueError, OSError) as error:
            return functions, dict(unit=unit['path'], reason=str(error))
        return functions, None

    functions, frontiers = [], []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        items = enumerate((*profiles['c'], *profiles['c++']))
        for captured, frontier in executor.map(c_family, items):
            functions.extend(captured)
            if frontier is not None:
                frontiers.append(frontier)
    if profiles['fortran']:
        from .fortran import capture_fortran
        fortran = capture_fortran(dict(manifest, units=profiles['fortran']), out)
        for unit in fortran['units']:
            functions.extend(read_ssa(Path(unit['raw']).read_text(), unit['path']))
        (out / 'fortran.json').write_text(json.dumps(fortran))
    return functions, frontiers
