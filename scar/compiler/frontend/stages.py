"""Isolated source-analysis workers used by the formal Python frontend."""

from importlib import import_module
import sys


_WORKERS = {
    'static': 'scar.analysis.static_program',
    'source-bytes': 'scar.analysis.source_materials',
    'source-tokens': 'scar.analysis.source_lexing',
    'source-syntax': 'scar.analysis.source_parsing',
    'source-symbols': 'scar.analysis.source_symbol_materials',
    'source-roles': 'scar.analysis.source_role_materials',
    'source-contexts': 'scar.analysis.source_context_materials',
    'source-declarations': 'scar.analysis.source_declaration_materials',
    'source-bindings': 'scar.analysis.source_binding_materials',
    'compiler-topology': 'scar.analysis.compiler_topology_materials',
    'compiler-symbols': 'scar.analysis.compiler_symbol_materials',
    'program-types': 'scar.analysis.program_type_materials',
    'source-anchors': 'scar.analysis.source_anchor_materials',
}
_DOMAINS = {'source-expressions', 'source-statements', 'source-roots'}


def main(argv=None):
    stage, *arguments = sys.argv[1:] if argv is None else argv
    if stage in _DOMAINS:
        from scar.analysis.source_expression_materials import main as materials
        return materials(arguments, domain=stage.removeprefix('source-'))
    if stage not in _WORKERS:
        raise SystemExit(f'unknown frontend stage {stage}')
    return import_module(_WORKERS[stage]).main(arguments)


if __name__ == '__main__':
    raise SystemExit(main())
