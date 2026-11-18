"""Seal a private dead-result projection as a conditional source rewrite."""

import argparse
from base64 import b64encode
import json
import marshal
from pathlib import Path

from scar.analysis.csr_dead_item_scope import find_csr_dead_item_sites
from scar.analysis.dead_result_item import seal_dead_item
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.runtime.sparse_parallel_entry import SCHEMA as SELECTION
from scar.trace.code_snapshot import code_key


SCHEMA = 'scar.dead-result-item-plan'


def seal_site(site, optimize):
    source = read_artifact(site['source'], json_document=False)
    return seal_dead_item(source, site['filename'], caller_key=site['caller_key'], callee_key=site['callee_key'],
        call_span=site['call_span'], producer_span=site['producer_span'], optimize=optimize)


def provider_contract():
    import numpy as np
    from numpy._core import _multiarray_umath as native
    from numpy.lib import _arraysetops_impl as arraysets

    producer = np.setdiff1d
    if producer is not arraysets.setdiff1d or np.arange is not native.arange:
        raise ValueError('standard NumPy complement and range providers required')
    return dict(version=np.__version__, producer_code_key=code_key(producer.__wrapped__.__code__),
        source=artifact_reference(arraysets.__file__, 'scar.python-source'),
        native=artifact_reference(native.__file__, 'scar.provider-binary'))


def make_plan(reference):
    selection = read_artifact(reference)
    optimize = selection['context']['interpreter']['optimize']
    found = find_csr_dead_item_sites(selection)
    choices = []
    for site in found['candidates']:
        try:
            facts, caller, callee = seal_site(site, optimize)
        except ValueError:
            continue
        choices.append((site, facts, caller, callee))
    if len(choices) != 1:
        raise ValueError('one proved private dead result item is required')
    site, facts, caller, callee = choices[0]
    sealed = {key: b64encode(marshal.dumps(code)).decode('ascii') for key, code in (
        ('caller', caller), ('callee', callee))}
    return dict(schema=SCHEMA, schema_version=2, source_selection=reference, site=site, facts=facts,
        sealed=sealed,
        interpreter=selection['context']['interpreter'], provider=provider_contract(),
        rewrite=dict(operation='ERASE_DEAD_RETURN_ITEM_PRODUCER', retained_first_call=True,
            existing_array_for_dead_slot=True, scope='ONLY_SEALED_CALLSITE',
            condition='EXCLUSIVE_INPUTS_AND_NORMAL_COMPLETION',
            frontiers=[reason for reason in found['frontiers']
                if reason != 'RETURN_ITEM_LIVENESS_UNPROVEN_BY_CALL_OBSERVATION'],
            downstream_item_dead=True, allocation_failure_equivalence=False),
        auto_apply=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    source = artifact_reference(args.selection, SELECTION)
    document = make_plan(source)
    with args.out.open('x') as output:
        json.dump(document, output, indent=2, sort_keys=True)
        output.write('\n')
    print(json.dumps(artifact_reference(args.out, SCHEMA)))


if __name__ == '__main__':
    main()
