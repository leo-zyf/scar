from scar.analysis.candidates import detect
from scar.analysis.candidates import same_logical_version
from scar.analysis.candidates import consumer_counts, graph_liveness
from .graph_candidates import graph_candidates
from .graph_loops import graph_loop_candidates
from .inventory import summarize_action_inventory
from scar.analysis.static.candidates import dead_expression_candidates
from .rejection_audit import audit_candidates, audit_path, audit_report
from .topdown import DynamicRegionIndex, PlacementEvidence, RegionSummary, topdown_report
from .constants import (ConstantDefinition, ConstantUse, ModuleEffectSummary,
                        constant_candidates, constant_report, find_constant_uses)

__all__ = ["detect", "same_logical_version", "consumer_counts", "graph_liveness",
           "graph_candidates", "graph_loop_candidates", "dead_expression_candidates"]
__all__.append("summarize_action_inventory")
__all__ += ["audit_candidates", "audit_report", "audit_path"]
__all__ += ["RegionSummary", "PlacementEvidence", "DynamicRegionIndex", "topdown_report"]
__all__ += ["ConstantDefinition", "ConstantUse", "find_constant_uses",
            "ModuleEffectSummary", "constant_candidates", "constant_report"]
from .materialization import allocation_candidates, synchronization_candidates
from scar.analysis.static.candidates import loop_invariant_candidates
from .repetition import loop_invariant_candidates as dynamic_loop_invariant_candidates

__all__.append("synchronization_candidates")
__all__.append("allocation_candidates")
__all__.append("loop_invariant_candidates")
__all__.append("dynamic_loop_invariant_candidates")
