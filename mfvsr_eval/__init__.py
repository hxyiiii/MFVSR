"""MFVSR evaluation package."""

PROTOCOL_VERSION = "mfvsr_spatial_decoupled_v2"
SCHEMA_VERSION = "2.0"
METRIC_DEFINITION_VERSION = "2.0.3"
EVALUATOR_VERSION = "2.0.6"
COMPATIBLE_EVALUATOR_VERSIONS = ("2.0.3", "2.0.4", "2.0.5", "2.0.6")
EXECUTION_REVISION = "lean_profiles_sampled_tc_without_raft_v9"

REGIONS = ("overall", "bbox", "subtitle", "bbox_bg", "outside")
ATOMIC_REGIONS = ("subtitle", "bbox_bg", "outside")
