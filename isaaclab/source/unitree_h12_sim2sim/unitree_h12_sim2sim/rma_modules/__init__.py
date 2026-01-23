"""Reusable neural modules for RMA (Rapid Motor Adaptation).

Phase 1 (privileged):
- `EnvFactorEncoder` maps sim-only env factors e_t -> z_t.

Phase 2 (adaptation):
- `AdaptationModule` maps history of observations/actions -> \hat{z}_t.
"""

from .env_factor_encoder import EnvFactorEncoder
from .adaptation_module import AdaptationModule
from .env_factor_spec import DEFAULT_ET_SPEC, RmaEtSpec

__all__ = ["EnvFactorEncoder", "AdaptationModule", "RmaEtSpec", "DEFAULT_ET_SPEC"]
