from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RmaEtSpec:
    """Specification of the privileged environment-factor vector e_t for Unitree H12 RMA.

    Leg-only variant (recommended starting point):
    e_t = [payload(3), leg_strength(12), friction(1), terrain(3)]  -> 19 dims

    Ordering is fixed so that:
    - the env can pack e_t consistently
    - the encoder μ(e_t) can be reused across training/inference code
    - sim2sim export can carry the same latent dimension

    Indices:
      0: payload_mass_add_kg
      1: payload_com_offset_x_m
      2: payload_com_offset_y_m

      3..14: leg_motor_strength_scale (12 values)

      15: ground_friction_coeff

      16: terrain_slope_x (forward slope proxy)
      17: terrain_slope_y (lateral slope proxy)
      18: terrain_height_at_base_m

    Notes:
    - payload entries should be quasi-static per episode (sample once at reset/startup).
    - strength scaling should be quasi-static per episode.
    - friction should be quasi-static per episode (or per env instance).
    - terrain slope/height are included here as *coarse/static* descriptors; if you want per-step local geometry,
      treat that separately (geom encoder) rather than in e_t.
    """

    payload_dim: int = 3
    leg_strength_dim: int = 12
    friction_dim: int = 1
    terrain_dim: int = 3

    @property
    def dim(self) -> int:
        return self.payload_dim + self.leg_strength_dim + self.friction_dim + self.terrain_dim

    @property
    def payload_slice(self) -> slice:
        return slice(0, 3)

    @property
    def leg_strength_slice(self) -> slice:
        return slice(3, 3 + self.leg_strength_dim)

    @property
    def friction_slice(self) -> slice:
        start = 3 + self.leg_strength_dim
        return slice(start, start + 1)

    @property
    def terrain_slice(self) -> slice:
        start = 3 + self.leg_strength_dim + 1
        return slice(start, start + self.terrain_dim)


# Default spec used by Unitree-H12-Walk-RMA-v0.
DEFAULT_ET_SPEC = RmaEtSpec()


# Leg joint ordering used for the 12-dim leg strength portion of e_t.
# This must match the description in RmaEtSpec and should remain stable across training/export.
LEG_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_yaw_joint",
    "left_hip_roll_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_yaw_joint",
    "right_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)
