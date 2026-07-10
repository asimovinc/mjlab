"""Asimov 1 robot constants.

Asimov 1 has 23 actuated DOFs:
- 12 leg joints: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
- 1 waist joint: waist_yaw
- 10 arm joints: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_yaw

The head meshes exist in the MJCF, but `asimov_1.xml` does not define any
actuated neck/head joints.

Motor families:
- EC-A6416-P2-25: hip_pitch, waist_yaw
- EC-A5013-H17-100: hip_roll, shoulder_pitch
- EC-A3814-H14-107: hip_yaw, shoulder_yaw
- EC-A4315-P2-36: knee, shoulder_roll
- EC-A4310-P2-36: ankle, elbow, wrist

This file uses `mjlab.actuator.DcMotorActuatorCfg`. In the current `mjlab` API,
command delay is configured directly on each actuator cfg, so there is no
separate delayed-actuator wrapper.
"""

from dataclasses import dataclass
from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF path.
##

ASIMOV_1_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "asimov_1" / "xmls" / "asimov_1.xml"
)
assert ASIMOV_1_XML.exists(), f"Asimov 1 XML not found at {ASIMOV_1_XML}"


def get_spec() -> mujoco.MjSpec:
  """Load the MuJoCo spec from XML."""
  return mujoco.MjSpec.from_file(str(ASIMOV_1_XML))


##
# Actuator config.
##


@dataclass(frozen=True)
class _MotorSpec:
  armature: float
  effort_limit: float
  saturation_effort: float
  velocity_limit: float
  frictionloss: float
  viscous_damping: float


@dataclass(frozen=True)
class _PdGains:
  stiffness: float
  damping: float


# Motor output-side reflected inertia (armature), Coulomb friction
# (frictionloss), and viscous damping, from hardware characterization. These
# override the placeholder joint values in `asimov_1.xml`.
A6416 = _MotorSpec(
  armature=0.0698,
  effort_limit=40.0,
  saturation_effort=120.0,
  velocity_limit=12.57,
  frictionloss=0.70,
  viscous_damping=0.050,
)
A5013 = _MotorSpec(
  armature=0.1400,
  effort_limit=30.0,
  saturation_effort=90.0,
  velocity_limit=3.98,
  frictionloss=0.20,
  viscous_damping=0.020,
)
A3814 = _MotorSpec(
  armature=0.0687,
  effort_limit=20.0,
  saturation_effort=60.0,
  velocity_limit=5.45,
  frictionloss=0.70,
  viscous_damping=0.050,
)
A4315 = _MotorSpec(
  armature=0.0330,
  effort_limit=25.0,
  saturation_effort=75.0,
  velocity_limit=12.25,
  frictionloss=0.70,
  viscous_damping=0.020,
)
A4310_ANKLE_PITCH = _MotorSpec(
  armature=0.0484,
  effort_limit=40.0,
  saturation_effort=145.4,
  velocity_limit=9.32,
  frictionloss=0.40,
  viscous_damping=0.015,
)
A4310_ANKLE_ROLL = _MotorSpec(
  armature=0.0484,
  effort_limit=17.0,
  saturation_effort=57.6,
  velocity_limit=9.32,
  frictionloss=0.40,
  viscous_damping=0.015,
)
A4310_SINGLE = _MotorSpec(
  armature=0.0242,
  effort_limit=12.0,
  saturation_effort=36.0,
  velocity_limit=9.32,
  frictionloss=0.40,
  viscous_damping=0.015,
)


# PD gains (stiffness, damping) per joint group.
HIP_PITCH_GAINS = _PdGains(150.0, 5.0)
HIP_ROLL_GAINS = _PdGains(150.0, 5.0)
HIP_YAW_GAINS = _PdGains(150.0, 5.0)
KNEE_GAINS = _PdGains(150.0, 5.0)
ANKLE_PITCH_GAINS = _PdGains(440.0, 20.0)
ANKLE_ROLL_GAINS = _PdGains(440.0, 20.0)
WAIST_YAW_GAINS = _PdGains(65.0, 5.0)
SHOULDER_PITCH_GAINS = _PdGains(57.0, 5.0)
SHOULDER_ROLL_GAINS = _PdGains(86.0, 5.0)
SHOULDER_YAW_GAINS = _PdGains(96.0, 5.0)
ELBOW_GAINS = _PdGains(40.0, 2.0)
WRIST_YAW_GAINS = _PdGains(40.0, 2.0)

# Command delay in physics substeps. With the 5 ms MJCF timestep, this models
# 0-5 ms of actuator command lag and resamples once per policy step.
DELAY_MIN_LAG = 0
DELAY_MAX_LAG = 1
DELAY_HOLD_PROB = 0.1
DELAY_UPDATE_PERIOD = 4


def _dc(
  target_names_expr: tuple[str, ...],
  gains: _PdGains,
  motor: _MotorSpec,
) -> DcMotorActuatorCfg:
  return DcMotorActuatorCfg(
    target_names_expr=target_names_expr,
    stiffness=gains.stiffness,
    damping=gains.damping,
    effort_limit=motor.effort_limit,
    saturation_effort=motor.saturation_effort,
    velocity_limit=motor.velocity_limit,
    armature=motor.armature,
    frictionloss=motor.frictionloss,
    viscous_damping=motor.viscous_damping,
    delay_min_lag=DELAY_MIN_LAG,
    delay_max_lag=DELAY_MAX_LAG,
    delay_hold_prob=DELAY_HOLD_PROB,
    delay_update_period=DELAY_UPDATE_PERIOD,
  )


ASIMOV_1_ACTUATOR_HIP_PITCH = _dc((".*_hip_pitch_joint",), HIP_PITCH_GAINS, A6416)
ASIMOV_1_ACTUATOR_HIP_ROLL = _dc((".*_hip_roll_joint",), HIP_ROLL_GAINS, A5013)
ASIMOV_1_ACTUATOR_HIP_YAW = _dc((".*_hip_yaw_joint",), HIP_YAW_GAINS, A3814)
ASIMOV_1_ACTUATOR_KNEE = _dc((".*_knee_joint",), KNEE_GAINS, A4315)
ASIMOV_1_ACTUATOR_ANKLE_PITCH = _dc(
  (".*_ankle_pitch_joint",), ANKLE_PITCH_GAINS, A4310_ANKLE_PITCH
)
ASIMOV_1_ACTUATOR_ANKLE_ROLL = _dc(
  (".*_ankle_roll_joint",), ANKLE_ROLL_GAINS, A4310_ANKLE_ROLL
)
ASIMOV_1_ACTUATOR_WAIST = _dc(("waist_yaw_joint",), WAIST_YAW_GAINS, A6416)
ASIMOV_1_ACTUATOR_SHOULDER_PITCH = _dc(
  (".*_shoulder_pitch_joint",), SHOULDER_PITCH_GAINS, A5013
)
ASIMOV_1_ACTUATOR_SHOULDER_ROLL = _dc(
  (".*_shoulder_roll_joint",), SHOULDER_ROLL_GAINS, A4315
)
ASIMOV_1_ACTUATOR_SHOULDER_YAW = _dc(
  (".*_shoulder_yaw_joint",), SHOULDER_YAW_GAINS, A3814
)
ASIMOV_1_ACTUATOR_ELBOW = _dc((".*_elbow_joint",), ELBOW_GAINS, A4310_SINGLE)
ASIMOV_1_ACTUATOR_WRIST = _dc((".*_wrist_yaw_joint",), WRIST_YAW_GAINS, A4310_SINGLE)

ASIMOV_1_ACTUATORS = (
  ASIMOV_1_ACTUATOR_HIP_PITCH,
  ASIMOV_1_ACTUATOR_HIP_ROLL,
  ASIMOV_1_ACTUATOR_HIP_YAW,
  ASIMOV_1_ACTUATOR_KNEE,
  ASIMOV_1_ACTUATOR_ANKLE_PITCH,
  ASIMOV_1_ACTUATOR_ANKLE_ROLL,
  ASIMOV_1_ACTUATOR_WAIST,
  ASIMOV_1_ACTUATOR_SHOULDER_PITCH,
  ASIMOV_1_ACTUATOR_SHOULDER_ROLL,
  ASIMOV_1_ACTUATOR_SHOULDER_YAW,
  ASIMOV_1_ACTUATOR_ELBOW,
  ASIMOV_1_ACTUATOR_WRIST,
)

##
# Keyframe config.
##

# Default initialization pose: a stable standing position.
KNEES_STAND_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.639),
  joint_pos={
    "left_hip_pitch_joint": 0.1,
    "right_hip_pitch_joint": -0.1,
    ".*_hip_roll_joint": 0.0,
    ".*_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.0,
    "right_knee_joint": 0.0,
    "left_ankle_pitch_joint": 0.0,
    "right_ankle_pitch_joint": 0.0,
    ".*_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "left_shoulder_pitch_joint": -0.35,
    "right_shoulder_pitch_joint": 0.35,
    "left_shoulder_roll_joint": -0.18,
    "right_shoulder_roll_joint": 0.18,
    ".*_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.87,
    "right_elbow_joint": -0.87,
    ".*_wrist_yaw_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={
    r"^(left|right)_foot[1-4]_collision$": 3,
    ".*_collision": 1,
  },
  priority={r"^(left|right)_foot[1-4]_collision$": 1},
  friction={r"^(left|right)_foot[1-4]_collision$": (0.9,)},
  disable_other_geoms=False,
)

##
# Final config.
##

ASIMOV_1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=ASIMOV_1_ACTUATORS,
  soft_joint_pos_limit_factor=0.9,
)


def get_asimov_1_robot_cfg() -> EntityCfg:
  """Return a fresh Asimov 1 robot config."""
  return EntityCfg(
    init_state=KNEES_STAND_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=ASIMOV_1_ARTICULATION,
  )


ASIMOV_1_ACTION_SCALE: dict[str, float] = {}
for actuator in ASIMOV_1_ACTUATORS:
  assert actuator.effort_limit is not None
  for target_name in actuator.target_names_expr:
    ASIMOV_1_ACTION_SCALE[target_name] = (
      0.30 * actuator.effort_limit / actuator.stiffness
    )

if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_asimov_1_robot_cfg())
  viewer.launch(robot.spec.compile())
