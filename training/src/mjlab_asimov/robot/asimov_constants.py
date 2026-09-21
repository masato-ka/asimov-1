"""Asimov-1 robot definition for mjlab.

The MJCF under ``sim-model/xmls/asimov_1.xml`` is deliberately actuator-free and
keyframe-free; everything needed for training is applied here in Python without
modifying the MJCF:

* the 12 leg joints get PD position actuators (the RL action space),
* the 11 arm / waist joints are *not* actuated and are held by MuJoCo's native joint
  spring-damper (the ``passive_upper`` pattern that the MJCF already defines but no
  joint uses),
* the home pose is defined as ``EntityCfg.InitialStateCfg``.
"""

import math
import re
from pathlib import Path

import mujoco
import numpy as np

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF.
##

# training/src/mjlab_asimov/robot/asimov_constants.py -> repo root is parents[4].
REPO_ROOT: Path = Path(__file__).resolve().parents[4]
ASIMOV_XML: Path = REPO_ROOT / "sim-model" / "xmls" / "asimov_1.xml"
assert ASIMOV_XML.exists(), f"MJCF not found: {ASIMOV_XML}"

LEG_JOINT_EXPR: tuple[str, ...] = (
  r".*_hip_pitch_joint",
  r".*_hip_roll_joint",
  r".*_hip_yaw_joint",
  r".*_knee_joint",
  r".*_ankle_pitch_joint",
  r".*_ankle_roll_joint",
)

# Arm and waist joints. Not part of the action space.
PASSIVE_JOINT_EXPR: tuple[str, ...] = (
  r".*_shoulder_pitch_joint",
  r".*_shoulder_roll_joint",
  r".*_shoulder_yaw_joint",
  r".*_elbow_joint",
  r".*_wrist_yaw_joint",
  r"waist_yaw_joint",
)

# Mirrors the ``passive_upper`` default class in the MJCF.
PASSIVE_STIFFNESS = 80.0
PASSIVE_DAMPING = 15.0

# In the MJCF the elbows carry ``ref=∓0.785398`` (all other joints have ref=0), i.e. the
# CAD-authored arm pose corresponds to qpos=ref. Rest the spring (and spawn) there
# instead of at qpos=0, which sits on the joint-range boundary.
ELBOW_REST: dict[str, float] = {
  "left_elbow_joint": 0.785398,
  "right_elbow_joint": -0.785398,
}


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(ASIMOV_XML))
  # The MJCF ships its own ground plane and lights for the standalone viewer. In mjlab
  # the scene's terrain provides the ground; a second coincident plane would double
  # every foot contact.
  for geom in [g for g in spec.geoms if g.name == "floor"]:
    spec.delete(geom)
  for light in list(spec.lights):
    spec.delete(light)
  passive = re.compile("|".join(f"(?:{e})" for e in PASSIVE_JOINT_EXPR))
  for joint in spec.joints:
    if passive.fullmatch(joint.name):
      # MjsJoint stiffness/damping are 3-vectors (polynomial coefficients); only the
      # linear term is used.
      joint.stiffness = np.array([PASSIVE_STIFFNESS, 0.0, 0.0])
      joint.damping = np.array([PASSIVE_DAMPING, 0.0, 0.0])
      joint.springref = ELBOW_REST.get(joint.name, 0.0)
  return spec


##
# Actuators (legs only).
##

# Starting point taken from the Menlo Asimov locomotion guide (system-identification
# chapter): kp=65, kd=5. The armature values quoted there (hip pitch 0.095625, knee
# 0.0339552, ankle 0.0565056) match this MJCF, so the numbers apply to this model.
# Effort limits are the per-joint URDF limits (sim-model/urdf/asimov_1.urdf).
LEG_STIFFNESS = 65.0
LEG_DAMPING = 5.0

LEG_EFFORT_LIMITS: dict[str, float] = {
  r".*_hip_pitch_joint": 45.0,
  r".*_hip_roll_joint": 45.0,
  r".*_hip_yaw_joint": 28.0,
  r".*_knee_joint": 45.0,
  r".*_ankle_pitch_joint": 40.0,
  r".*_ankle_roll_joint": 17.0,
}

# Actuator command latency in physics steps (Menlo: delay_min_lag=0, delay_max_lag=1).
ACTION_DELAY_MIN_LAG = 0
ACTION_DELAY_MAX_LAG = 1

ASIMOV_ARTICULATION = EntityArticulationInfoCfg(
  actuators=tuple(
    BuiltinPositionActuatorCfg(
      target_names_expr=(expr,),
      stiffness=LEG_STIFFNESS,
      damping=LEG_DAMPING,
      effort_limit=effort,
      delay_min_lag=ACTION_DELAY_MIN_LAG,
      delay_max_lag=ACTION_DELAY_MAX_LAG,
    )
    for expr, effort in LEG_EFFORT_LIMITS.items()
  ),
  soft_joint_pos_limit_factor=0.9,
)

# Menlo: action scale = 0.30 * effort_limit / stiffness (per joint).
ASIMOV_ACTION_SCALE: dict[str, float] = {
  expr: 0.30 * effort / LEG_STIFFNESS for expr, effort in LEG_EFFORT_LIMITS.items()
}

##
# Home keyframe.
##

# Left/right joint axes are mirrored in the MJCF (left hip_pitch/knee/ankle_pitch axis
# +y, right axis -y), so the *sign* of the joint value flips between sides for the same
# physical motion. Physical knee flexion is +a on the left and -a on the right.
#
# Pose (physical rotations about +y): thigh -HIP (tilted forward), knee +KNEE (shank
# swings back), pelvis pitched forward by BASE_PITCH, ankle chosen so the sole stays
# parallel to the floor: BASE_PITCH - HIP + KNEE + ankle = 0. The upper body mass sits
# behind the hips, so without the forward lean the whole-body CoM projects ~3 cm behind
# the centre of the feet. This combination was found by a grid search (CoM within ~5 mm
# of the foot centre, ankle angle kept small because the ankle pitch range is only
# +-0.35 rad). Values are for the left side; the right side is negated.
_HIP = 0.25
_KNEE = 0.30
_BASE_PITCH = 0.12
_ANKLE = -(_BASE_PITCH - _HIP + _KNEE)

# Pelvis height at which the foot spheres just touch the floor for the pose above
# (0.6171 m measured with scripts/check_standing.py, plus a few mm of clearance).
# Straight-leg reference is 0.630.
HOME_BASE_HEIGHT = 0.62
HOME_BASE_ROT = (math.cos(_BASE_PITCH / 2), 0.0, math.sin(_BASE_PITCH / 2), 0.0)  # wxyz

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, HOME_BASE_HEIGHT),
  rot=HOME_BASE_ROT,
  joint_pos={
    "left_hip_pitch_joint": -_HIP,
    "right_hip_pitch_joint": _HIP,
    "left_knee_joint": _KNEE,
    "right_knee_joint": -_KNEE,
    "left_ankle_pitch_joint": _ANKLE,
    "right_ankle_pitch_joint": -_ANKLE,
    **ELBOW_REST,
  },
  joint_vel={".*": 0.0},
)


##
# Collision.
##

FOOT_SITE_NAMES: tuple[str, ...] = ("left_foot", "right_foot")
FOOT_GEOM_NAMES: tuple[str, ...] = tuple(
  f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 5)
)
_FOOT_GEOM_REGEX = r"^(left|right)_foot[1-4]_collision$"

# Keep the MJCF's collision geoms (and its parent/child <contact><exclude> pairs) but
# make the contact structure explicit: feet get condim=3, friction=0.6 and priority 1
# (so their friction, which the friction DR overrides, wins over the terrain's); every
# other collision capsule gets condim=1 (the MJCF default of 6 is needlessly expensive
# for links that are not expected to touch the ground) and priority 0.
ASIMOV_COLLISION = CollisionCfg(
  geom_names_expr=(r".*_collision",),
  contype=1,
  conaffinity=1,
  condim={_FOOT_GEOM_REGEX: 3, r".*_collision": 1},
  priority={_FOOT_GEOM_REGEX: 1, r".*": 0},
  friction={_FOOT_GEOM_REGEX: (0.6,)},
)


def get_asimov_robot_cfg() -> EntityCfg:
  """Fresh Asimov-1 EntityCfg (new instance each call to avoid shared mutation)."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(ASIMOV_COLLISION,),
    spec_fn=get_spec,
    articulation=ASIMOV_ARTICULATION,
  )
