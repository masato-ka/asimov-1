"""Static (CPU-only) checks of the Asimov-1 robot definition."""

import mujoco
import numpy as np
import pytest

from mjlab.entity import Entity
from mjlab_asimov.robot import asimov_constants as C


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
  return Entity(C.get_asimov_robot_cfg()).spec.compile()


def _jid(model: mujoco.MjModel, name: str) -> int:
  return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def test_twelve_leg_actuators_and_no_others(model):
  names = {
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)
  }
  expected = {
    f"{side}_{j}_joint"
    for side in ("left", "right")
    for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
  }
  assert names == expected


def test_ground_plane_removed(model):
  # mjlab's terrain provides the ground; the MJCF's own plane would double contacts.
  assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor") == -1
  assert model.nlight == 0


def test_passive_joints_are_sprung_not_actuated(model):
  actuated = {int(t) for t in model.actuator_trnid[:, 0]}
  passive = [
    j for j in range(1, model.njnt) if j not in actuated
  ]  # skip the freejoint
  assert len(passive) == 11
  for j in passive:
    assert model.jnt_stiffness[j] == pytest.approx(C.PASSIVE_STIFFNESS)
    assert model.dof_damping[model.jnt_dofadr[j]] == pytest.approx(C.PASSIVE_DAMPING)


def test_elbow_spring_rests_at_ref(model):
  for name, ref in C.ELBOW_REST.items():
    adr = model.jnt_qposadr[_jid(model, name)]
    assert model.qpos_spring[adr] == pytest.approx(ref)
    assert model.qpos0[adr] == pytest.approx(ref)


def test_home_pose_is_inside_joint_ranges_with_mirrored_signs(model):
  cfg = C.get_asimov_robot_cfg()
  for name, q in cfg.init_state.joint_pos.items():
    lo, hi = model.jnt_range[_jid(model, name)]
    assert lo <= q <= hi, f"{name}={q} outside [{lo}, {hi}]"
  home = cfg.init_state.joint_pos
  # Mirrored axes: same physical motion has opposite joint sign on the two sides.
  for j in ("hip_pitch", "knee", "ankle_pitch"):
    assert home[f"left_{j}_joint"] == pytest.approx(-home[f"right_{j}_joint"])


def test_action_scale_covers_every_leg_joint():
  import re

  for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"):
    hits = [
      s
      for expr, s in C.ASIMOV_ACTION_SCALE.items()
      if re.fullmatch(expr, f"left_{j}_joint")
    ]
    assert len(hits) == 1 and hits[0] > 0


def test_home_height_puts_feet_just_above_floor(model):
  data = mujoco.MjData(model)
  key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "init_state")
  mujoco.mj_resetDataKeyframe(model, data, key)
  mujoco.mj_forward(model, data)
  feet = [
    i
    for i in range(model.ngeom)
    if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) in C.FOOT_GEOM_NAMES
  ]
  assert len(feet) == 8
  lowest = min(data.geom_xpos[i, 2] - model.geom_size[i, 0] for i in feet)
  assert 0.0 <= lowest < 0.01
  assert np.isfinite(data.qpos).all()
