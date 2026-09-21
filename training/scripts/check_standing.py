"""Spawn-and-settle check for the Asimov-1 home pose (CPU MuJoCo, no GPU needed).

Builds the mjlab Entity exactly as training does, drops it on a plane with the PD
targets held at the home pose (zero policy action) and reports whether it stays
upright. Also reports the pelvis height at which the feet just touch the floor so
HOME_BASE_HEIGHT can be corrected.

Usage: uv run python training/scripts/check_standing.py [--seconds 3]
"""

import argparse

import mujoco
import numpy as np

from mjlab.entity import Entity
from mjlab_asimov.robot.asimov_constants import get_asimov_robot_cfg


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--seconds", type=float, default=3.0)
  args = parser.parse_args()

  entity = Entity(get_asimov_robot_cfg())
  spec = entity.spec
  spec.worldbody.add_geom(
    name="floor",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[0, 0, 0.05],
    friction=[1.2, 0.5, 0.1],
  )
  model = spec.compile()
  data = mujoco.MjData(model)

  key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "init_state")
  mujoco.mj_resetDataKeyframe(model, data, key_id)
  mujoco.mj_forward(model, data)

  foot_geoms = [
    i
    for i in range(model.ngeom)
    if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").endswith(
      tuple(f"foot{k}_collision" for k in range(1, 5))
    )
  ]
  assert len(foot_geoms) == 8, f"expected 8 foot geoms, found {len(foot_geoms)}"
  lowest = min(data.geom_xpos[i, 2] - model.geom_size[i, 0] for i in foot_geoms)
  pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link")
  print(f"nq={model.nq} nv={model.nv} nu={model.nu}")
  print(f"initial pelvis z={data.xpos[pelvis_id, 2]:.4f}  lowest foot point z={lowest:+.4f}")
  print(f"=> HOME_BASE_HEIGHT that just touches the floor ~ {data.xpos[pelvis_id, 2] - lowest:.4f}")

  # Support polygon (x extent of the foot spheres) vs. whole-body CoM in the pelvis-yaw
  # frame. A home pose is only balanceable if the CoM projects inside the feet.
  com = data.subtree_com[pelvis_id]
  foot_xy = np.array([data.geom_xpos[i, :2] for i in foot_geoms])
  print(
    f"CoM x={com[0]:+.4f} y={com[1]:+.4f} | feet x-range [{foot_xy[:, 0].min():+.3f}, "
    f"{foot_xy[:, 0].max():+.3f}] mean {foot_xy[:, 0].mean():+.3f}"
  )

  q_home = data.qpos.copy()
  timestep = model.opt.timestep
  report_every = int(0.5 / timestep)
  n_steps = int(args.seconds / timestep)
  for step in range(1, n_steps + 1):
    mujoco.mj_step(model, data)
    if not np.isfinite(data.qpos).all():
      raise RuntimeError(f"simulation diverged at step {step}")
    if step % report_every == 0:
      up_z = data.xmat[pelvis_id].reshape(3, 3)[2, 2]
      tilt = np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0)))
      print(
        f"  t={step * timestep:4.1f}s pelvis z={data.xpos[pelvis_id, 2]:.3f} "
        f"xy=({data.xpos[pelvis_id, 0]:+.3f},{data.xpos[pelvis_id, 1]:+.3f}) tilt={tilt:5.1f} deg"
      )

  dq = data.qpos[7:] - q_home[7:]
  names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, model.njnt)
  ]
  print("largest joint drift from home at the end (rad):")
  for j in np.argsort(-np.abs(dq))[:5]:
    print(f"  {names[j]:32s} {dq[j]:+.4f}")
  # A static PD hold is not expected to balance a 35 kg biped (ankle kp is below the
  # inverted-pendulum stiffness m*g*h); balancing is the policy's job. This script only
  # validates the model build, the foot/floor contact height and the CoM placement.


if __name__ == "__main__":
  main()
