"""Asimov-1 stair-climbing (ascent-only) task.

Built the same way as ``asimov_velocity_env_cfg.py`` (start from mjlab's
``make_velocity_env_cfg()`` template, then customise), but where the flat task
*strips* the template's generator-terrain wiring (it has nothing to scan on a plane),
this task *keeps and restores* it, pointed at ``ASIMOV_STAIRS_TERRAINS_CFG`` (see
``asimov_stairs_terrain.py`` for why that config uses inverted, pit-shaped pyramid
stairs rather than mjlab's shipped ``STAIRS_TERRAINS_CFG``: a forward velocity
command must drive the robot *up and out* of a pit, not down off a platform).

Design points:

* Action space is unchanged from the flat task: the 12 leg joints only.
* Actor observation grows from 45-d to 232-d (45 + a 17x11 `height_scan` heightmap
  ahead of the pelvis) so the policy can see the stairs coming. This is a first-order
  change to the policy's input; PPO hyperparameters carried over from the flat task
  are an untested starting point, not a validated choice, for this input size.
* `foot_clearance` / `foot_swing_height` target_height is raised to clear the tallest
  riser (0.10 m) with margin. The `pose` reward's walking std is loosened further than
  the flat task's (already loosened) values, since climbing structurally requires
  larger hip/knee excursion every step, not just occasionally.
* Velocity commands are slower and forward-biased (climbing is deliberate, not brisk),
  and `terrain_levels` curriculum is enabled (the terrain's own `curriculum=True`
  interpolates step height from 0.02 m up to 0.10 m across 10 rows per tier).
"""

import dataclasses

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_asimov.robot.asimov_constants import (
  ASIMOV_ACTION_SCALE,
  FOOT_GEOM_NAMES,
  FOOT_SITE_NAMES,
  LEG_JOINT_EXPR,
  get_asimov_robot_cfg,
)
from mjlab_asimov.tasks.asimov_stairs_terrain import ASIMOV_STAIRS_TERRAINS_CFG

# The pelvis carries the IMU; the upper body is rigidly held on top of it.
BASE_BODY = "pelvis_link"


def _leg_joints() -> SceneEntityCfg:
  return SceneEntityCfg("robot", joint_names=LEG_JOINT_EXPR)


def asimov_stairs_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Asimov-1 ascent-only stair-climbing configuration."""
  cfg = make_velocity_env_cfg()

  ##
  # Scene: robot on the inverted-pyramid stairs terrain.
  ##

  cfg.scene.entities = {"robot": get_asimov_robot_cfg()}

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "generator"
  cfg.scene.terrain.terrain_generator = dataclasses.replace(ASIMOV_STAIRS_TERRAINS_CFG)
  # Start everyone on the flat/easiest row; terrain_levels_vel promotes from there.
  cfg.scene.terrain.max_init_terrain_level = 0

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = BASE_BODY
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in FOOT_SITE_NAMES
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern=BASE_BODY, entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern=BASE_BODY, entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground_cfg, self_collision_cfg)

  # Generator terrain has far more contact-capable geoms than a flat plane; use
  # the same generous buffers as mjlab's shipped rough-terrain (G1) config.
  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 70
  # cfg.sim.njmax stays at the template's default (1500).

  cfg.viewer.body_name = BASE_BODY

  ##
  # Actions: 12 leg joints, same scale as the flat task.
  ##

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = ASIMOV_ACTION_SCALE

  ##
  # Observations.
  ##

  actor = cfg.observations["actor"]
  critic = cfg.observations["critic"]

  # Unlike the flat task, height_scan (the terrain-ahead heightmap) is KEPT: it is
  # how the policy sees the stairs coming. Actor obs grows from 45-d to 232-d
  # (45 + 17x11 grid).
  actor.terms.pop("base_lin_vel")  # privileged: real robot can't measure this.

  actor.terms["base_ang_vel"].noise = Unoise(n_min=-0.01, n_max=0.01)
  actor.terms["projected_gravity"].noise = Unoise(n_min=-0.05, n_max=0.05)
  actor.terms["joint_pos"].noise = Unoise(n_min=-0.01, n_max=0.01)
  actor.terms["joint_vel"].noise = Unoise(n_min=-0.1, n_max=0.1)

  for group in (actor, critic):
    group.terms["joint_pos"].params["asset_cfg"] = _leg_joints()
    group.terms["joint_vel"].params["asset_cfg"] = _leg_joints()

  for name in ("joint_pos", "joint_vel"):
    actor.terms[name].delay_min_lag = 0
    actor.terms[name].delay_max_lag = 1

  ##
  # Commands: slow, forward-biased (climbing is deliberate, not brisk).
  ##

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_x = (-0.1, 0.4)
  twist_cmd.ranges.lin_vel_y = (-0.1, 0.1)
  twist_cmd.ranges.ang_vel_z = (-0.3, 0.3)
  # SE4 (tried and reverted): lowering rel_forward_envs (0.6 -> 0.45) was meant to
  # reduce how many envs get forced through mjlab's hardcoded
  # `vel_command_b[fwd_ids,0].abs().clamp(min=0.3)` floor (~80% of forward-flagged envs
  # were pinned at exactly vx=0.30, population mean commanded vx only 0.234). It did not
  # improve gait cadence (touchdown Hz regressed from ~1.02 back to ~0.85, roughly the
  # pre-SE1 baseline) and substantially hurt curriculum progress (terrain_levels mean
  # 3.38 -> 1.85 after 3000 more iterations): with more envs sampling the full range
  # (which includes low/negative vx and isn't forward-clamped), average per-episode
  # forward progress dropped, so fewer envs covered the ~4 m needed for
  # terrain_levels_vel to promote them. Kept at 0.6.
  twist_cmd.rel_forward_envs = 0.6
  twist_cmd.rel_standing_envs = 0.05
  twist_cmd.rel_heading_envs = 0.2
  twist_cmd.viz.z_offset = 1.05

  ##
  # Curriculum: keep terrain_levels (the terrain's own curriculum=True drives step
  # height up as the robot demonstrates it can walk the current difficulty). Drop
  # command_vel: there is no reason a stair-climbing policy should ever need the
  # escalated 2-3 m/s top speeds that curriculum ramps toward.
  ##

  cfg.curriculum.pop("command_vel", None)

  ##
  # Events / domain randomization: same as the flat task.
  ##

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_GEOM_NAMES
  cfg.events["foot_friction"].params["ranges"] = (1.0, 1.5)
  cfg.events["encoder_bias"].params["bias_range"] = (-0.02, 0.02)
  cfg.events.pop("base_com", None)
  cfg.events["pd_gains"] = EventTermCfg(
    mode="reset",
    func=dr.pd_gains,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "kp_range": (0.9, 1.1),
      "kd_range": (0.9, 1.1),
      "operation": "scale",
    },
  )
  cfg.events["push_robot"].interval_range_s = (4.0, 8.0)
  cfg.events["push_robot"].params["velocity_range"] = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
  }

  ##
  # Rewards.
  ##

  # SE3 (training/TECHNICAL_REPORT.md-style tuning log): the first stairs policy
  # (trained with hip_pitch/knee/ankle_pitch stds of 0.9/1.0/0.6) converged to one
  # uniform, slow ~0.87 Hz gait on EVERY terrain type, including flat cells -- roughly
  # 2x the joint excursion of the dedicated flat task even where the terrain didn't
  # require it. SE1 (tightening the air_time reward's threshold_max from 0.5 to 0.35,
  # since the old gait's ~0.5-0.6s swing duration sat right past that dead cutoff) only
  # nudged cadence to ~0.98 Hz after 3000 more iterations -- not enough. Tightening the
  # sagittal stds partway back toward the flat task's values (0.6/0.7/0.5) -- but still
  # looser, since the hardest stair tier still needs more excursion than flat ground --
  # is the next lever. Roll/yaw/ankle_roll stay tight -- lateral stability on a tread
  # edge is if anything more safety-critical than on flat ground.
  stairs_walking_std = {
    r".*hip_pitch.*": 0.75,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.85,
    r".*ankle_pitch.*": 0.55,
    r".*ankle_roll.*": 0.1,
  }
  pose = cfg.rewards["pose"].params
  pose["asset_cfg"] = _leg_joints()
  pose["std_standing"] = {".*": 0.05}
  pose["std_walking"] = stairs_walking_std
  pose["std_running"] = stairs_walking_std

  cfg.rewards["upright"].params["asset_cfg"].body_names = (BASE_BODY,)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (BASE_BODY,)
  cfg.rewards["dof_pos_limits"].params["asset_cfg"] = _leg_joints()

  for reward_name in ("foot_clearance", "foot_slip"):
    cfg.rewards[reward_name].params["asset_cfg"].site_names = FOOT_SITE_NAMES

  cfg.rewards["body_ang_vel"].weight = -0.08
  cfg.rewards["angular_momentum"].weight = -0.03

  # Raise the swing-height target above the tallest riser (0.10 m) with margin: a
  # foot swinging toward a higher tread needs to clear its leading edge, not just
  # reach the flat-ground target used by the walking task.
  cfg.rewards["foot_clearance"].params["target_height"] = 0.12
  cfg.rewards["foot_swing_height"].params["target_height"] = 0.12
  cfg.rewards["foot_swing_height"].weight = -2.0
  # threshold_max=0.5 (the mjlab default) turned out to be a dead zone: the first
  # trained policy converged to a uniform, slow ~0.87 Hz gait (swing duration ~0.5-0.6s)
  # on EVERY terrain type, including flat cells where nothing about the terrain forces
  # it. That swing duration sits right at/over the reward's upper cutoff, so `air_time`
  # measured exactly 0 even restricted to flat-terrain envs -- no gradient was pushing
  # cadence up. Tightening threshold_max to ~0.35s (close to the dedicated flat task's
  # natural ~0.3s swing duration at a similar speed) puts the current gait outside the
  # rewarded band on the correctable side, restoring a live signal toward shorter
  # swings / higher cadence. Weight raised alongside it to strengthen the pull once back
  # inside the window. See training/scripts/diagnose_gait.py for the per-terrain-type
  # diagnostic that found this.
  cfg.rewards["air_time"].params["threshold_max"] = 0.35
  cfg.rewards["air_time"].weight = 0.8
  cfg.rewards["action_rate_l2"].weight = -0.03

  cfg.rewards["torques"] = RewardTermCfg(func=mdp.joint_torques_l2, weight=-5e-5)
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  ##
  # Terminations: keep out_of_terrain_bounds (meaningful on generator terrain, unlike
  # on the flat task where it's a no-op).
  ##

  ##
  # Play mode: smaller, non-curriculum terrain for interactive viewing; no noise, no
  # pushes, effectively endless episodes.
  ##

  if play:
    cfg.episode_length_s = int(1e9)
    actor.enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )
    terrain_generator = cfg.scene.terrain.terrain_generator
    assert terrain_generator is not None
    terrain_generator.curriculum = False
    terrain_generator.num_rows = 5
    terrain_generator.num_cols = 4
    terrain_generator.border_width = 10.0

  return cfg
