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
* The raw 232-d `height_scan` heightmap is kept for the CRITIC only (privileged: it has
  no real-hardware analog anyway, see TECHNICAL_REPORT.md). The actor instead gets a
  1-d `climbing_mode` flag (ground truth from `env.scene.terrain.terrain_types`, which
  is constant per env per episode -- no online classification needed), fed alongside
  the twist command. Diagnosis (TECHNICAL_REPORT.md §5.5) found the first policy
  learned one uniform, slow gait on every terrain type despite height_scan technically
  containing terrain-discriminating information -- nothing told it to actually switch
  behaviour. `climbing_mode` also selects mode-specific parameters for the `pose`,
  `foot_clearance`/`foot_swing_height`, and `air_time` rewards (see
  `mjlab_asimov.tasks.mdp`), instead of relying on the policy to discover the switch
  is worthwhile under one fixed reward.
* Actor observation is therefore 45 (flat-task-sized) + 1 (climbing_mode) = 46-d.
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
from mjlab.managers.observation_manager import ObservationTermCfg
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
from mjlab_asimov.tasks import mdp as asimov_mdp
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

  actor.terms.pop("base_lin_vel")  # privileged: real robot can't measure this.
  # Unlike height_scan (kept critic-only, see module docstring), climbing_mode is a
  # single ground-truth bit the actor is allowed to use: it is a stand-in for what a
  # real depth sensor / terrain classifier would eventually provide, not something the
  # real robot could not in principle ever know.
  actor.terms.pop("height_scan")
  actor.terms["climbing_mode"] = ObservationTermCfg(func=asimov_mdp.climbing_mode)
  critic.terms["climbing_mode"] = ObservationTermCfg(func=asimov_mdp.climbing_mode)

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

  # track_linear_velocity / track_angular_velocity: the template's std (0.5 / sqrt(0.5))
  # is tuned for the flat task's much wider command ranges (lin_vel_x +-0.8, width 1.6;
  # ang_vel_z +-0.6, width 1.2). This task's ranges are 3.2x / 2x narrower
  # (lin_vel_x -0.1..0.4, width 0.5; ang_vel_z +-0.3, width 0.6), so the same std made
  # the tracking reward nearly insensitive to whether the robot actually moved at the
  # low end of the range: standing still at a commanded vx=0.2-0.3 was already a small
  # absolute error relative to std=0.5, so `exp(-error^2/std^2)` stayed close to its
  # maximum with almost no incentive to move. Measured effect: achieved velocity was
  # ~0 for commands up to ~0.15, only starting to track meaningfully above ~0.2 (see
  # training/TECHNICAL_REPORT.md and the keyboard-play feedback that prompted this).
  # Scaling std down by the same ratio as the range narrowing restores a comparably
  # sharp tracking incentive across the whole (narrower) command range.
  cfg.rewards["track_linear_velocity"].params["std"] = 0.15
  cfg.rewards["track_angular_velocity"].params["std"] = 0.35

  # `climbing_mode`-aware reward design (see mjlab_asimov.tasks.mdp module docstring for
  # the full rationale): the SE1-SE4 tuning history (training/TECHNICAL_REPORT.md §5.6)
  # spent many experiments trying to find ONE set of pose/clearance/air-time parameters
  # that works acceptably on both flat ground and stairs, under a single fixed reward
  # applied everywhere -- with limited success (the policy applied one uniform,
  # compromise gait regardless of terrain). Instead of continuing to tune a single
  # shared config, `pose`/`foot_clearance`/`foot_swing_height`/`air_time` now pick
  # between the flat task's proven walking values and the SE1/SE3-tuned climbing values
  # based on `climbing_mode`, so the reward itself tells the policy which behaviour is
  # wanted rather than leaving it to infer that from a raw heightmap.
  #
  # "Walking" values are exactly the dedicated flat task's tuned settings
  # (asimov_velocity_env_cfg.py); "climbing" values are this task's own SE1/SE3 tuning.
  flat_task_walking_std = {
    r".*hip_pitch.*": 0.6,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.7,
    r".*ankle_pitch.*": 0.5,
    r".*ankle_roll.*": 0.1,
  }
  stairs_climbing_std = {
    r".*hip_pitch.*": 0.75,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.85,
    r".*ankle_pitch.*": 0.55,
    r".*ankle_roll.*": 0.1,
  }
  cfg.rewards["pose"] = RewardTermCfg(
    func=asimov_mdp.variable_posture_by_mode,
    weight=1.0,
    params={
      "asset_cfg": _leg_joints(),
      "command_name": "twist",
      "std_standing": {".*": 0.05},
      "std_walking": flat_task_walking_std,
      "std_running": flat_task_walking_std,
      "std_climbing": stairs_climbing_std,
      "walking_threshold": 0.05,
      "running_threshold": 1.5,
    },
  )

  cfg.rewards["upright"].params["asset_cfg"].body_names = (BASE_BODY,)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (BASE_BODY,)
  cfg.rewards["dof_pos_limits"].params["asset_cfg"] = _leg_joints()

  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = FOOT_SITE_NAMES

  cfg.rewards["body_ang_vel"].weight = -0.08
  cfg.rewards["angular_momentum"].weight = -0.03

  # foot_clearance / foot_swing_height: target_height 0.1 m (flat task's proven value)
  # when walking, 0.12 m (SE1/SE3's value: the tallest riser, 0.10 m, plus margin) when
  # climbing -- a foot swinging toward a higher tread needs to clear its leading edge,
  # not just reach the flat-ground target.
  cfg.rewards["foot_clearance"] = RewardTermCfg(
    func=asimov_mdp.feet_clearance_by_mode,
    weight=-2.0,
    params={
      "target_height_walking": 0.1,
      "target_height_climbing": 0.12,
      "height_sensor_name": "foot_height_scan",
      "command_name": "twist",
      "command_threshold": 0.05,
      "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
    },
  )
  cfg.rewards["foot_swing_height"] = RewardTermCfg(
    func=asimov_mdp.feet_swing_height_by_mode,
    weight=-2.0,
    params={
      "sensor_name": "feet_ground_contact",
      "height_sensor_name": "foot_height_scan",
      "target_height_walking": 0.1,
      "target_height_climbing": 0.12,
      "command_name": "twist",
      "command_threshold": 0.05,
    },
  )
  # air_time: the flat task's natural ~0.3s swing duration falls inside the mjlab
  # default window (threshold_max=0.5, weight=0.5); the first stairs policy's ~0.5-0.6s
  # swing duration sat right past that cutoff -- a dead zone with no reward, hence no
  # gradient pushing cadence up (see training/scripts/diagnose_gait.py and
  # TECHNICAL_REPORT.md §5.5). SE1's fix (threshold_max=0.35, weight=0.8) is now applied
  # only when climbing_mode=1; walking_mode keeps the flat task's untouched defaults.
  cfg.rewards["air_time"] = RewardTermCfg(
    func=asimov_mdp.feet_air_time_by_mode,
    weight=1.0,  # per-env weight is applied inside the function; see its docstring.
    params={
      "sensor_name": "feet_ground_contact",
      "threshold_min": 0.05,
      "threshold_max_walking": 0.5,
      "threshold_max_climbing": 0.35,
      "weight_walking": 0.5,
      "weight_climbing": 0.8,
      "command_name": "twist",
      "command_threshold": 0.5,
    },
  )
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
