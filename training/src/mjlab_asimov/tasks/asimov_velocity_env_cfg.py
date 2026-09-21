"""Asimov-1 flat-ground velocity-tracking task.

Built on mjlab's ``make_velocity_env_cfg()`` template (same pattern as the Unitree G1
config shipped with mjlab): create the template, then customise it for this robot.

Design points (see the Menlo Asimov locomotion guide for the source of most numbers):

* Action space: the 12 leg joints only. Arms / waist are held by passive springs.
* Actor observation (45-d): base angular velocity, projected gravity, velocity command,
  12 leg joint positions, 12 leg joint velocities, previous action. Base linear velocity
  is *not* given to the actor because the real robot cannot measure it; the critic
  (asymmetric actor-critic) still sees it together with foot contact information.
* Physics 200 Hz, policy 50 Hz (decimation 4).
* Targeted domain randomization only: encoder offset, PD gains, foot friction, pushes,
  observation / actuator latency. No mass, link-length or gravity randomization.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
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

# The pelvis carries the IMU; the upper body is rigidly held on top of it.
BASE_BODY = "pelvis_link"


def _leg_joints() -> SceneEntityCfg:
  return SceneEntityCfg("robot", joint_names=LEG_JOINT_EXPR)


def asimov_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Asimov-1 flat terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  ##
  # Scene: robot on flat ground.
  ##

  cfg.scene.entities = {"robot": get_asimov_robot_cfg()}

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # No terrain to scan on flat ground.
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )

  for sensor in cfg.scene.sensors:
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
  cfg.scene.sensors = cfg.scene.sensors + (feet_ground_cfg, self_collision_cfg)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  cfg.viewer.body_name = BASE_BODY

  ##
  # Actions: 12 leg joints, per-joint scale = 0.30 * effort_limit / stiffness.
  ##

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  # The passive arm / waist joints have no actuator, so ".*" matches the legs only.
  joint_pos_action.scale = ASIMOV_ACTION_SCALE

  ##
  # Observations.
  ##

  actor = cfg.observations["actor"]
  critic = cfg.observations["critic"]

  # The template's height scan is for rough terrain.
  actor.terms.pop("height_scan", None)
  critic.terms.pop("height_scan", None)
  # Privileged: only the critic gets the ground-truth base linear velocity.
  actor.terms.pop("base_lin_vel")

  # Sensor noise levels from the Menlo guide.
  actor.terms["base_ang_vel"].noise = Unoise(n_min=-0.01, n_max=0.01)
  actor.terms["projected_gravity"].noise = Unoise(n_min=-0.05, n_max=0.05)
  actor.terms["joint_pos"].noise = Unoise(n_min=-0.01, n_max=0.01)
  actor.terms["joint_vel"].noise = Unoise(n_min=-0.1, n_max=0.1)

  # Only the 12 leg joints are observed (the passive upper body is not part of the
  # policy interface).
  for group in (actor, critic):
    group.terms["joint_pos"].params["asset_cfg"] = _leg_joints()
    group.terms["joint_vel"].params["asset_cfg"] = _leg_joints()

  # Joint feedback arrives late on the real robot (CAN polling). Placeholder: one policy
  # step of jitter for every joint; the guide uses tiers of 0-2 / 0-1 / 0 steps depending
  # on the bus polling order, which is not known for Asimov-1 yet.
  for name in ("joint_pos", "joint_vel"):
    actor.terms[name].delay_min_lag = 0
    actor.terms[name].delay_max_lag = 1

  ##
  # Commands: fixed ranges (no curriculum).
  ##

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_x = (-0.8, 0.8)
  twist_cmd.ranges.lin_vel_y = (-0.6, 0.6)
  twist_cmd.ranges.ang_vel_z = (-0.6, 0.6)
  twist_cmd.viz.z_offset = 1.05

  cfg.curriculum = {}

  ##
  # Events / domain randomization.
  ##

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_GEOM_NAMES
  cfg.events["foot_friction"].params["ranges"] = (1.0, 1.5)
  cfg.events["encoder_bias"].params["bias_range"] = (-0.02, 0.02)
  # The guide deliberately does not randomize mass / geometry / gravity.
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
  # Velocity pushes of +-0.5 m/s in the horizontal plane every few seconds.
  cfg.events["push_robot"].interval_range_s = (4.0, 8.0)
  cfg.events["push_robot"].params["velocity_range"] = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
  }

  ##
  # Rewards.
  ##

  # Per-joint std of the pose prior (reward = exp(-mean(err^2 / std^2)) for staying close
  # to the home pose). Roll / yaw stay tight to limit lateral sway. The sagittal joints
  # (hip pitch, knee, ankle pitch) are twice the G1 values (0.3 / 0.35 / 0.25): with the
  # G1 values this term was the strongest force against lifting the swing foot and the
  # policy kept the foot ~3.5 cm off the ground while walking at 0.4 m/s.
  walking_std = {
    r".*hip_pitch.*": 0.6,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.7,
    r".*ankle_pitch.*": 0.5,
    r".*ankle_roll.*": 0.1,
  }
  pose = cfg.rewards["pose"].params
  pose["asset_cfg"] = _leg_joints()
  pose["std_standing"] = {".*": 0.05}
  pose["std_walking"] = walking_std
  # Commands never reach running speed for this task.
  pose["std_running"] = walking_std

  cfg.rewards["upright"].params["asset_cfg"].body_names = (BASE_BODY,)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (BASE_BODY,)
  cfg.rewards["dof_pos_limits"].params["asset_cfg"] = _leg_joints()

  for reward_name in ("foot_clearance", "foot_slip"):
    cfg.rewards[reward_name].params["asset_cfg"].site_names = FOOT_SITE_NAMES

  # Narrow-stance stability penalties (Menlo: -0.08 body ang vel, -0.03 ang momentum).
  cfg.rewards["body_ang_vel"].weight = -0.08
  cfg.rewards["angular_momentum"].weight = -0.03
  # Gait shaping. Swing height is penalised at landing (strongly, the template's -0.25 was
  # ~1 % of the tracking reward), and time in the air is rewarded; the latter also raised
  # the achieved yaw rate when turning. Effort / smoothness penalties are lower than the
  # Menlo guide's values so that lifting the foot is not the expensive option.
  cfg.rewards["foot_swing_height"].weight = -2.0
  cfg.rewards["air_time"].weight = 0.5
  cfg.rewards["action_rate_l2"].weight = -0.03

  cfg.rewards["torques"] = RewardTermCfg(func=mdp.joint_torques_l2, weight=-5e-5)
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  ##
  # Terminations.
  ##

  cfg.terminations.pop("out_of_terrain_bounds", None)

  ##
  # Play mode: no noise, no pushes, effectively endless episodes.
  ##

  if play:
    cfg.episode_length_s = int(1e9)
    actor.enable_corruption = False
    cfg.events.pop("push_robot", None)

  return cfg
