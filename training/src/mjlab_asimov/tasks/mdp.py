"""Climbing-mode-aware observation/reward functions for the Asimov-1 stairs task.

Diagnosis of the first stairs policy (see training/TECHNICAL_REPORT.md §5.5) found it
converged to one uniform, slow gait applied on every terrain type -- including flat
cells within its own training population -- because nothing in the observation or
reward explicitly told it to behave differently on stairs vs. flat ground. The
232-dimensional `height_scan` heightmap technically contains terrain-discriminating
information, but the policy never learned to exploit it for switching gait style, and
reward terms were applied identically regardless of terrain.

Rather than have the policy infer "am I on stairs?" from noisy raw sensor data, this
module derives it directly from ground truth (each environment spawns on, and stays on,
exactly one terrain type for its whole episode -- `env.scene.terrain.terrain_types` is
constant per env per episode, so no online classification is needed) and threads a
`climbing_mode` flag through both a small observation term and mode-aware variants of
the reward terms diagnosed as needing different tuning per mode (`pose`, `foot_clearance`
/`foot_swing_height`, `air_time`).

Caveat: `climbing_mode`, like `height_scan`, is a simulation-only privileged signal with
no real-hardware analog yet (the real robot has no depth/terrain sensor at all, see
TECHNICAL_REPORT.md). This is an accepted simplification for training, not a deployment-
ready mechanism.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.utils.lab_api.string import resolve_matching_names_values

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _climbing_mask(env: ManagerBasedRlEnv, flat_index: int = 0) -> torch.Tensor:
  """Per-env 0.0/1.0: 1.0 if the env's terrain patch is a stairs tier, not flat.

  Requires generator terrain (`env.scene.terrain.terrain_types` populated); only
  wired into the stairs task, never the flat task.
  """
  terrain = env.scene.terrain
  assert terrain is not None and terrain.terrain_types is not None, (
    "_climbing_mask requires generator terrain with terrain_types populated."
  )
  return (terrain.terrain_types != flat_index).float()


def climbing_mode(env: ManagerBasedRlEnv, flat_index: int = 0) -> torch.Tensor:
  """Observation term: [B, 1] climbing-mode flag, fed alongside the twist command."""
  return _climbing_mask(env, flat_index).unsqueeze(-1)


class variable_posture_by_mode:
  """Like mjlab's ``variable_posture``, plus a 4th std table selected by climbing mode.

  When climbing_mode=1, ``std_climbing`` is used regardless of commanded speed
  (climbing is inherently a deliberate motion; the standing/walking/running speed
  bands are a distinction that matters mainly for flat-ground locomotion). When
  climbing_mode=0, falls back to mjlab's original speed-based std_standing/
  std_walking/std_running blend.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    def _resolve(key: str) -> torch.Tensor:
      _, _, values = resolve_matching_names_values(
        data=cfg.params[key], list_of_strings=joint_names
      )
      return torch.tensor(values, device=env.device, dtype=torch.float32)

    self.std_standing = _resolve("std_standing")
    self.std_walking = _resolve("std_walking")
    self.std_running = _resolve("std_running")
    self.std_climbing = _resolve("std_climbing")

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    std_climbing,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
    flat_index: int = 0,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running, std_climbing  # Resolved in __init__.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    climbing_mask = _climbing_mask(env, flat_index).bool().unsqueeze(1)
    std = torch.where(
      climbing_mask, self.std_climbing.unsqueeze(0).expand_as(std), std
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))


def _blend_by_mode(
  env: ManagerBasedRlEnv, walking_value: float, climbing_value: float, flat_index: int
) -> torch.Tensor:
  """[B] tensor, walking_value on flat envs, climbing_value on stairs envs."""
  climbing = _climbing_mask(env, flat_index)
  return walking_value + climbing * (climbing_value - walking_value)


def feet_clearance_by_mode(
  env: ManagerBasedRlEnv,
  target_height_walking: float,
  target_height_climbing: float,
  height_sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  flat_index: int = 0,
) -> torch.Tensor:
  """Like mjlab's ``feet_clearance``, with target_height selected by climbing mode."""
  asset: Entity = env.scene[asset_cfg.name]
  height_sensor = env.scene[height_sensor_name]
  assert isinstance(height_sensor, TerrainHeightSensor)
  foot_height = height_sensor.data.heights  # [B, F]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, F, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, F]
  target = _blend_by_mode(
    env, target_height_walking, target_height_climbing, flat_index
  )  # [B]
  delta = torch.abs(foot_height - target.unsqueeze(1))  # [B, F]
  cost = torch.sum(delta * vel_norm, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class feet_swing_height_by_mode:
  """Like mjlab's ``feet_swing_height``, with target_height selected by climbing mode."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    height_sensor = env.scene[cfg.params["height_sensor_name"]]
    assert isinstance(height_sensor, TerrainHeightSensor)
    num_feet = height_sensor.num_frames
    self.peak_heights = torch.zeros(
      (env.num_envs, num_feet), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    height_sensor_name: str,
    target_height_walking: float,
    target_height_climbing: float,
    command_name: str,
    command_threshold: float,
    flat_index: int = 0,
  ) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    height_sensor: TerrainHeightSensor = env.scene[height_sensor_name]
    foot_heights = height_sensor.data.heights
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air, torch.maximum(self.peak_heights, foot_heights), self.peak_heights
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    target = _blend_by_mode(
      env, target_height_walking, target_height_climbing, flat_index
    )  # [B]
    error = self.peak_heights / target.unsqueeze(1) - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    self.peak_heights = torch.where(
      first_contact, torch.zeros_like(self.peak_heights), self.peak_heights
    )
    return cost


def feet_air_time_by_mode(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold_min: float = 0.05,
  threshold_max_walking: float = 0.5,
  threshold_max_climbing: float = 0.35,
  weight_walking: float = 0.5,
  weight_climbing: float = 0.8,
  command_name: str | None = None,
  command_threshold: float = 0.5,
  flat_index: int = 0,
) -> torch.Tensor:
  """Like mjlab's ``feet_air_time``, with threshold_max and weight selected by mode.

  The per-env weight is folded into the returned value (rather than left to the
  RewardTermCfg's single scalar weight, which cannot vary per env) -- set this term's
  own `weight=1.0` in the RewardTermCfg.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  current_air_time = sensor.data.current_air_time
  assert current_air_time is not None
  threshold_max = _blend_by_mode(
    env, threshold_max_walking, threshold_max_climbing, flat_index
  )  # [B]
  in_range = (current_air_time > threshold_min) & (
    current_air_time < threshold_max.unsqueeze(1)
  )
  reward = torch.sum(in_range.float(), dim=1)
  per_env_weight = _blend_by_mode(env, weight_walking, weight_climbing, flat_index)
  reward = reward * per_env_weight
  in_air = current_air_time > 0
  num_in_air = torch.sum(in_air.float())
  mean_air_time = torch.sum(current_air_time * in_air.float()) / torch.clamp(
    num_in_air, min=1
  )
  env.extras["log"]["Metrics/air_time_mean"] = mean_air_time
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command > command_threshold).float()
      reward = reward * scale
  return reward
