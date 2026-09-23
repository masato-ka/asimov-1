"""Static tests for the climbing_mode-aware observation/reward functions (mdp.py).

Uses a small real ManagerBasedRlEnv (not a mock) since the reward/observation
functions need a real scene/terrain/command_manager, but avoids stepping physics:
joint positions and terrain_types are written directly so results are exact and fast,
rather than relying on a physics rollout (whose contact dynamics differ between flat
and stairs envs for reasons unrelated to the mode-selection logic under test).
"""

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab_asimov.robot.asimov_constants import LEG_JOINT_EXPR
from mjlab_asimov.tasks.asimov_stairs_env_cfg import asimov_stairs_env_cfg

NUM_ENVS = 4
# env 0,1 = flat (terrain_types index 0); env 2,3 = a stairs tier (index != 0).
FLAT_ENVS = (0, 1)
STAIRS_ENVS = (2, 3)


@pytest.fixture(scope="module")
def env() -> ManagerBasedRlEnv:
  cfg = asimov_stairs_env_cfg()
  cfg.scene.num_envs = NUM_ENVS
  e = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  e.reset()
  e.scene.terrain.terrain_types[:] = torch.tensor([0, 0, 1, 1])
  return e


def _reward_fn(env: ManagerBasedRlEnv, name: str):
  for term_name, term_cfg in zip(
    env.reward_manager._term_names, env.reward_manager._term_cfgs, strict=False
  ):
    if term_name == name:
      return term_cfg.func, term_cfg.params
  raise KeyError(name)


def test_climbing_mode_matches_terrain_type(env: ManagerBasedRlEnv):
  obs, _ = env.get_observations(), None
  actor = env.observation_manager.compute_group("actor")
  # climbing_mode is the last actor term (see asimov_stairs_env_cfg.py ordering).
  flag = actor[:, -1]
  for i in FLAT_ENVS:
    assert flag[i].item() == pytest.approx(0.0)
  for i in STAIRS_ENVS:
    assert flag[i].item() == pytest.approx(1.0)


def test_pose_reward_uses_looser_std_when_climbing(env: ManagerBasedRlEnv):
  """Same joint offset on every env; climbing (looser std) must be penalized less."""
  robot = env.scene["robot"]
  asset_cfg = SceneEntityCfg("robot", joint_names=LEG_JOINT_EXPR)
  asset_cfg.resolve(env.scene)

  default = robot.data.default_joint_pos.clone()
  offset = torch.zeros_like(default)
  offset[:, asset_cfg.joint_ids] = 0.3  # identical deviation for every env
  robot.write_joint_position_to_sim(default + offset)

  pose_fn, params = _reward_fn(env, "pose")
  value = pose_fn(env, **params)

  for i in FLAT_ENVS:
    for j in STAIRS_ENVS:
      assert value[j].item() > value[i].item(), (
        "climbing (looser std) should be penalized less than walking for the same "
        "joint deviation"
      )
  # Same terrain type -> identical reward (sanity check the mask, not just an
  # inequality that could pass by coincidence).
  assert value[FLAT_ENVS[0]].item() == pytest.approx(value[FLAT_ENVS[1]].item())
  assert value[STAIRS_ENVS[0]].item() == pytest.approx(value[STAIRS_ENVS[1]].item())


def test_foot_clearance_target_height_by_mode(env: ManagerBasedRlEnv):
  fn, params = _reward_fn(env, "foot_clearance")
  assert params["target_height_walking"] == 0.1
  assert params["target_height_climbing"] == 0.12
  # _blend_by_mode is exercised indirectly through the reward call in the rollout
  # smoke test (test_asimov_robot.py); here just confirm the mode mask itself.
  from mjlab_asimov.tasks.mdp import _climbing_mask

  mask = _climbing_mask(env)
  for i in FLAT_ENVS:
    assert mask[i].item() == 0.0
  for i in STAIRS_ENVS:
    assert mask[i].item() == 1.0


def test_air_time_thresholds_and_weights_by_mode(env: ManagerBasedRlEnv):
  _, params = _reward_fn(env, "air_time")
  assert params["threshold_max_walking"] == 0.5
  assert params["threshold_max_climbing"] == 0.35
  assert params["weight_walking"] == 0.5
  assert params["weight_climbing"] == 0.8
