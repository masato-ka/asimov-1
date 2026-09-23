"""Per-terrain-type gait diagnostic for the Asimov-1 stairs policy.

Unlike eval_policy.py's task-wide aggregate, this splits achieved velocity, per-foot
touchdown frequency and stride length by sub-terrain name (flat / easy_stairs /
moderate_stairs / challenging_stairs), because a single average can hide a policy that
walks fine on flat ground but poorly on stairs, or -- the failure mode found in the first
stairs training run -- one uniformly slow, low-cadence gait applied everywhere regardless
of terrain difficulty.

Only meaningful for Asimov-Velocity-Stairs (the flat task has no terrain types to split
by).

Usage:
  uv run python training/scripts/diagnose_gait.py --checkpoint logs/rsl_rl/.../model_X.pt
"""

import argparse
from dataclasses import asdict

import torch

import mjlab_asimov  # noqa: F401  (registers the tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab_asimov.tasks import TASK_ID_STAIRS


def run(checkpoint: str, cmd: tuple[float, float, float], num_envs: int, seconds: float):
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env_cfg = load_env_cfg(TASK_ID_STAIRS, play=True)
  agent_cfg = load_rl_cfg(TASK_ID_STAIRS)
  env_cfg.scene.num_envs = num_envs
  env_cfg.episode_length_s = 1e6

  twist = env_cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.ranges.lin_vel_x = (cmd[0], cmd[0])
  twist.ranges.lin_vel_y = (cmd[1], cmd[1])
  twist.ranges.ang_vel_z = (cmd[2], cmd[2])
  twist.rel_standing_envs = 0.0
  twist.rel_heading_envs = 0.0
  twist.rel_forward_envs = 0.0
  twist.heading_command = False
  twist.ranges.heading = None
  twist.resampling_time_range = (1e6, 1e6)

  env = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(cfg=env_cfg, device=device), clip_actions=agent_cfg.clip_actions
  )
  runner_cls = load_runner_cls(TASK_ID_STAIRS) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  robot = env.unwrapped.scene["robot"]
  feet = env.unwrapped.scene["feet_ground_contact"]
  terrain = env.unwrapped.scene.terrain
  assert terrain is not None
  sub_terrain_names = list(env_cfg.scene.terrain.terrain_generator.sub_terrains.keys())
  dt = env.unwrapped.step_dt
  warmup = int(1.0 / dt)
  steps = int(seconds / dt)

  obs = env.get_observations()
  # terrain_types are frozen after the single reset triggered by env construction (play
  # mode's `randomize_terrain` reset event only fires on reset, and falls are rare over
  # this short a rollout), so it's safe to snapshot them once up front.
  types = terrain.terrain_types.clone()

  vel_hist, contact_hist, fall_hist = [], [], []
  for _ in range(steps):
    with torch.no_grad():
      actions = policy(obs)
    obs, _, dones, _ = env.step(actions)
    vel_hist.append(
      torch.cat(
        [robot.data.root_link_lin_vel_b[:, :2], robot.data.root_link_ang_vel_b[:, 2:3]],
        dim=1,
      )
    )
    contact_hist.append(feet.data.found.reshape(num_envs, -1)[:, :2] > 0)
    fall_hist.append(dones.clone())

  v = torch.stack(vel_hist)[warmup:]  # (T, N, 3)
  c = torch.stack(contact_hist)[warmup:].float()  # (T, N, 2)
  falls = torch.stack(fall_hist)[warmup:]  # (T, N)
  touchdowns = (c[1:] == 1) & (c[:-1] == 0)  # (T-1, N, 2)
  seconds_elapsed = c.shape[0] * dt

  print(f"{'terrain':20s} {'n':>5s} {'vx':>8s} {'touchdown_hz':>13s} {'stride_m':>9s} {'falls':>6s}")
  for i, name in enumerate(sub_terrain_names):
    mask = types == i
    n = int(mask.sum().item())
    if n == 0:
      print(f"{name:20s} {n:5d}  (no envs landed on this terrain type)")
      continue
    vx = v[:, mask, 0].mean().item()
    # Per-foot touchdown rate (touchdowns summed over time+env, kept separate per foot,
    # then averaged across the two feet) -- summing across the foot dim too would double
    # count and silently report ~2x the true per-foot cadence.
    hz = (
      touchdowns[:, mask].float().sum(dim=(0, 1)) / (mask.sum() * seconds_elapsed)
    ).mean().item()
    stride = vx / hz if hz > 1e-6 else float("nan")
    n_falls = int(falls[:, mask].sum().item())
    print(f"{name:20s} {n:5d} {vx:+8.3f} {hz:13.3f} {stride:9.3f} {n_falls:6d}")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--vx", type=float, default=0.4)
  parser.add_argument("--vy", type=float, default=0.0)
  parser.add_argument("--wz", type=float, default=0.0)
  parser.add_argument("--num-envs", type=int, default=800)
  parser.add_argument("--seconds", type=float, default=8.0)
  args = parser.parse_args()
  run(args.checkpoint, (args.vx, args.vy, args.wz), args.num_envs, args.seconds)


if __name__ == "__main__":
  main()
