"""Headless quantitative evaluation of a trained Asimov-1 velocity policy.

For each fixed velocity command it rolls out many envs (no noise, no pushes) and reports
the achieved body-frame velocity, falls and the stepping pattern (touchdowns per foot per
second, fraction of time both feet are airborne, left/right alternation).

Usage:
  uv run python training/scripts/eval_policy.py --checkpoint logs/rsl_rl/.../model_1499.pt
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

TASK_ID = "Asimov-Velocity-Flat"

# name -> (vx, vy, wz) body-frame command
SCENARIOS = {
  "stand": (0.0, 0.0, 0.0),
  "forward 0.4": (0.4, 0.0, 0.0),
  "forward 0.8": (0.8, 0.0, 0.0),
  "backward -0.4": (-0.4, 0.0, 0.0),
  "lateral 0.4": (0.0, 0.4, 0.0),
  "turn 0.5": (0.0, 0.0, 0.5),
}


def run(checkpoint: str, cmd: tuple[float, float, float], num_envs: int, steps: int):
  device = "cuda:0"
  env_cfg = load_env_cfg(TASK_ID, play=True)
  agent_cfg = load_rl_cfg(TASK_ID)
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
  runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  robot = env.unwrapped.scene["robot"]
  feet = env.unwrapped.scene["feet_ground_contact"]
  height_sensor = env.unwrapped.scene["foot_height_scan"]
  dt = env.unwrapped.step_dt
  warmup = int(1.0 / dt)

  obs = env.get_observations()
  vel, falls = [], 0
  contact_hist = []
  # Peak foot height per swing, recorded at each touchdown (after the warm-up second).
  peak = torch.zeros(num_envs, 2, device=device)
  prev_contact = torch.ones(num_envs, 2, dtype=torch.bool, device=device)
  peak_sum, peak_count = 0.0, 0
  for _ in range(steps):
    with torch.no_grad():
      actions = policy(obs)
    obs, _, dones, _ = env.step(actions)
    falls += int(dones.sum().item())
    vel.append(
      torch.cat(
        [robot.data.root_link_lin_vel_b[:, :2], robot.data.root_link_ang_vel_b[:, 2:3]],
        dim=1,
      )
    )
    contact = feet.data.found.reshape(num_envs, -1)[:, :2] > 0
    contact_hist.append(contact)

    heights = height_sensor.data.heights.reshape(num_envs, -1)[:, :2]
    peak = torch.where(~contact, torch.maximum(peak, heights), peak)
    landing = contact & ~prev_contact
    if len(contact_hist) > warmup:
      peak_sum += float((peak * landing).sum().item())
      peak_count += int(landing.sum().item())
    peak = torch.where(landing, torch.zeros_like(peak), peak)
    prev_contact = contact

  v = torch.stack(vel)[warmup:].mean(dim=(0, 1))
  c = torch.stack(contact_hist)[warmup:].float()  # (T, N, 2)
  seconds = c.shape[0] * dt
  touchdowns = ((c[1:] == 1) & (c[:-1] == 0)).float().sum(dim=0)  # (N, 2)
  both_air = ((c.sum(dim=2) == 0).float().mean()).item()
  return {
    "v": v.tolist(),
    "falls": falls,
    "touchdown_hz": (touchdowns.mean(dim=0) / seconds).tolist(),
    "both_air": both_air,
    "both_ground": ((c.sum(dim=2) == 2).float().mean()).item(),
    "peak_cm": 100.0 * peak_sum / max(peak_count, 1),
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--num-envs", type=int, default=64)
  parser.add_argument("--seconds", type=float, default=8.0)
  args = parser.parse_args()

  steps = int(args.seconds / 0.02)
  print(f"{'command (vx,vy,wz)':28s} {'achieved (vx,vy,wz)':26s} falls  step Hz L/R  both-air both-ground  swing-peak(cm)")
  for name, cmd in SCENARIOS.items():
    r = run(args.checkpoint, cmd, args.num_envs, steps)
    v = r["v"]
    print(
      f"{name:14s}{str(cmd):14s} ({v[0]:+.2f},{v[1]:+.2f},{v[2]:+.2f})".ljust(56)
      + f" {r['falls']:4d}   {r['touchdown_hz'][0]:.2f}/{r['touchdown_hz'][1]:.2f}"
      + f"     {r['both_air']:.2f}     {r['both_ground']:.2f}        {r['peak_cm']:5.1f}"
    )


if __name__ == "__main__":
  main()
