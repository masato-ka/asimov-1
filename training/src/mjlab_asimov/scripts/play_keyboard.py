"""Play a trained Asimov-1 velocity policy and steer it with the keyboard.

Type in the TERMINAL that launched this script (not in the viewer window):

  Up / Down      forward / backward      (vx +/- step)
  Left / Right   turn left / right       (wz +/- step)
  A / E          strafe left / right     (vy +/- step)
  Space          zero every command
  C              toggle climbing_mode (stairs task only; overrides the terrain's own
                 ground-truth flag, for exploring how the policy responds to each mode)

Keys are read from the terminal because the MuJoCo viewer window binds all 26 letters
(and Space, Left/Right, ...) to display toggles; keeping the keys out of the window means
nothing in the viewer changes. Each key press (or auto-repeat) changes the command by
``--step``. The command is clamped to the range the policy was trained on.

Works with any registered task via ``--task-id`` (default: the flat velocity task):

  uv run asimov-play-keyboard --checkpoint logs/rsl_rl/asimov1_velocity/<run>/model_1499.pt
  uv run asimov-play-keyboard --task-id Asimov-Velocity-Stairs --step 0.02 \\
      --checkpoint logs/rsl_rl/asimov1_stairs/<run>/model_29999.pt

The stairs task's command ranges are much narrower than the flat task's (e.g. vy is
only +-0.1 m/s, vs. +-0.6 on flat) -- the default ``--step`` of 0.1 will jump straight to
a strafe axis's limit in a single keypress there, so pass a smaller ``--step`` (e.g.
0.02) for finer control on narrow-range tasks.
"""

import argparse
import os
import select
import sys
import termios
import threading
import tty
from dataclasses import asdict
from typing import Callable

import torch

import mjlab_asimov  # noqa: F401  (registers the tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.viewer import NativeMujocoViewer
from mjlab_asimov.tasks import TASK_ID

# key token -> (index into (vx, vy, wz), direction)
_KEY_TO_AXIS = {
  "up": (0, +1.0),
  "down": (0, -1.0),
  "a": (1, +1.0),
  "e": (1, -1.0),
  "left": (2, +1.0),
  "right": (2, -1.0),
}
STOP_KEY = "space"
MODE_TOGGLE_KEY = "c"

HELP = """\
Keyboard twist control -- type in THIS terminal (the viewer window is display only):
  Up / Down     forward / backward
  Left / Right  turn left / right
  A / E         strafe left / right
  Space         zero every command
  Ctrl-C        quit"""

HELP_CLIMBING_MODE = "  C             toggle climbing_mode (currently walking/0)"

_ARROW_FINALS = {ord("A"): "up", ord("B"): "down", ord("C"): "right", ord("D"): "left"}


def parse_keys(data: bytes) -> list[str]:
  """Turn raw terminal bytes into key tokens ("up", "a", "space", ...).

  Arrow keys arrive as ``ESC [ A..D`` (or ``ESC O A..D`` in application mode). Several
  keys may be in one read (auto-repeat), and other escape sequences (function keys,
  modified arrows) are skipped whole so their trailing letters are not mistaken for keys.
  """
  keys: list[str] = []
  i, n = 0, len(data)
  while i < n:
    b = data[i]
    if b == 0x1B:  # ESC
      if i + 1 < n and data[i + 1] == ord("["):  # CSI: params/intermediates, then a final byte
        j = i + 2
        while j < n and 0x20 <= data[j] <= 0x3F:
          j += 1
        if j < n and j == i + 2 and data[j] in _ARROW_FINALS:
          keys.append(_ARROW_FINALS[data[j]])
        i = j + 1
      elif i + 2 < n and data[i + 1] == ord("O"):  # SS3
        if data[i + 2] in _ARROW_FINALS:
          keys.append(_ARROW_FINALS[data[i + 2]])
        i += 3
      else:
        i += 1
      continue
    if b == 0x20:
      keys.append(STOP_KEY)
    elif b < 0x80 and chr(b).lower() in ("a", "e", MODE_TOGGLE_KEY):
      keys.append(chr(b).lower())
    i += 1
  return keys


class KeyboardTwist:
  """Holds the commanded twist (vx, vy, wz) and updates it from key tokens.

  ``on_key`` runs on the terminal-reader thread while ``snapshot`` is read from the
  simulation thread, so all access goes through a lock.
  """

  def __init__(
    self,
    limits: tuple[float, float, float],
    step: float = 0.1,
    on_change: Callable[[tuple[float, float, float]], None] | None = None,
  ):
    self._limits = limits
    self._step = step
    self._on_change = on_change
    self._cmd = [0.0, 0.0, 0.0]
    self._lock = threading.Lock()

  def snapshot(self) -> tuple[float, float, float]:
    with self._lock:
      return (self._cmd[0], self._cmd[1], self._cmd[2])

  def on_key(self, key: str) -> None:
    with self._lock:
      old = list(self._cmd)
      if key == STOP_KEY:
        self._cmd = [0.0, 0.0, 0.0]
      elif key in _KEY_TO_AXIS:
        axis, direction = _KEY_TO_AXIS[key]
        limit = self._limits[axis]
        value = self._cmd[axis] + direction * self._step
        # Round so repeated +/- steps do not accumulate float error.
        self._cmd[axis] = round(max(-limit, min(limit, value)), 6)
      else:
        return
      changed = self._cmd != old
      new = (self._cmd[0], self._cmd[1], self._cmd[2])
    if changed and self._on_change is not None:
      self._on_change(new)


class ClimbingModeToggle:
  """Lets a human override the stairs task's ground-truth ``climbing_mode`` flag.

  ``climbing_mode`` (see mjlab_asimov.tasks.mdp) is normally derived from the terrain
  patch each env actually spawned on. This toggle overwrites that per-env ground truth
  directly (``env.scene.terrain.terrain_types``) so a human can ask "how does the
  policy behave if told it's climbing / walking", independent of the actual terrain
  underfoot. No-op for tasks without generator terrain (e.g. the flat task).
  """

  def __init__(self, on_change: Callable[[bool], None] | None = None):
    self._mode = False
    self._lock = threading.Lock()
    self._on_change = on_change

  def snapshot(self) -> bool:
    with self._lock:
      return self._mode

  def on_key(self, key: str) -> None:
    if key != MODE_TOGGLE_KEY:
      return
    with self._lock:
      self._mode = not self._mode
      new = self._mode
    if self._on_change is not None:
      self._on_change(new)


class KeyboardPolicy:
  """Wraps a policy and writes the keyboard command into the env before each call."""

  def __init__(
    self,
    policy,
    command_term,
    twist: KeyboardTwist,
    mode_toggle: ClimbingModeToggle | None = None,
    terrain=None,
  ):
    self._policy = policy
    self._term = command_term
    self._twist = twist
    self._mode_toggle = mode_toggle
    self._terrain = terrain

  def __call__(self, obs):
    cmd = self._term.vel_command_b
    cmd[:] = torch.tensor(self._twist.snapshot(), device=cmd.device, dtype=cmd.dtype)
    if self._mode_toggle is not None and self._terrain is not None:
      # 1 = any stairs tier (climbing_mode only checks "!= flat index 0").
      self._terrain.terrain_types[:] = 1 if self._mode_toggle.snapshot() else 0
    return self._policy(obs)

  def reset(self, *args, **kwargs):
    reset = getattr(self._policy, "reset", None)
    if reset is not None:
      return reset(*args, **kwargs)


class TerminalKeys:
  """Feeds key presses typed in the terminal to one or more ``on_key(str)`` listeners.

  Puts the terminal in cbreak mode (no line buffering, no echo, Ctrl-C still works) and
  reads it on a daemon thread; the original terminal settings are always restored.
  """

  def __init__(self, *listeners, fd: int | None = None):
    self._listeners = [listener for listener in listeners if listener is not None]
    self._fd = sys.stdin.fileno() if fd is None else fd
    self._saved = None
    self._stop = threading.Event()
    self._thread: threading.Thread | None = None

  def __enter__(self) -> "TerminalKeys":
    if not os.isatty(self._fd):
      raise RuntimeError(
        "stdin is not a terminal: run asimov-play-keyboard from an interactive terminal "
        "(keys are read from it)."
      )
    self._saved = termios.tcgetattr(self._fd)
    tty.setcbreak(self._fd)
    self._stop.clear()
    self._thread = threading.Thread(target=self._loop, daemon=True)
    self._thread.start()
    return self

  def _loop(self) -> None:
    while not self._stop.is_set():
      ready, _, _ = select.select([self._fd], [], [], 0.05)
      if not ready:
        continue
      try:
        data = os.read(self._fd, 64)
      except OSError:
        break
      if not data:
        break
      for key in parse_keys(data):
        for listener in self._listeners:
          listener.on_key(key)

  def __exit__(self, *exc) -> None:
    self._stop.set()
    if self._thread is not None:
      self._thread.join(timeout=1.0)
    if self._saved is not None:
      termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)


def _print_command(cmd: tuple[float, float, float]) -> None:
  print(f"cmd vx={cmd[0]:+.2f} vy={cmd[1]:+.2f} wz={cmd[2]:+.2f}", flush=True)


def build_play(checkpoint: str, task_id: str = TASK_ID):
  """Build the single-env play environment and policy with a keyboard-owned command.

  Returns (env, policy, command_term, limits) where limits = (vx, vy, wz) maxima taken
  from the ranges the policy was trained on.
  """
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  env_cfg.scene.num_envs = 1

  twist_cfg = env_cfg.commands["twist"]
  assert isinstance(twist_cfg, UniformVelocityCommandCfg)
  limits = (
    max(abs(v) for v in twist_cfg.ranges.lin_vel_x),
    max(abs(v) for v in twist_cfg.ranges.lin_vel_y),
    max(abs(v) for v in twist_cfg.ranges.ang_vel_z),
  )
  # Stop the command manager from resampling / rewriting the command: the keyboard owns it.
  twist_cfg.ranges.lin_vel_x = (0.0, 0.0)
  twist_cfg.ranges.lin_vel_y = (0.0, 0.0)
  twist_cfg.ranges.ang_vel_z = (0.0, 0.0)
  twist_cfg.rel_standing_envs = 0.0
  twist_cfg.rel_heading_envs = 0.0
  twist_cfg.rel_forward_envs = 0.0
  twist_cfg.heading_command = False
  twist_cfg.ranges.heading = None
  twist_cfg.resampling_time_range = (1e6, 1e6)

  env = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(cfg=env_cfg, device=device), clip_actions=agent_cfg.clip_actions
  )
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)
  command_term = env.unwrapped.command_manager.get_term("twist")
  return env, policy, command_term, limits


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  parser.add_argument("--checkpoint", required=True, help="path to a model_*.pt")
  parser.add_argument("--task-id", default=TASK_ID)
  parser.add_argument("--step", type=float, default=0.1, help="command change per key")
  args = parser.parse_args()

  env, policy, command_term, limits = build_play(args.checkpoint, args.task_id)
  twist = KeyboardTwist(limits, step=args.step, on_change=_print_command)
  print(HELP)
  print(f"limits: vx +-{limits[0]}, vy +-{limits[1]}, wz +-{limits[2]}; step {args.step}")
  _print_command(twist.snapshot())

  terrain = env.unwrapped.scene.terrain
  has_climbing_mode = terrain is not None and terrain.terrain_types is not None
  mode_toggle = None
  if has_climbing_mode:

    def _print_mode(mode: bool) -> None:
      print(f"climbing_mode={int(mode)}", flush=True)

    mode_toggle = ClimbingModeToggle(on_change=_print_mode)
    print(HELP_CLIMBING_MODE)

  keyboard_policy = KeyboardPolicy(
    policy, command_term, twist, mode_toggle, terrain if has_climbing_mode else None
  )
  viewer = NativeMujocoViewer(env, keyboard_policy)
  with TerminalKeys(twist, mode_toggle):
    viewer.run()
  env.close()


if __name__ == "__main__":
  main()
