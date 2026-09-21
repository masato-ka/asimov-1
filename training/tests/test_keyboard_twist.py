"""KeyboardTwist / terminal key parsing (no GPU or display needed)."""

import os
import pty
import termios
import time

import pytest

from mjlab_asimov.scripts.play_keyboard import KeyboardTwist, TerminalKeys, parse_keys

LIMITS = (0.8, 0.6, 0.6)

UP, DOWN, RIGHT, LEFT = b"\x1b[A", b"\x1b[B", b"\x1b[C", b"\x1b[D"


def press(twist: KeyboardTwist, key: str, times: int = 1) -> None:
  for _ in range(times):
    twist.on_key(key)


# --- KeyboardTwist ---------------------------------------------------------------


def test_starts_at_zero():
  assert KeyboardTwist(LIMITS).snapshot() == (0.0, 0.0, 0.0)


def test_each_key_moves_its_axis_by_one_step():
  t = KeyboardTwist(LIMITS, step=0.1)
  press(t, "up")
  assert t.snapshot() == (0.1, 0.0, 0.0)
  press(t, "a")  # strafe left = +vy
  assert t.snapshot() == (0.1, 0.1, 0.0)
  press(t, "left")  # turn left = +wz
  assert t.snapshot() == (0.1, 0.1, 0.1)
  press(t, "down", 2)
  press(t, "e", 2)  # strafe right = -vy
  press(t, "right", 2)  # turn right = -wz
  assert t.snapshot() == (-0.1, -0.1, -0.1)


def test_clamped_to_training_limits():
  t = KeyboardTwist(LIMITS, step=0.1)
  press(t, "up", 20)
  press(t, "a", 20)
  press(t, "right", 20)
  assert t.snapshot() == (0.8, 0.6, -0.6)


def test_no_float_drift_after_many_steps():
  t = KeyboardTwist(LIMITS, step=0.1)
  press(t, "up", 7)
  press(t, "down", 7)
  assert t.snapshot() == (0.0, 0.0, 0.0)


def test_space_zeroes_every_command():
  t = KeyboardTwist(LIMITS)
  press(t, "up", 3)
  press(t, "a", 2)
  press(t, "left", 4)
  press(t, "space")
  assert t.snapshot() == (0.0, 0.0, 0.0)


def test_unknown_keys_are_ignored():
  t = KeyboardTwist(LIMITS)
  press(t, "up")
  press(t, "w")
  press(t, "x")
  assert t.snapshot() == (0.1, 0.0, 0.0)


def test_on_change_only_fires_on_actual_change():
  seen = []
  t = KeyboardTwist(LIMITS, step=0.1, on_change=seen.append)
  press(t, "space")  # already zero: no change
  press(t, "up")
  press(t, "w")  # not ours
  press(t, "up", 10)  # hits the 0.8 limit, then stops changing
  assert seen[0] == (0.1, 0.0, 0.0)
  assert seen[-1] == (0.8, 0.0, 0.0)
  assert len(seen) == 8


@pytest.mark.parametrize("step", [0.05, 0.2])
def test_custom_step(step):
  t = KeyboardTwist(LIMITS, step=step)
  press(t, "up")
  assert t.snapshot()[0] == pytest.approx(step)


# --- parse_keys ------------------------------------------------------------------


def test_parse_arrows_normal_and_application_mode():
  assert parse_keys(UP + DOWN + RIGHT + LEFT) == ["up", "down", "right", "left"]
  assert parse_keys(b"\x1bOA\x1bOB\x1bOC\x1bOD") == ["up", "down", "right", "left"]


def test_parse_letters_case_insensitive_and_space():
  assert parse_keys(b"aAeE ") == ["a", "a", "e", "e", "space"]


def test_parse_batched_auto_repeat():
  assert parse_keys(UP * 5) == ["up"] * 5


def test_parse_ignores_other_keys():
  assert parse_keys(b"wsdqx1\n\t") == []


def test_parse_skips_whole_escape_sequences():
  # Ctrl-Up (ESC [ 1 ; 5 A), F5 (ESC [ 1 5 ~) and Delete (ESC [ 3 ~) must not leak
  # their trailing 'A' or digits as keys; the up arrow that follows still registers.
  assert parse_keys(b"\x1b[1;5A" + b"\x1b[15~" + b"\x1b[3~" + UP) == ["up"]
  assert parse_keys(b"\x1b") == []  # lone ESC


# --- TerminalKeys (through a pseudo terminal) ---------------------------------


def _wait_for(pred, timeout=2.0):
  end = time.time() + timeout
  while time.time() < end:
    if pred():
      return True
    time.sleep(0.01)
  return False


def test_terminal_keys_drive_the_command_and_restore_the_terminal():
  master, slave = pty.openpty()
  try:
    before = termios.tcgetattr(slave)
    t = KeyboardTwist(LIMITS, step=0.1)
    with TerminalKeys(t, fd=slave):
      os.write(master, UP * 3)
      assert _wait_for(lambda: t.snapshot() == pytest.approx((0.3, 0.0, 0.0)))
      os.write(master, b"a" + LEFT)
      assert _wait_for(lambda: t.snapshot() == pytest.approx((0.3, 0.1, 0.1)))
      os.write(master, b" ")
      assert _wait_for(lambda: t.snapshot() == (0.0, 0.0, 0.0))
    assert termios.tcgetattr(slave) == before
  finally:
    os.close(master)
    os.close(slave)


def test_terminal_keys_require_a_tty():
  r, w = os.pipe()
  try:
    with pytest.raises(RuntimeError, match="terminal"):
      with TerminalKeys(KeyboardTwist(LIMITS), fd=r):
        pass
  finally:
    os.close(r)
    os.close(w)
