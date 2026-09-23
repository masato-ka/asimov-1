"""Ascent-only stairs terrain for Asimov-1.

mjlab's shipped ``STAIRS_TERRAINS_CFG`` (mjlab.terrains.config) only uses the
non-inverted ``pyramid_stairs`` preset. ``BoxPyramidStairsTerrainCfg.function()``
places each sub-terrain's spawn origin at ``(num_steps + 1) * step_height`` — the
platform at the *top* of the pyramid — so walking outward from spawn always goes
downhill. For a climbing task we want the opposite: ``BoxInvertedPyramidStairsTerrainCfg``
(the ``pyramid_stairs_inv`` preset) places the origin at
``-(num_steps + 1) * step_height``, the bottom of a pit, so a forward velocity
command drives the robot up and out — a genuine ascent.

This mirrors ``STAIRS_TERRAINS_CFG``'s three difficulty tiers exactly, just built
from ``pyramid_stairs_inv`` instead of ``pyramid_stairs``. A fresh
``TerrainGeneratorCfg`` is constructed (not a ``dataclasses.replace()`` of the
shipped config) because ``sub_terrains`` is a plain dict on a module-level
singleton; mutating a shallow copy would still mutate the shared original.

A ``flat`` tile is mixed in so (a) the robot gets a chance to walk on totally flat
ground with the stairs-sized 232-d observation before ``terrain_levels_vel``
promotes it onto stairs, and (b) it serves as the easiest curriculum row.

To later add descent, add sibling ``pyramid_stairs`` (non-inverted) columns here —
``terrain_levels_vel`` logs curriculum progress per sub-terrain name automatically.
"""

from mjlab.terrains import TerrainGeneratorCfg
from mjlab.terrains.config import flat, pyramid_stairs_inv

ASIMOV_STAIRS_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=10,
  curriculum=True,  # num_cols is forced to len(sub_terrains) = 4.
  sub_terrains={
    "flat": flat(proportion=0.25),
    "easy_stairs": pyramid_stairs_inv(
      proportion=0.35, step_height_range=(0.02, 0.05), step_width=0.40
    ),
    "moderate_stairs": pyramid_stairs_inv(
      proportion=0.25,
      step_height_range=(0.05, 0.08),
      step_width=0.35,
      platform_width=2.5,
      border_width=0.8,
    ),
    "challenging_stairs": pyramid_stairs_inv(
      proportion=0.15,
      step_height_range=(0.08, 0.10),
      step_width=0.30,
      platform_width=2.0,
      border_width=0.5,
    ),
  },
  add_lights=True,
)


if __name__ == "__main__":
  import mujoco.viewer

  from mjlab.terrains import TerrainEntity, TerrainEntityCfg

  terrain = TerrainEntity(
    TerrainEntityCfg(terrain_type="generator", terrain_generator=ASIMOV_STAIRS_TERRAINS_CFG),
    device="cpu",
  )
  mujoco.viewer.launch(terrain.spec.compile())
