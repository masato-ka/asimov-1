"""Task registration. Importing this package registers the Asimov-1 tasks with mjlab."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .asimov_rl_cfg import asimov_ppo_runner_cfg
from .asimov_stairs_env_cfg import asimov_stairs_env_cfg
from .asimov_stairs_rl_cfg import asimov_stairs_ppo_runner_cfg
from .asimov_velocity_env_cfg import asimov_flat_env_cfg

TASK_ID = "Asimov-Velocity-Flat"
TASK_ID_STAIRS = "Asimov-Velocity-Stairs"

register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=asimov_flat_env_cfg(),
  play_env_cfg=asimov_flat_env_cfg(play=True),
  rl_cfg=asimov_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id=TASK_ID_STAIRS,
  env_cfg=asimov_stairs_env_cfg(),
  play_env_cfg=asimov_stairs_env_cfg(play=True),
  rl_cfg=asimov_stairs_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
