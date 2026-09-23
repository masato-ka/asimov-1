"""RSL-RL PPO configuration for the Asimov-1 stair-climbing task.

Same network/algorithm hyperparameters as the flat-walking task (asimov_rl_cfg.py) as
a starting point -- but note the actor input grows from 45-d to 232-d (a 17x11
height_scan heightmap is added, see asimov_stairs_env_cfg.py), so these hyperparameters
are an unvalidated carry-over, not a confirmed-good choice, for this task.

max_iterations is raised well above the flat task's 10_000: generator terrain costs
more per step (raycasting, more contact-capable geoms), and terrain_levels curriculum
learning is inherently slower than fixed-difficulty velocity tracking (each difficulty
row must be "earned" by walking far enough before harder rows are even seen).
"""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def asimov_stairs_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="asimov1_stairs",
    logger="wandb",
    wandb_project="asimov1-locomotion",
    wandb_tags=("asimov1", "velocity", "stairs"),
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=30_000,
  )
