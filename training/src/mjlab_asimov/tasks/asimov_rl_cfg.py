"""RSL-RL PPO configuration for the Asimov-1 velocity task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def asimov_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    # Menlo guide: actor MLP 45 -> 512 -> 256 -> 128 -> 12 with ELU.
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
    experiment_name="asimov1_velocity",
    # Metrics, config and checkpoints (.pt / .onnx) go to Weights & Biases (mjlab's
    # standard logger). Override per run with e.g. --agent.logger tensorboard,
    # --agent.wandb-project <name>, --agent.run-name <label>, --agent.upload-model False.
    logger="wandb",
    wandb_project="asimov1-locomotion",
    wandb_tags=("asimov1", "velocity", "flat"),
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=10_000,
  )
