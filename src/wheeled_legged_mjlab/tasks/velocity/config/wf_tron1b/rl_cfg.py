"""RL configuration for WF-TRON1B velocity task."""

from dataclasses import dataclass
from typing import Tuple

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


class WFTRON1BRslRlOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
    trial_message: str = ""


@dataclass
class RslRlRepresentationModelCfg(RslRlModelCfg):
    """Config for representation-level teacher-student actor-critic."""

    encoder_hidden_dims: Tuple[int, ...] = (512, 256, 128)
    latent_dim: int = 32
    normalize_latent: bool = True
    class_name: str = "RepresentationActorCritic"


@dataclass
class RslRlVisualRepresentationModelCfg(RslRlRepresentationModelCfg):
    """Config for Visual-CTS representation actor-critic."""

    height_latent_dim: int = 32
    height_scan_start: int | None = None
    height_dim: int = 121
    height_teacher_hidden_dims: Tuple[int, ...] = (512, 256)
    height_proprio_feature_dim: int = 64
    height_depth_feature_dim: int = 64
    height_gru_hidden_dim: int = 128
    height_proprio_hidden_dims: Tuple[int, ...] = (512, 256)
    height_depth_channels: Tuple[int, ...] = (16, 32, 32)
    height_decoder_hidden_dims: Tuple[int, ...] = (256, 512)
    privileged_decoder_hidden_dims: Tuple[int, ...] = (256, 512)
    class_name: str = "VisualRepresentationActorCritic"


@dataclass
class RslRlRepresentationTeacherStudentPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Config for representation-level teacher-student PPO."""

    proprio_encoder_learning_rate: float = 1.0e-3
    num_proprio_encoder_substeps: int = 1
    num_representation_epochs: int = 1
    num_representation_mini_batches: int = 4
    representation_chunk_length: int = 8
    teacher_student_ratio: float | None = None
    class_name: str = "RepresentationTeacherStudentPPO"


def wf_tron1b_ppo_runner_cfg() -> WFTRON1BRslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for WF-TRON1B velocity task."""
    return WFTRON1BRslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 256, 128),
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
        experiment_name="wf_tron1b_velocity",
        save_interval=200,
        num_steps_per_env=24,
        max_iterations=30_000,
        clip_actions=2.0,
        upload_model=False,
    )


def wf_tron1b_rep_ts_runner_cfg() -> WFTRON1BRslRlOnPolicyRunnerCfg:
    """Create representation-level teacher-student runner configuration."""
    return WFTRON1BRslRlOnPolicyRunnerCfg(
        actor=RslRlRepresentationModelCfg(
            hidden_dims=(512, 256, 128),
            encoder_hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            latent_dim=32,
            normalize_latent=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        algorithm=RslRlRepresentationTeacherStudentPpoAlgorithmCfg(
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
            proprio_encoder_learning_rate=1.0e-3,
            num_proprio_encoder_substeps=1,
            num_representation_epochs=1,
            num_representation_mini_batches=4,
            representation_chunk_length=8,
        ),
        obs_groups={
            "teacher_actor": ("actor",),
            "critic": ("critic",),
            "student_history": ("actor_history",),
            "privileged_encoder": ("critic",),
        },
        experiment_name="wf_tron1b_velocity_rep_ts",
        save_interval=200,
        num_steps_per_env=24,
        max_iterations=30_000,
        clip_actions=2.0,
        upload_model=False,
    )


def wf_tron1b_visual_cts_runner_cfg() -> WFTRON1BRslRlOnPolicyRunnerCfg:
    """Create Visual-CTS runner configuration with depth-height estimation."""
    return WFTRON1BRslRlOnPolicyRunnerCfg(
        actor=RslRlVisualRepresentationModelCfg(
            hidden_dims=(512, 256, 128),
            encoder_hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            latent_dim=32,
            normalize_latent=True,
            height_latent_dim=32,
            height_scan_start=None,
            height_dim=121,
            height_teacher_hidden_dims=(512, 256),
            height_proprio_feature_dim=64,
            height_depth_feature_dim=64,
            height_gru_hidden_dim=128,
            height_proprio_hidden_dims=(512, 256),
            height_depth_channels=(16, 32, 32),
            height_decoder_hidden_dims=(256, 512),
            privileged_decoder_hidden_dims=(256, 512),
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        algorithm=RslRlRepresentationTeacherStudentPpoAlgorithmCfg(
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
            proprio_encoder_learning_rate=1.0e-3,
            num_proprio_encoder_substeps=1,
            num_representation_epochs=1,
            num_representation_mini_batches=4,
            representation_chunk_length=8,
            teacher_student_ratio=1.0,
        ),
        obs_groups={
            "teacher_actor": ("actor",),
            "critic": ("critic",),
            "student_history": ("actor_history",),
            "privileged_encoder": ("privileged",),
            "depth_encoder": ("depth_camera",),
            "height_encoder": ("height_scan",),
        },
        experiment_name="wf_tron1b_velocity_visual_cts",
        save_interval=200,
        num_steps_per_env=24,
        max_iterations=30_000,
        clip_actions=2.0,
        upload_model=False,
    )
