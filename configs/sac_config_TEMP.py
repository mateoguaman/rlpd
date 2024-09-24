from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any, Optional
from configs.utils import flatten_config_dataclass

# ### ======================= Train Configs =============================

# @dataclass
# class PolicyConfig:
#     _target_: str = "rsl_rl.modules.ActorCritic"
#     init_noise_std: float = 0.25
#     fixed_std: bool = False
#     ortho_init: bool = True
#     last_actor_layer_scaling: float = 0.01
#     actor_hidden_dims: Tuple[int, ...] = (512, 256, 128)
#     critic_hidden_dims: Tuple[int, ...] = (512, 256, 128)
#     activation: str = "elu" # elu, relu, selu, leaky_relu, tanh, sigmoid

# @dataclass
# class AlgorithmConfig:
#     _target_: str = "rsl_rl.algorithms.PPO"
#     value_loss_coef: float = 1.
#     use_clipped_value_loss: bool = False
#     clip_param: float = 0.2
#     entropy_coef: float = 0.01
#     num_learning_epochs: int = 5
#     num_mini_batches: int = 4 # minibatch size = num_envs*nsteps / nminibatches
#     learning_rate: float = 1e-3
#     max_learning_rate: float = 1e-2
#     min_learning_rate: float = 0.
#     schedule: str = 'adaptive' # adaptive, fixed
#     gamma: float = 0.99
#     lam: float = 0.95
#     desired_kl: float = 0.01
#     max_grad_norm: float = 1.

# @dataclass
# class RunnerConfig:
#     num_steps_per_env: int = 24 # per iteration
#     iterations: int = "${oc.select: iterations,1500}" # number of policy updates

#     # logging
#     save_interval: int = 50 # check for potential saves every this many iterations
#     episode_buffer_len: int = 100 # window of previous episodes to consider for average rewards/lengths

#     # load and resume
#     resume_root: str = ""
#     checkpoint: int = -1 # -1 = last saved model

#     # Logging with Weights and Biases
#     use_wandb: bool = True
#     log_videos: bool = True
#     video_frequency: Optional[int] = 1 ## How often to save videos, every <frequency> episodes


## Config for family of SAC algorithms. Includes SAC, REDQ, DroQ, RLPD-SAC, RLPD-DroQ, RLPD-REDQ

'''
Mateo developer notes:

There are three configs:
- Policy: includes details about NNs such as initialization, dropout, size, activations, etc.
- Algorithm: includes details about training such as learning rate, batch size, entropy initializations, etc.
- Runner: includes details about seed, device, max_iterations, saving interval, experiment name, run_name, logger info, resume, load_checkpoint, etc.
IMPORTANT: The runner contains fields for algorithm, and the algorithm contains the policy cfg. This is so that we can for now just flatten the algorithm
config (including the policy config) and pass it onto the create method for each Learner.

One issue we face is that the create method just takes a **kwargs, with a flat structure for all the config stuff. We can either flatten the dataclass config and then pass it
into the create methods so that it is compatible with other jaxrl implementations, or we can pass a config class from which everything is loaded. I kinda prefer the second 
option so I'm gonna do that for now.  
'''

@dataclass
class SACPolicyConfig:
    hidden_dims: Tuple[int] = (128, 128, 128)
    num_qs: int = 2
    num_min_qs: Optional[int] = None  ## Used for REDQ/DroQ
    critic_layer_norm: bool = False  ## Used in RLPD

    critic_weight_decay: Optional[float] = None
    # actor_weight_decay: Optional[float] = None #0.1 I think I added this when doing APRL-like experiments
    critic_dropout_rate: Optional[float] = None  ## Used in DroQ
    use_pnorm: bool = False  ## Unsure why it's here, seems to always be False. Leaving it here for completeness
    use_critic_resnet: bool = False
    # gradient_clipping_norm: Optional[float] = None  # I added this for my own experiments, not currently implemented in this repo
    # use_tanh_normal: bool = True
    # state_dependent_std: bool = True

    def get_flat_config(self, use_prefix: bool = True) -> Dict[str, Any]:
        return flatten_config_dataclass(self, '' if use_prefix else None)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class SACAlgorithmConfig:
    class_name: str = "sac"
    actor_lr: float = 3e-4  
    critic_lr: float = 3e-4
    temp_lr: float = 3e-4
    discount: float = 0.99
    tau: float = 0.005
    init_temperature: float = 1.0
    target_entropy: Optional[float] = None
    backup_entropy: bool = True

    policy: SACPolicyConfig = SACPolicyConfig()

    def get_flat_config(self, use_prefix: bool = True) -> Dict[str, Any]:
        return flatten_config_dataclass(self, '' if use_prefix else None)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SACRunnerConfig:
    algorithm: SACAlgorithmConfig = SACAlgorithmConfig()

    seed: int = 42
    env_name: str = "HalfCheetah-v4"
    experiment_name: str = "sac"  # Used for wandb and project folder logging
    run_name: Optional[str] = None  # If not None, used to save logs in dir date_run_name
    save_dir: str = "checkpoints"
    log_interval: int = 1000
    save_interval: int = 100000
    eval_episodes: int = 10  # Not currently used
    eval_interval: int = 5000  # Not currently used
    checkpoint_model: bool = True
    checkpoint_buffer: bool = True
    save_video: bool = False
    wandb: bool = True
    video_interval: int = 20
    tqdm: bool = True
    episode_buffer_len: int = 100  # Window of previous episodes to consider for average rewards/lengths

    max_steps: int = int(1e6)
    start_training: int = 1000
    # offline_ratio: float = 0.5  # This is used in hybrid rl only
    batch_size: int = 256
    utd_ratio: int = 1
    reset_param_interval: Optional[int] = None #int(1e4)  # Not currently integrated

    def get_flat_config(self, use_prefix: bool = True) -> Dict[str, Any]:
        return flatten_config_dataclass(self, '' if use_prefix else None)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class REDQRunnerConfig(SACRunnerConfig):
    algorithm: SACAlgorithmConfig = SACAlgorithmConfig(
        class_name="redq",
        policy=SACPolicyConfig(
            num_qs=10,  ## This is what makes it REDQ
            num_min_qs=2,  ## This is what makes it REDQ
            critic_layer_norm=True,  ## REDQ paper doesn't have this, but leaving this on for now.
        )
    )

@dataclass
class DroQRunnerConfig(SACRunnerConfig):
    algorithm: SACAlgorithmConfig = SACAlgorithmConfig(
        class_name="droq",
        init_temperature=0.1,
        policy=SACPolicyConfig(
            critic_dropout_rate=0.01,  ## This is what makes it DroQ
            critic_layer_norm=True,  ## Also part of DroQ paper
        )
    )












# @dataclass
# class RLPDAlgorithmConfig:
#     ## Present in td_config
#     actor_lr: float = 3e-4  
#     critic_lr: float = 3e-4
#     hidden_dims: Tuple[int] = (256, 256)
#     discount: float = 0.99
#     tau: float = 0.005
#     num_qs: int = 2
#     critic_layer_norm: bool = False

#     ## Present in SAC
#     temp_lr: float = 3e-4
#     init_temperature: float = 1.0
#     target_entropy: Optional[float] = None
#     backup_entropy: bool = True
#     critic_weight_decay: Optional[float] = None
#     actor_weight_decay: Optional[float] = 0.1

#     ## REDQ/DroQ/RLPD config
#     num_min_qs: Optional[int] = None
#     critic_dropout_rate: Optional[float] = None
#     use_pnorm: bool = False
#     use_critic_resnet: bool = False   

#     ## Gradient clipping
#     gradient_clipping_norm: Optional[float] = None

#     use_tanh_normal: bool = True
#     state_dependent_std: bool = True

#     ## APRL-like loss for actions outside range
#     exterior_linear_c: float = 1.0

# # @dataclass
# # class RLPDRunnerConfig:
# #     a: Optional[bool] = None

# @dataclass
# class IQLAlgorithmConfig:
#     actor_lr: float = 3e-4
#     value_lr: float = 3e-4
#     critic_lr: float = 3e-4
#     hidden_dims: Tuple[int] = (256, 256)
#     discount: float = 0.99
#     tau: float = 0.005
#     expectile: float = 0.8
#     temperature: float = 0.1 # Same as A_scaling
#     actor_weight_decay: Optional[float] = None
#     critic_weight_decay: Optional[float] = None
#     actor_weight_decay: Optional[float] = None
#     value_weight_decay: Optional[float] = None
#     critic_layer_norm: Optional[bool] = True#False
#     value_layer_norm: Optional[bool] = True#False
#     num_qs: int = 2
#     num_min_qs: Optional[int]= None
#     num_vs: int = 1
#     num_min_vs: Optional[int] = None
#     use_tanh_normal: bool = False
#     state_dependent_std: bool = False

# @dataclass
# class IQLPretrainConfig:
#     algorithm: IQLAlgorithmConfig = IQLAlgorithmConfig()
#     name: Optional[str] = None
#     project_name: str = "iql_pretraining"
#     pretrain_steps: int = int(1e5)
#     seed: int = 42
#     log_interval: int = 100
#     batch_size: int = 256
#     tqdm: bool = True

#     def as_dict(self):
#         return asdict(self)
    
# @dataclass
# class IQLFinetuneConfig:
#     algorithm: IQLAlgorithmConfig = IQLAlgorithmConfig()
#     name: Optional[str] = None
#     project_name: str = "iql_finetuning"
#     max_steps: int = int(1e5)
#     seed: int = 42
#     offline_data_dir: Optional[str] = None 
#     episode_buffer_len: int = 100
#     pretrain_steps: int = 0
#     tqdm: bool = True
#     batch_size: int = 256
#     utd_ratio: int = 1
#     log_interval: int = 100
#     start_training: int = 0
#     offline_ratio: float = 0.  ## Unsure what this should be for IQL finetuning
#     use_offline_ratio_schedule: bool = False
#     video_interval: int = 20
#     eval_episodes: int = 1
#     eval_interval: int = 1000
#     wandb: bool = True

#     def as_dict(self):
#         return asdict(self)


# @dataclass
# class RLPDSACTrainConfig:
#     algorithm: RLPDAlgorithmConfig = RLPDAlgorithmConfig(
#         critic_layer_norm=True
#     )
#     name: Optional[str] = None
#     project_name: str = "rlpd_locomotion_sac"
#     max_steps: int = int(1e5)
#     seed: int = 42
#     offline_data_dir: Optional[str] = None 
#     episode_buffer_len: int = 100
#     pretrain_steps: int = 0
#     tqdm: bool = True
#     batch_size: int = 256
#     utd_ratio: int = 1
#     log_interval: int = 1000
#     start_training: int = 1000
#     offline_ratio: float = 0.5
#     use_offline_ratio_schedule: bool = False
#     use_offline_data: bool = True
#     video_interval: int = 20
#     eval_episodes: int = 1
#     eval_interval: int = 1000
#     reset_param_interval: Optional[int] = None#int(1e4)
#     action_soft_range: Optional[float] = None#0.5  ## Between [0,1]. If |action| > action_soft_range*joint_limit, penalizes action 
#     wandb: bool = True
#     checkpoint_dir: str = "checkpoints"
#     save_interval: int = 10000

#     def as_dict(self):
#         return asdict(self)
    
# @dataclass
# class SACTrainConfig(RLPDSACTrainConfig):
#     project_name: str = "locomotion_sac"
#     offline_ratio: float = 0.
#     use_offline_ratio_schedule: bool = False
    
# @dataclass
# class RLPDDroQTrainConfig:
#     algorithm: RLPDAlgorithmConfig = RLPDAlgorithmConfig(
#         critic_dropout_rate=0.01,
#         critic_layer_norm=True,
#         init_temperature=0.1,  ## Try 1.0
#     )
#     name: Optional[str] = None
#     project_name: str = "rlpd_locomotion_droq"
#     max_steps: int = int(1e5)
#     seed: int = 42
#     offline_data_dir: Optional[str] = None 
#     episode_buffer_len: int = 100
#     pretrain_steps: int = 0
#     tqdm: bool = True
#     batch_size: int = 256
#     utd_ratio: int = 20
#     log_interval: int = 1000
#     start_training: int = 1000
#     offline_ratio: float = 0.5
#     use_offline_ratio_schedule: bool = False
#     use_offline_data: bool = True
#     video_interval: int = 20
#     eval_episodes: int = 1
#     eval_interval: int = 1000
#     reset_param_interval: Optional[int] = None#int(1e4)
#     action_soft_range: Optional[float] = None#0.3  ## Between [0,1]. If |action| > action_soft_range*joint_limit, penalizes action 
#     wandb: bool = True
#     checkpoint_dir: str = "checkpoints"
#     save_interval: int = 10000

#     def as_dict(self):
#         return asdict(self)

# @dataclass
# class DroQTrainConfig(RLPDDroQTrainConfig):
#     project_name: str = "locomotion_droq"
#     offline_ratio: float = 0.
#     use_offline_ratio_schedule: bool = False

# @dataclass
# class RLPDREDQTrainConfig:
#     algorithm: RLPDAlgorithmConfig = RLPDAlgorithmConfig(
#         num_qs=10,
#         num_min_qs=2,
#         critic_layer_norm=True,
#         init_temperature=1.0,
#         ## TODO: Remove after redq/rlpd-redq finetuning tests
#         actor_lr=1e-4,
#         critic_lr=1e-4,
#         temp_lr=1e-4
#     )
#     name: Optional[str] = None
#     project_name: str = "rlpd_locomotion_redq"
#     max_steps: int = int(1e5)
#     seed: int = 42
#     offline_data_dir: Optional[str] = None 
#     episode_buffer_len: int = 100
#     pretrain_steps: int = 0
#     tqdm: bool = True
#     batch_size: int = 256
#     utd_ratio: int = 20
#     log_interval: int = 1000
#     start_training: int = 1000
#     offline_ratio: float = 0.5
#     use_offline_ratio_schedule: bool = False
#     use_offline_data: bool = True
#     video_interval: int = 20
#     eval_episodes: int = 1
#     eval_interval: int = 1000
#     reset_param_interval: Optional[int] = None#int(1e4)
#     action_soft_range: Optional[float] = None#0.5  ## Between [0,1]. If |action| > action_soft_range*joint_limit, penalizes action 
#     wandb: bool = True
#     checkpoint_dir: str = "checkpoints"
#     save_interval: int = 10000

#     def as_dict(self):
#         return asdict(self)

# @dataclass
# class REDQTrainConfig(RLPDREDQTrainConfig):
#     project_name: str = "locomotion_redq"
#     offline_ratio: float = 0.
#     use_offline_ratio_schedule: bool = False

# @dataclass
# class DistillConfig:
#     lr: float = 3e-4
#     num_epochs: int = 2
#     batch_size: int = 512
#     seed: int = 42

#     project_name: str = "distill"
#     name: str = ""
#     log_interval: int = 10


# Example usage:
if __name__ == "__main__":
    config = REDQRunnerConfig()
    print("Flat config with prefixes:")
    print(config.get_flat_config(use_prefix=True))
    print("\nFlat config without prefixes:")
    print(config.get_flat_config(use_prefix=False))
    print("\nConfig as dict:")
    print(config.to_dict())
    import ipdb;ipdb.set_trace()
    print("Done debugging")