from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any, Optional
from configs.utils import flatten_config_dataclass

@dataclass
class BCPolicyConfig:
    hidden_dims: Tuple[int] = (128, 128, 128)
    dropout_rate: Optional[float] = None

    def get_flat_config(self, use_prefix: bool = True) -> Dict[str, Any]:
        return flatten_config_dataclass(self, '' if use_prefix else None)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class BCAlgorithmConfig:
    class_name: str = "bc"
    actor_lr: float = 3e-4
    warmup_steps: int = 1000 
    decay_steps: int = 1000000 
    weight_decay: Optional[float] = None 
    distr: str = "normal"  # Choose between ["tanh_normal", "unitstd_normal", "normal"]

    policy: BCPolicyConfig = BCPolicyConfig()

    def get_flat_config(self, use_prefix: bool = True) -> Dict[str, Any]:
        return flatten_config_dataclass(self, '' if use_prefix else None)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BCRunnerConfig:
    algorithm: BCAlgorithmConfig = BCAlgorithmConfig()

    seed: int = 42  ## Good
    env_name: str = "HalfCheetah-v4"  ## Good
    dataset_path: str = "buffer.npz"  ## Good
    experiment_name: str = "bc"  # Used for wandb and project folder logging
    run_name: Optional[str] = None  # If not None, used to save logs in dir date_run_name
    save_dir: str = "checkpoints"
    log_interval: int = 1000  ## Good
    save_interval: int = 100000
    eval_episodes: int = 10  # Not currently used  ## Good
    eval_interval: int = 5000  # Not currently used  ## Good
    checkpoint_model: bool = True
    save_video: bool = False
    wandb: bool = True
    video_interval: int = 20
    tqdm: bool = True  ## Good
    episode_buffer_len: int = 100  # Window of previous episodes to consider for average rewards/lengths
    filter_percentile: Optional[float] = None  ## Good
    filter_threshold: Optional[float] = None  ## Good
    clip_to_eps: bool = True  ## Replace [-1, 1] with [-1-eps, 1+eps] in dataset
    eps: float = 1e-5

    max_steps: int = int(1e6)  ## Good
    start_training: int = 10000
    batch_size: int = 256  ## Good

    def get_flat_config(self, use_prefix: bool = True) -> Dict[str, Any]:
        return flatten_config_dataclass(self, '' if use_prefix else None)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)