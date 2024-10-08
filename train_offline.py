#! /usr/bin/env python
import gymnasium as gym
import jax
import tqdm
import wandb
import datetime
import os
# from absl import app, flags
# from ml_collections import config_flags
# import ml_collections
# from ml_collections.config_dict import config_dict
import orbax.checkpoint as ocp
import hydra
from hydra.core.config_store import ConfigStore

from jaxrl.agents import BCLearner, IQLLearner
from jaxrl.data import load_replay_buffer
from jaxrl.evaluation import evaluate
from jaxrl.wrappers import wrap_gym, set_universal_seed

from configs import get_flat_config, to_dict
# from configs import SACRunnerConfig, REDQRunnerConfig, DroQRunnerConfig
# from configs import TD3RunnerConfig
from configs import BCRunnerConfig, IQLRunnerConfig

cs = ConfigStore.instance()
cs.store(name="bc", node=BCRunnerConfig)
cs.store(name="iql", node=IQLRunnerConfig)
# FLAGS = flags.FLAGS

# flags.DEFINE_string("env_name", "HalfCheetah-v4", "Environment name.")
# flags.DEFINE_string("algorithm", "bc", "Algorithm to use for offline learning. Options: bc, iql")
# flags.DEFINE_string("dataset_path", "buffer.npz", "Path to the npz buffer which contains offline data.")
# flags.DEFINE_integer("seed", 42, "Random seed.")
# flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
# flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
# flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
# flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
# flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
# flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
# flags.DEFINE_float("filter_percentile", None, "Take top N% trajectories.")
# flags.DEFINE_float(
#     "filter_threshold", None, "Take trajectories with returns above the threshold."
# )
# # config_flags.DEFINE_config_file(
# #     "config",
# #     "configs/offline_config.py:bc",
# #     "File path to the training hyperparameter configuration.",
# #     lock_config=False,
# # )


# def get_bc_config():
#     config = ml_collections.ConfigDict()

#     config.actor_lr = 1e-3
#     config.hidden_dims = (256, 256)
#     # config.cosine_decay = True
#     # config.dropout_rate = 0.1/
#     # config.weight_decay = config_dict.placeholder(float)
#     config.apply_tanh = True

#     config.distr = "unitstd_normal"
#     # unitstd_normal | tanh_normal | normal

#     return config

# def get_iql_config():
#     config = ml_collections.ConfigDict()

#     config.actor_lr = 3e-4
#     config.value_lr = 3e-4
#     config.critic_lr = 3e-4

#     config.hidden_dims = (256, 256)

#     config.discount = 0.99

#     config.expectile = 0.7  # The actual tau for expectiles.
#     # config.A_scaling = 3.0
#     config.temperature = 3.0
#     # config.dropout_rate = config_dict.placeholder(float)
#     # config.cosine_decay = True

#     config.tau = 0.005  # For soft target updates.

#     # config.critic_reduction = "min"
    
#     config.use_tanh_normal = False
#     state_dependent_std = True

#     return config

@hydra.main(version_base=None, config_path="configs", config_name="bc")
def main(cfg: BCRunnerConfig):
    wandb.init(project="jaxrl2_offline")
    wandb.config.update(to_dict(cfg))

    exp_prefix = cfg.experiment_name 
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    log_dir = os.path.abspath(os.path.join(cfg.save_dir, exp_prefix, date_str))

    if cfg.checkpoint_model:
        chkpt_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(chkpt_dir, exist_ok=True)

        ## Set up Orbax checkpointer manager
        # import absl.logging
        # absl.logging.set_verbosity(absl.logging.INFO)  ## this can make it less verbose
        options = ocp.CheckpointManagerOptions(create=True)
        checkpoint_manager = ocp.CheckpointManager(
            chkpt_dir, options=options)

    env = gym.make(cfg.env_name)
    env = wrap_gym(env, rescale_actions=True)
    set_universal_seed(env, cfg.seed)

    # dataset = D4RLDataset(env)
    # if FLAGS.filter_percentile is not None or FLAGS.filter_threshold is not None:
    #     dataset.filter(
    #         percentile=FLAGS.filter_percentile, threshold=FLAGS.filter_threshold
    #     )
    # dataset.seed(FLAGS.seed)

    replay_buffer, obs_space, action_space = load_replay_buffer(cfg.dataset_path)
    if cfg.clip_to_eps:
        lim = 1 - cfg.eps
        replay_buffer.dataset_dict["actions"].clip(-lim, lim)

    kwargs = get_flat_config(cfg.algorithm, use_prefix=False)
    class_name = kwargs.pop('class_name')

    if class_name == "bc":
        agent = BCLearner.create(cfg.seed, env.observation_space, env.action_space, **kwargs)
    elif class_name == "iql":
        agent = IQLLearner.create(cfg.seed, env.observation_space, env.action_space, **kwargs)
        
    
    # agent = globals()[FLAGS.config.model_constructor](
    #     FLAGS.seed, env.observation_space.sample(), env.action_space.sample(), **kwargs
    # )

    for i in tqdm.tqdm(
        range(1, cfg.max_steps + 1), smoothing=0.1, disable=not cfg.tqdm
    ):
        batch = replay_buffer.sample(cfg.batch_size)
        agent, info = agent.update(batch)

        if i % cfg.log_interval == 0:
            info = jax.device_get(info)
            wandb.log(info, step=i)

        if i % cfg.eval_interval == 0:
            eval_info = evaluate(agent, env, num_episodes=cfg.eval_episodes)
            # eval_info["return"] = env.get_normalized_score(eval_info["return"]) * 100.0
            for k, v in eval_info.items():
                wandb.log({f"evaluation/{k}": v}, step=i)

        if i % cfg.save_interval == 0 and i > 0:
            if cfg.checkpoint_model:
                try:
                    checkpoint_manager.save(step=i, args=ocp.args.StandardSave(agent))
                except:
                    print("Could not save model checkpoint.")


if __name__ == "__main__":
    main()


'''
Run with

MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false python train_offline.py --env_name=HalfCheetah-v4  --algorithm=iql --dataset_path=/home/mateo/projects/jaxrl/expert_data/HalfCheetah-v4_1000000_steps.npz

'''