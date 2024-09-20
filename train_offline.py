#! /usr/bin/env python
import gymnasium as gym
import jax
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags
import ml_collections
from ml_collections.config_dict import config_dict

from rlpd.agents import BCLearner, IQLLearner
from rlpd.data import load_replay_buffer
from rlpd.evaluation import evaluate
from rlpd.wrappers import wrap_gym, set_universal_seed

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "HalfCheetah-v4", "Environment name.")
flags.DEFINE_string("algorithm", "bc", "Algorithm to use for offline learning. Options: bc, iql")
flags.DEFINE_string("buffer_path", "buffer.npz", "Path to the npz buffer which contains offline data.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_float("filter_percentile", None, "Take top N% trajectories.")
flags.DEFINE_float(
    "filter_threshold", None, "Take trajectories with returns above the threshold."
)
# config_flags.DEFINE_config_file(
#     "config",
#     "configs/offline_config.py:bc",
#     "File path to the training hyperparameter configuration.",
#     lock_config=False,
# )


def get_bc_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 1e-3
    config.hidden_dims = (256, 256)
    # config.cosine_decay = True
    # config.dropout_rate = 0.1/
    # config.weight_decay = config_dict.placeholder(float)
    config.apply_tanh = True

    config.distr = "unitstd_normal"
    # unitstd_normal | tanh_normal | normal

    return config

def get_iql_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4
    config.value_lr = 3e-4
    config.critic_lr = 3e-4

    config.hidden_dims = (256, 256)

    config.discount = 0.99

    config.expectile = 0.7  # The actual tau for expectiles.
    # config.A_scaling = 3.0
    config.temperature = 3.0
    # config.dropout_rate = config_dict.placeholder(float)
    # config.cosine_decay = True

    config.tau = 0.005  # For soft target updates.

    # config.critic_reduction = "min"
    
    config.use_tanh_normal = False
    state_dependent_std = True

    return config


def main(_):
    wandb.init(project="jaxrl2_offline")
    wandb.config.update(FLAGS)

    env = gym.make(FLAGS.env_name)
    env = wrap_gym(env, rescale_actions=True)
    set_universal_seed(env, FLAGS.seed)

    # dataset = D4RLDataset(env)
    # if FLAGS.filter_percentile is not None or FLAGS.filter_threshold is not None:
    #     dataset.filter(
    #         percentile=FLAGS.filter_percentile, threshold=FLAGS.filter_threshold
    #     )
    # dataset.seed(FLAGS.seed)

    replay_buffer, obs_space, action_space = load_replay_buffer(FLAGS.buffer_path)

    # if "antmaze" in FLAGS.env_name:
    #     dataset.dataset_dict["rewards"] *= 100
    # elif FLAGS.env_name.split("-")[0] in ["hopper", "halfcheetah", "walker2d"]:
    #     dataset.normalize_returns(scaling=1000)

    # kwargs = dict(FLAGS.config.model_config)
    # if kwargs.pop("cosine_decay", False):
    #     kwargs["decay_steps"] = FLAGS.max_steps

    if FLAGS.algorithm == "bc":
        kwargs = dict(get_bc_config())
        agent = BCLearner.create(FLAGS.seed, env.observation_space, env.action_space, **kwargs)
    elif FLAGS.algorithm == "iql":
        kwargs = dict(get_iql_config())
        agent = IQLLearner.create(FLAGS.seed, env.observation_space, env.action_space, **kwargs)
        
    
    # agent = globals()[FLAGS.config.model_constructor](
    #     FLAGS.seed, env.observation_space.sample(), env.action_space.sample(), **kwargs
    # )

    for i in tqdm.tqdm(
        range(1, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        batch = replay_buffer.sample(FLAGS.batch_size)
        agent, info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            info = jax.device_get(info)
            wandb.log(info, step=i)

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(agent, env, num_episodes=FLAGS.eval_episodes)
            # eval_info["return"] = env.get_normalized_score(eval_info["return"]) * 100.0
            for k, v in eval_info.items():
                wandb.log({f"evaluation/{k}": v}, step=i)


if __name__ == "__main__":
    app.run(main)


'''
Run with

MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false python train_offline.py --env_name=HalfCheetah-v4  --algorithm=iql --buffer_path=/home/mateo/projects/rlpd/expert_data/HalfCheetah-v4_1000000_steps.npz

'''