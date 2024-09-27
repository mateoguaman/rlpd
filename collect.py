import os
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
import numpy as np
from absl import app, flags
from ml_collections import config_flags
from jaxrl.data import save_replay_buffer
from jaxrl.agents import SACLearner
from jaxrl.data import ReplayBuffer
from jaxrl.wrappers import wrap_gym
import orbax.checkpoint as ocp

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "HalfCheetah-v4", "Environment name.")
flags.DEFINE_string("save_dir", "./collected_data/", "Directory to save collected data.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("num_steps", int(1e6), "Number of steps to collect.")
flags.DEFINE_string("checkpoint_path", "./logs/sac/latest/checkpoints/", "Path to the agent checkpoint.")
flags.DEFINE_integer("checkpoint_step", None, "Step to load. If None, will load latest.")
config_flags.DEFINE_config_file(
    "config",
    "configs/sac_default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

def make_env(env_name, seed):
    env = gym.make(env_name)
    env = wrap_gym(env, rescale_actions=True)
    env = RecordEpisodeStatistics(env)
    env.reset(seed=seed)
    return env

def main(_):
    env = make_env(FLAGS.env_name, FLAGS.seed)
    
    # Create the agent
    kwargs = dict(FLAGS.config)
    agent = SACLearner.create(FLAGS.seed, env.observation_space, env.action_space, **kwargs)
    
    # Load the trained weights
    options = ocp.CheckpointManagerOptions(create=True)
    checkpoint_manager = ocp.CheckpointManager(
        FLAGS.checkpoint_path, options=options)
    agent = checkpoint_manager.restore(FLAGS.checkpoint_step, args=ocp.args.StandardRestore(agent))
    
    # Create a replay buffer to store the collected data
    replay_buffer = ReplayBuffer(env.observation_space, env.action_space, FLAGS.num_steps)
    
    observation, _ = env.reset()
    for step in range(FLAGS.num_steps):
        action, agent = agent.sample_actions(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if not done:
            mask = 1.0
        else:
            mask = 0.0
        
        replay_buffer.insert(
            dict(
                observations=observation,
                actions=action,
                rewards=reward,
                masks=mask,
                dones=done,
                next_observations=next_observation,
            )
        )
        
        if done:
            observation, _ = env.reset()
            print(f"Episode finished. Total reward: {info['episode']['r']}")
        else:
            observation = next_observation
        
        if (step + 1) % 10000 == 0:
            print(f"Collected {step + 1} steps")
    
    # Save the collected data
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    
    save_path = os.path.join(FLAGS.save_dir, f"{FLAGS.env_name}_{FLAGS.num_steps}_steps.npz")
    save_replay_buffer(replay_buffer, save_path, env.observation_space, env.action_space)

    print(f"Collected data saved to {save_path}")

if __name__ == "__main__":
    app.run(main)

'''
Run with: 

MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false python collect.py --env_name=HalfCheetah-v4  --save_dir ./expert_data --checkpoint_path=/home/mateo/projects/jaxrl/logs/sac/2024-09-17_13-26-45/checkpoints --checkpoint_step=1000000 --config=configs/rlpd_config.py
'''