#! /usr/bin/env python
import gymnasium as gym
from gymnasium.envs.registration import registry
import tqdm
import wandb
import os
import datetime
from typing import Any, Optional, Dict

# from absl import app, flags
# from ml_collections import config_flags

import orbax.checkpoint as ocp
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, DictConfig

from jaxrl.agents import SACLearner, TD3Learner
from jaxrl.data import ReplayBuffer
from jaxrl.evaluation import evaluate
from jaxrl.wrappers import wrap_gym, set_universal_seed
from jaxrl.data import load_replay_buffer, save_replay_buffer

from configs import get_flat_config, to_dict
from configs import SACRunnerConfig, REDQRunnerConfig, DroQRunnerConfig
from configs import TD3RunnerConfig

cs = ConfigStore.instance()
cs.store(name="sac", node=SACRunnerConfig)
cs.store(name="redq", node=REDQRunnerConfig)
cs.store(name="droq", node=DroQRunnerConfig)
cs.store(name="td3", node=TD3RunnerConfig)

def check_env_id(env_id):
    dm_control_env_ids = [
        id
        for id in registry
        if id.startswith("dm_control/") and id != "dm_control/compatibility-env-v0"
    ]
    if not env_id.startswith("dm_control/"):
        for id in dm_control_env_ids:
            if env_id in id:
                env_id = "dm_control/" + env_id
    if env_id not in registry:
        raise ValueError("Provide valid env id.")
    return env_id

@hydra.main(version_base=None, config_path="configs", config_name="sac")
def main(cfg: SACRunnerConfig):
    wandb.init(project="jaxrl2_online")
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

    if cfg.checkpoint_buffer:
        buffer_dir = os.path.join(log_dir, "buffers")
        os.makedirs(buffer_dir, exist_ok=True)

    def make_and_wrap_env(env_id):
        env = gym.make(check_env_id(env_id))
        return wrap_gym(env, rescale_actions=True)
    
    env = make_and_wrap_env(cfg.env_name)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    set_universal_seed(env, cfg.seed)


    eval_env = make_and_wrap_env(cfg.env_name)
    set_universal_seed(eval_env, cfg.seed + 42)

    kwargs = get_flat_config(cfg.algorithm, use_prefix=False)
    class_name = kwargs.pop('class_name')
    if class_name == "sac" or class_name == "redq" or class_name == "droq":
        agent = SACLearner.create(cfg.seed, env.observation_space, env.action_space, **kwargs)
    elif class_name == "td3":
        agent = TD3Learner.create(cfg.seed, env.observation_space, env.action_space, **kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {cfg.algorithm.class_name}")

    replay_buffer = ReplayBuffer(
        env.observation_space, env.action_space, cfg.max_steps
    )
    replay_buffer.seed(cfg.seed)

    observation, _ = env.reset()
    done = False
    for i in tqdm.tqdm(
        range(1, cfg.max_steps + 1), smoothing=0.1, disable=not cfg.tqdm
    ):
        if i < cfg.start_training:
            action = env.action_space.sample()
        else:
            action, agent = agent.sample_actions(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if not terminated:
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
        observation = next_observation

        if done:
            observation, _ = env.reset()
            done = False
            for k, v in info["episode"].items():
                decode = {"r": "return", "l": "length", "t": "time"}
                wandb.log({f"training/{decode[k]}": v}, step=i)

        if i >= cfg.start_training:
            batch = replay_buffer.sample(cfg.batch_size)
            agent, update_info = agent.update(batch, utd_ratio=1)


        if i % cfg.eval_interval == 0:
            eval_info = evaluate(agent, eval_env, num_episodes=cfg.eval_episodes, save_video=cfg.save_video)
            for k, v in eval_info.items():
                wandb.log({f"evaluation/{k}": v}, step=i)

        if i % cfg.save_interval == 0 and i > 0:
            if cfg.checkpoint_model:
                try:
                    checkpoint_manager.save(step=i, args=ocp.args.StandardSave(agent))
                except:
                    print("Could not save model checkpoint.")

            if cfg.checkpoint_buffer:
                try:
                    save_replay_buffer(replay_buffer, os.path.join(buffer_dir, f"buffer_{i}"), env.observation_space, env.action_space)
                except:
                    print("Could not save agent buffer.")
    checkpoint_manager.wait_until_finished()

if __name__ == "__main__":
    main()
'''
Run with

MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false python train_online.py --config-name=droq

'''