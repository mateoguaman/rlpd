from typing import Dict

import gymnasium as gym
import numpy as np

from rlpd.wrappers.wandb_video import WANDBVideo
from rlpd.data.dataset import Dataset


def evaluate(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False
) -> Dict[str, float]:
    if save_video:
        env = WANDBVideo(env, name="eval_video", max_videos=1)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)

    for i in range(num_episodes):
        observation, _ = env.reset()
        done = False
        while not done:
            action = agent.eval_actions(observation)
            observation, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
    return {"return": np.mean(env.return_queue), "length": np.mean(env.length_queue)}

def evaluate_log_prob(agent, dataset: Dataset, batch_size: int = 2048) -> float:
    num_iters = len(dataset) // batch_size
    total_log_prob = 0.0
    for j in range(num_iters):
        indx = np.arange(j * batch_size, (j + 1) * batch_size)
        batch = dataset.sample(batch_size, keys=("observations", "actions"), indx=indx)
        log_prob = agent.eval_log_probs(batch)
        total_log_prob += log_prob

    return total_log_prob / num_iters