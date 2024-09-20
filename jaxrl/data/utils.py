import numpy as np
from jaxrl.data.dataset import _subselect
from jaxrl.data.replay_buffer import ReplayBuffer
import gymnasium as gym
from typing import Tuple, Optional, Union

def save_replay_buffer(replay_buffer: ReplayBuffer, save_path: str, env_observation_space: gym.spaces.Space, env_action_space: gym.spaces.Space):
    """
    Save a ReplayBuffer to a .npz file, including environment spaces.
    
    Args:
        replay_buffer (ReplayBuffer): The replay buffer to save.
        save_path (str): The path where to save the .npz file.
        env_observation_space (gym.spaces.Space): The observation space of the environment.
        env_action_space (gym.spaces.Space): The action space of the environment.
    """
    np.savez(
        save_path,
        observations=replay_buffer.dataset_dict['observations'],
        actions=replay_buffer.dataset_dict['actions'],
        rewards=replay_buffer.dataset_dict['rewards'],
        masks=replay_buffer.dataset_dict['masks'],
        dones=replay_buffer.dataset_dict['dones'],
        next_observations=replay_buffer.dataset_dict['next_observations'],
        # observation_labels=replay_buffer.dataset_dict['observation_labels'],
        size=replay_buffer._size,
        capacity=replay_buffer._capacity,
        insert_index=replay_buffer._insert_index,
        env_observation_space=env_observation_space,
        env_action_space=env_action_space
    )
    print(f"Saved replay buffer to {save_path}")


def load_replay_buffer(load_path: str) -> Tuple[ReplayBuffer, gym.spaces.Space, gym.spaces.Space]:
    """
    Load a ReplayBuffer from a .npz file.
    
    Args:
        load_path (str): The path to the .npz file to load.
    
    Returns:
        Tuple[ReplayBuffer, gym.spaces.Space, gym.spaces.Space]: The loaded replay buffer, observation space, and action space.
    """
    loaded_data = np.load(load_path, allow_pickle=True)
    
    observation_space = loaded_data['env_observation_space'].item()
    action_space = loaded_data['env_action_space'].item()
    
    replay_buffer = ReplayBuffer(
        observation_space,
        action_space,
        capacity=int(loaded_data['capacity']),
        next_observation_space=observation_space,
        # observation_labels=loaded_data['observation_labels'].item()
    )
    
    replay_buffer.dataset_dict['observations'] = loaded_data['observations']
    replay_buffer.dataset_dict['actions'] = loaded_data['actions']
    replay_buffer.dataset_dict['rewards'] = loaded_data['rewards']
    replay_buffer.dataset_dict['masks'] = loaded_data['masks']
    replay_buffer.dataset_dict['dones'] = loaded_data['dones']
    replay_buffer.dataset_dict['next_observations'] = loaded_data['next_observations']
    replay_buffer._size = int(loaded_data['size'])
    replay_buffer._capacity = int(loaded_data['capacity'])
    replay_buffer._insert_index = int(loaded_data['insert_index'])
    
    print(f"Loaded replay buffer from {load_path}")
    return replay_buffer, observation_space, action_space

def get_size(data):
    if isinstance(data, np.ndarray):
        return data.shape[0]
    elif isinstance(data, dict):
        return get_size(next(iter(data.values())))
    else:
        raise TypeError(f"Unexpected data type: {type(data)}")

def combine(one_dict, other_dict, seed=None):
    combined = {}
    rng = np.random.default_rng(seed)

    # Get the total size of the combined data
    total_size = get_size(one_dict) + get_size(other_dict)
    
    # Generate a shuffled index array
    shuffled_indices = rng.permutation(total_size)

    for k, v in one_dict.items():
        if k == "observation_labels":
            combined[k] = v
            continue
        if isinstance(v, dict):
            combined[k] = combine(v, other_dict[k], seed=rng)
        else:
            tmp = np.empty((total_size, *v.shape[1:]), dtype=v.dtype)
            tmp[:v.shape[0]] = v
            tmp[v.shape[0]:] = other_dict[k]
            
            # Use the shuffled indices to reorder the data
            combined[k] = tmp[shuffled_indices]

    return combined


def combine_replay_buffers(buffer1: ReplayBuffer, buffer2: ReplayBuffer, seed: Optional[int] = None) -> ReplayBuffer:
    """
    Combine two replay buffers into a new one.

    Args:
        buffer1 (ReplayBuffer): The first replay buffer.
        buffer2 (ReplayBuffer): The second replay buffer.

    Returns:
        ReplayBuffer: A new replay buffer containing data from both input buffers.
    """
    # Ensure the buffers are compatible
    assert buffer1.dataset_dict.keys() == buffer2.dataset_dict.keys(), "Replay buffers have different keys"
    assert buffer1.dataset_dict['observations'].shape[1:] == buffer2.dataset_dict['observations'].shape[1:], "Observation spaces don't match"
    assert buffer1.dataset_dict['actions'].shape[1:] == buffer2.dataset_dict['actions'].shape[1:], "Action spaces don't match"

    # Calculate the total size of the new buffer
    total_size = buffer1._size + buffer2._size
    
    # Create a new replay buffer with the combined capacity
    new_buffer = ReplayBuffer(
        observation_space=buffer1.dataset_dict['observations'].shape[1:],
        action_space=buffer1.dataset_dict['actions'].shape[1:],
        capacity=total_size
    )

    # Combine the data from both buffers
    for key in buffer1.dataset_dict.keys():
        if isinstance(buffer1.dataset_dict[key], np.ndarray):
            new_buffer.dataset_dict[key] = np.concatenate([
                buffer1.dataset_dict[key][:buffer1._size],
                buffer2.dataset_dict[key][:buffer2._size]
            ])
        elif isinstance(buffer1.dataset_dict[key], dict):
            new_buffer.dataset_dict[key] = combine(
                buffer1.dataset_dict[key],
                buffer2.dataset_dict[key]
            )

    # Update the new buffer's metadata
    new_buffer._size = total_size
    new_buffer._capacity = total_size
    new_buffer._insert_index = 0

    return new_buffer

def extract_subset(
    replay_buffer: ReplayBuffer,
    subset_size: int,
    seed: Optional[int] = None
) -> ReplayBuffer:
    """
    Extract a random subset of the replay buffer using the existing _subselect method.

    Args:
        replay_buffer (ReplayBuffer): The original replay buffer.
        subset_size (int): The size of the subset to extract. 
        seed (Optional[int]): Seed for random number generation.

    Returns:
        ReplayBuffer: A new replay buffer containing the randomly extracted subset.
    """
    rng = np.random.default_rng(seed)
    
    assert subset_size <= replay_buffer._size, "Subset size cannot be larger than the original buffer"

    # Randomly select indices
    indices = rng.choice(replay_buffer._size, size=subset_size, replace=False)

    # Use the _subselect method to extract the subset
    subset_dict = _subselect(replay_buffer.dataset_dict, indices)

    # Create a new replay buffer with the subset of data
    new_buffer = ReplayBuffer(
        observation_space=replay_buffer.dataset_dict['observations'].shape[1:],
        action_space=replay_buffer.dataset_dict['actions'].shape[1:],
        capacity=subset_size
    )

    new_buffer.dataset_dict = subset_dict
    new_buffer._size = subset_size
    new_buffer._capacity = subset_size
    new_buffer._insert_index = 0

    return new_buffer