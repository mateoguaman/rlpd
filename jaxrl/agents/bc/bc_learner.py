"""Implementations of algorithms for continuous control."""

from functools import partial
from typing import Dict, Tuple, Optional, Sequence, Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
import flax
import math
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState

from jaxrl import networks
from jaxrl.agents.agent import Agent
from jaxrl.data.dataset import DatasetDict
from jaxrl.types import Params, PRNGKey
from jaxrl.networks import MLP
from jaxrl.distributions import Normal, TanhNormal

nonpytree_field = partial(flax.struct.field, pytree_node=False)

class BCLearner(Agent):
    lr_schedule: Any = nonpytree_field()
    step: int = 0

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        actor_lr: float = 3e-4,
        warmup_steps: int = 1000,
        decay_steps: int = 1000000,
        hidden_dims: Sequence[int] = (256, 256),
        dropout_rate: Optional[float] = None,
        weight_decay: Optional[float] = None,
        distr: str = "tanh_normal",
    ):
        rng = jax.random.PRNGKey(seed)
        rng, actor_key = jax.random.split(rng)

        action_dim = action_space.shape[-1]
        observations = observation_space.sample()
        actions = action_space.sample()

        actor_base_cls = partial(
            MLP,
            hidden_dims=hidden_dims,
            activate_final=True,
        )

        if distr == "unitstd_normal":
            actor_def = Normal(
                base_cls=actor_base_cls,
                action_dim=action_dim,
                state_dependent_std=False
            )
        elif distr == "normal":
            actor_def = Normal(
                base_cls=actor_base_cls,
                action_dim=action_dim,
                state_dependent_std=True
            )
        elif distr == "tanh_normal":
            actor_def = TanhNormal(
                base_cls=actor_base_cls,
                action_dim=action_dim,
                log_std_min=math.log(0.1),
                log_std_max=math.log(0.1),
                state_dependent_std=False,
            )

        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=actor_lr,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=0.0,
        )

        if weight_decay is not None:
            tx = optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay)
        else:
            tx = optax.adam(learning_rate=lr_schedule)

        params = actor_def.init(actor_key, observations)["params"]

        actor = TrainState.create(
            apply_fn=actor_def.apply, 
            params=params, 
            tx=tx
        )

        step = 0

        return cls(
            rng=rng,
            actor=actor,
            lr_schedule=lr_schedule,
            step=step
        )
    def initialize_pretrained_model(self, pretrained_agent):
        new_actor = self.actor.replace(params=pretrained_agent["actor"]["params"])
        ## TODO: Check if below is necessary
        new_lr_schedule = pretrained_agent["lr_schedule"]
        new_step = pretrained_agent["step"]


        return self.replace(
            actor=new_actor,
            lr_schedule=new_lr_schedule,
            step=new_step
        )
    
    @jax.jit
    def update(self, batch: DatasetDict):
        rng = self.rng
        key, rng = jax.random.split(rng)

        def loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            dist = self.actor.apply_fn(
                {"params": actor_params},
                batch["observations"],
                training=True,
                rngs={"dropout": key},
            )
            # actions, log_probs = dist.sample_and_log_prob(seed=key)
            pi_actions = dist.mode()
            log_probs = dist.log_prob(batch["actions"])
            mse = ((pi_actions - batch["actions"]) ** 2).sum(-1)
            actor_loss = -log_probs.mean()
            # actor_std = dist.stddev().mean(axis=1)

            return actor_loss, {
                "bc_loss": actor_loss,
                "mse": mse.mean(),
                "log_probs": log_probs.mean(),
                # "pi_actions": pi_actions,
                # "mean_std": actor_std.mean(),
                # "max_std": actor_std.max(),
            }
        
        grads, actor_info = jax.grad(loss_fn, has_aux=True)(self.actor.params)
        actor = self.actor.apply_gradients(grads=grads)

        actor_info["lr"] = self.lr_schedule(self.step)

        return self.replace(actor=actor, rng=rng, step=self.step+1), actor_info