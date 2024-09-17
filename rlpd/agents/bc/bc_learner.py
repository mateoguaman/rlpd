"""Implementations of algorithms for continuous control."""

from functools import partial
from typing import Dict, Tuple, Optional, Sequence, Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
import flax
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState

from rlpd import networks
from rlpd.agents.agent import Agent
from rlpd.data.dataset import DatasetDict
from rlpd.types import Params, PRNGKey
from rlpd.networks import MLP
from distributions import Normal, TanhNormal

# def log_prob_update(
#     rng: PRNGKey, actor: TrainState, batch: FrozenDict
# ) -> Tuple[PRNGKey, TrainState, Dict[str, float]]:
#     rng, key = jax.random.split(rng)

#     def loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
#         dist = actor.apply_fn(
#             {"params": actor_params},
#             batch["observations"],
#             training=True,
#             rngs={"dropout": key},
#         )
#         log_probs = dist.log_prob(batch["actions"])
#         actor_loss = -log_probs.mean()
#         return actor_loss, {"bc_loss": actor_loss}

#     grads, info = jax.grad(loss_fn, has_aux=True)(actor.params)
#     new_actor = actor.apply_gradients(grads=grads)

#     return rng, new_actor, info

# _log_prob_update_jit = jax.jit(log_prob_update)


# class BCLearner(Agent):
#     def __init__(
#         self,
#         seed: int,
#         observations: jnp.ndarray,
#         actions: jnp.ndarray,
#         actor_lr: float = 1e-3,
#         decay_steps: Optional[int] = None,
#         hidden_dims: Sequence[int] = (256, 256),
#         dropout_rate: Optional[float] = None,
#         weight_decay: Optional[float] = None,
#         distr: str = "tanh_normal",
#         apply_tanh: bool = True,
#     ):

#         rng = jax.random.PRNGKey(seed)
#         rng, actor_key = jax.random.split(rng)

#         action_dim = actions.shape[-1]
#         if distr == "unitstd_normal":
#             actor_def = networks.UnitStdNormalPolicy(
#                 hidden_dims,
#                 action_dim,
#                 dropout_rate=dropout_rate,
#                 apply_tanh=apply_tanh,
#             )
#         elif distr == "tanh_normal":
#             actor_def = networks.NormalTanhPolicy(
#                 hidden_dims, action_dim, dropout_rate=dropout_rate
#             )
#         elif distr == "ar":
#             actor_def = networks.ARPolicy(
#                 hidden_dims, action_dim, dropout_rate=dropout_rate
#             )

#         if decay_steps is not None:
#             actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

#         if weight_decay is not None:
#             optimiser = optax.adamw(learning_rate=actor_lr, weight_decay=weight_decay)
#         else:
#             optimiser = optax.adam(learning_rate=actor_lr)

#         params = actor_def.init(actor_key, observations)["params"]
#         self._actor = TrainState.create(
#             apply_fn=actor_def.apply, params=params, tx=optimiser
#         )
#         self._rng = rng

#     def update(self, batch: FrozenDict) -> Dict[str, float]:
#         self._rng, self._actor, info = _log_prob_update_jit(
#             self._rng, self._actor, batch
#         )
#         return info
    


## Mateo's implementation
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
        apply_tanh: bool = True,
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
            actor_std = dist.stddev().mean(axis=1)

            return actor_loss, {
                "bc_loss": actor_loss,
                "mse": mse.mean(),
                "log_probs": log_probs,
                "pi_actions": pi_actions,
                "mean_std": actor_std.mean(),
                "max_std": actor_std.max(),}
        
        grads, actor_info = jax.grad(loss_fn, has_aux=True)(self.actor.params)
        actor = self.actor.apply_gradients(grads=grads)

        actor_info["lr"] = self.lr_schedule(self.step)

        return self.replace(actor=actor, rng=rng, step=self.step+1), actor_info