# UW RLL jaxrl

This jaxrl codebase is modified from https://github.com/ikostrikov/jaxrl. Specifically, the style, structure, and SAC implementation comes from https://github.com/ikostrikov/rlpd. The IQL implementation is adapted from https://github.com/kylestach/fastrlap-release/blob/main/jaxrl5/jaxrl5/agents/iql/iql_learner.py. The BC implementation is adapted from https://github.com/ikostrikov/jaxrl2. The TD3 implementation is adapted from jaxrl5 (internal).

# Installation

```bash
conda create -n jaxrl python=3.10
conda activate jaxrl
pip install -e .
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html  # Note: wheels only available on linux.

```
