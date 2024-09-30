from typing import Any

import jax
from flax.training import train_state


class TrainState(train_state.TrainState):
    params_ema: Any = None


def copy_params_to_ema(state):
    state = state.replace(params_ema=state.params)
    return state


def apply_ema_decay(state, ema_decay: float):
    params_ema = jax.tree_map(lambda p_ema, p: p_ema * ema_decay + p * (1. - ema_decay), state.params_ema, state.params)
    state = state.replace(params_ema=params_ema)
    return state
