from functools import partial

import einops
import jax
import jax.numpy as jnp
import numpy as np
import optax

from diffuser.diffusion import GaussianDiffusion, ModelMeanType
from utilities.jax_utils import next_rng, value_and_multi_grad
from utilities.flax_utils import apply_ema_decay, copy_params_to_ema, TrainState

from .base_algo import Algo


class DecisionDiffuser(Algo):
    def __init__(self, cfg, planner, inv_model, reward_model):
        self.config = cfg
        self.planner = planner
        self.inv_model = inv_model
        self.reward_model = reward_model
        self.horizon = self.config.horizon
        # import ipdb
        # ipdb.set_trace()
        self.edit_sar = self.config.edit_sar
        self.use_cross = self.config.use_cross

        assert inv_model is None or not self.edit_sar

        # if inv_model is None:
        #     self.observation_dim = planner.sample_dim - planner.action_dim
        #     self.action_dim = planner.action_dim
        #     self.reward_dim = 1
        if self.edit_sar:
            self.observation_dim = planner.sample_dim - planner.action_dim - 1
            self.action_dim = planner.action_dim
            self.reward_dim = 1
        else:
            self.observation_dim = planner.sample_dim
            self.action_dim = inv_model.action_dim
            self.reward_dim = 1

        self.diffusion: GaussianDiffusion = self.planner.diffusion
        self.diffusion.loss_weights = self.get_loss_weights(
            self.config.loss_discount,
            self.config.action_weight if inv_model is None else None,
            self.config.reward_weight if reward_model is None else None,
        )

        self._total_steps = 0
        self._train_states = {}

        def get_lr(lr_decay=False):
            return self.config.lr

        def get_optimizer(lr_decay=False, weight_decay=cfg.weight_decay):
            opt = optax.adam(self.config.lr)
            return opt
        
        # import ipdb
        # ipdb.set_trace()
        
        planner_params = self.planner.init(
            next_rng(),
            next_rng(),
            jnp.zeros((10, self.horizon, self.planner.sample_dim)),  # samples
            {0: jnp.zeros((10, planner.sample_dim))},  # conditions
            {-1: jnp.zeros((10, planner.sample_dim))}, # goal_conditions
            jnp.zeros((10,), dtype=jnp.int32),  # ts
            returns=jnp.zeros((10, 1)),  # returns
            method=self.planner.loss,
        )
        self._train_states["planner"] = TrainState.create(
            params=planner_params,
            params_ema=planner_params,
            tx=get_optimizer(self.config.lr_decay, weight_decay=0.0),
            apply_fn=None,
        )

        model_keys = ["planner"]
        if inv_model is not None:
            inv_model_params = self.inv_model.init(
                next_rng(),
                jnp.zeros((10, self.observation_dim * 2)),
            )
            self._train_states["inv_model"] = TrainState.create(
                params=inv_model_params,
                tx=get_optimizer(),
                apply_fn=None,
            )
            model_keys.append("inv_model")

        if reward_model is not None:
            reward_model_params = self.reward_model.init(
                next_rng(),
                jnp.zeros((10, self.observation_dim * 2 + self.action_dim)),
            )
            self._train_states["reward_model"] = TrainState.create(
                params=reward_model_params,
                tx=get_optimizer(),
                apply_fn=None,
            )
            model_keys.append("reward_model")

        self._model_keys = tuple(model_keys)

    def get_loss_weights(self, discount: float, act_weight: float, reward_weight: float) -> jnp.ndarray:
        dim_weights = np.ones(self.planner.sample_dim, dtype=np.float32)

        # decay loss with trajectory timestep: discount**t
        discounts = discount ** np.arange(self.horizon, dtype=np.float32)
        discounts = discounts / discounts.mean()
        loss_weights = einops.einsum(discounts, dim_weights, "h,t->h t")
        # Cause things are conditioned on t=0
        if self.diffusion.model_mean_type == ModelMeanType.EPSILON:
            loss_weights[0, :] = 0
        if self.inv_model is None:
            loss_weights[0, -self.action_dim - self.reward_dim : -self.reward_dim] = act_weight
        if self.reward_model is None:
            loss_weights[0, -self.reward_dim:] = reward_weight

        return jnp.array(loss_weights)

    @partial(jax.jit, static_argnames=("self"))
    def _train_step(self, train_states, rng, batch):
        diff_loss_fn = self.get_diff_loss(batch)

        params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_planner), grad_planner = value_and_multi_grad(
            diff_loss_fn, 1, has_aux=True
        )(params, rng)

        train_states["planner"] = train_states["planner"].apply_gradients(
            grads=grad_planner[0]["planner"]
        )
        metrics = dict(diff_loss=aux_planner["loss"])

        if self.inv_model is not None:
            inv_loss_fn = self.get_inv_loss(batch)
            params = {key: train_states[key].params for key in self.model_keys}
            (_, aux_inv_model), grad_inv_model = value_and_multi_grad(
                inv_loss_fn, 1, has_aux=True
            )(params, rng)

            train_states["inv_model"] = train_states["inv_model"].apply_gradients(
                grads=grad_inv_model[0]["inv_model"]
            )
            metrics["inv_loss"] = aux_inv_model["loss"]
        
        if self.reward_model is not None:
            reward_loss_fun = self.get_reward_loss(batch)
            params = {key: train_states[key].params for key in self.model_keys}
            (_, aux_reward_model), grad_reward_model = value_and_multi_grad(
                reward_loss_fun, 1, has_aux=True
            )(params, rng)

            train_states["reward_model"] = train_states["reward_model"].apply_gradients(
                grads=grad_reward_model[0]["reward_model"]
            )
            metrics["reward_loss"] = aux_reward_model["loss"]

        return train_states, metrics

    def get_inv_loss(self, batch):
        def inv_loss(params, rng):
            samples = batch["samples"]
            actions = batch["actions"]

            samples_t = samples[:, :-1]
            samples_tp1 = samples[:, 1:]
            samples_comb = jnp.concatenate([samples_t, samples_tp1], axis=-1)
            samples_comb = jnp.reshape(samples_comb, (-1, self.observation_dim * 2))

            # TODO: What? Suppose to be actions[:,:-1]
            actions = actions[:, :-1]
            actions = jnp.reshape(actions, (-1, self.action_dim))

            pred_actions = self.inv_model.apply(params["inv_model"], samples_comb)
            # import ipdb
            # ipdb.set_trace()
            loss = jnp.mean((pred_actions - actions) ** 2)
            return (loss,), locals()

        return inv_loss
    
    def get_reward_loss(self, batch):
        def reward_loss(params, rng):
            samples = batch["samples"]
            actions = batch["actions"]
            rewards = batch["rewards"]

            samples_t = samples[:, :-1]
            samples_tp1 = samples[:, 1:]
            

            # TODO: What? Suppose to be actions[:,:-1]
            actions = actions[:, :-1]
            # actions = jnp.reshape(actions, (-1, self.action_dim))
            rewards = rewards[:, :-1]

            samples_comb = jnp.concatenate([samples_t, samples_tp1, actions], axis=-1)
            samples_comb = jnp.reshape(samples_comb, (-1, self.observation_dim * 2 + self.action_dim))

            pred_reward = self.reward_model.apply(params["reward_model"], samples_comb)
            # import ipdb
            # ipdb.set_trace()
            loss = jnp.mean((pred_reward.reshape(rewards.shape) - rewards) ** 2)
            return (loss,), locals()

        return reward_loss

    def get_diff_loss(self, batch):
        def diff_loss(params, rng):
            
            # import ipdb
            # ipdb.set_trace()
            if self.inv_model is None and not self.edit_sar:
                samples = jnp.concatenate((batch["samples"], batch["actions"]), axis=-1)
            elif  self.inv_model is None and self.edit_sar:
                samples = jnp.concatenate((batch["samples"], batch["actions"], jnp.expand_dims(batch["rewards"], axis = -1)), axis=-1)
            else:
                samples = batch["samples"]
            
            conditions = batch["conditions"]
            goal_conditions = batch["goal_conditions"]
            returns = batch.get("returns", None)
            # import ipdb
            # ipdb.set_trace()

            terms, ts = self.get_diff_terms(
                params, samples, conditions, goal_conditions, returns, rng
            )
            loss = terms["loss"].mean()

            return (loss,), locals()
        # import ipdb
        # ipdb.set_trace()
        return diff_loss

    def get_diff_terms(self, params, samples, conditions, goal_conditions, returns, rng):
        # import ipdb
        # ipdb.set_trace()
        rng, split_rng = jax.random.split(rng)
        ts = jax.random.randint(
            split_rng,
            (samples.shape[0],),
            minval=0,
            maxval=self.diffusion.num_timesteps,
        )
        rng, split_rng = jax.random.split(rng)
        terms = self.planner.apply(
            params["planner"],
            split_rng,
            samples,
            conditions,
            goal_conditions,
            ts,
            returns=returns,
            method=self.planner.loss,
        )

        return terms, ts

    def train(self, batch):
        self._total_steps += 1
        self._train_states, metrics = self._train_step(
            self._train_states, next_rng(), batch
        )
        if self._total_steps % self.config.update_ema_every == 0:
            self.step_ema()
        return metrics

    def step_ema(self):
        if self._total_steps < self.config.step_start_ema:
            self._train_states["planner"] = copy_params_to_ema(
                self._train_states["planner"]
            )
        else:
            self._train_states["planner"] = apply_ema_decay(
                self._train_states["planner"], self.config.ema_decay
            )

    @property
    def model_keys(self):
        return self._model_keys

    @property
    def train_states(self):
        return self._train_states

    @property
    def train_params(self):
        return {key: self.train_states[key].params for key in self.model_keys}

    @property
    def eval_params(self):
        return {
            key: self.train_states[key].params_ema or self.train_states[key].params
            for key in self.model_keys
        }

    @property
    def total_steps(self):
        return self._total_steps
