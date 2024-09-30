from functools import partial

import jax
import jax.numpy as jnp

from utilities.jax_utils import next_rng


class SamplerPolicy(object):  # used for dql
    def __init__(
        self, policy, qf=None, mean=0, std=1, ensemble=False, act_method="ddpm"
    ):
        self.policy = policy
        self.qf = qf
        self.mean = mean
        self.std = std
        self.num_samples = 50
        self.act_method = act_method

    def update_params(self, params):
        self.params = params
        return self

    @partial(jax.jit, static_argnames=("self", "deterministic"))
    def act(self, params, rng, observations, deterministic):
        conditions = {}
        return self.policy.apply(
            params["policy"], rng, observations, conditions, deterministic, repeat=None
        )

    @partial(jax.jit, static_argnames=("self", "deterministic", "num_samples"))
    def ensemble_act(self, params, rng, observations, deterministic, num_samples):
        rng, key = jax.random.split(rng)
        conditions = {}
        actions = self.policy.apply(
            params["policy"],
            key,
            observations,
            conditions,
            deterministic,
            repeat=num_samples,
        )
        q1 = self.qf.apply(params["qf1"], observations, actions)
        q2 = self.qf.apply(params["qf2"], observations, actions)
        q = jnp.minimum(q1, q2)

        idx = jax.random.categorical(rng, q)
        return actions[jnp.arange(actions.shape[0]), idx]

    @partial(jax.jit, static_argnames=("self", "deterministic", "num_samples"))
    def ddpmensemble_act(self, params, rng, observations, deterministic, num_samples):
        rng, key = jax.random.split(rng)
        conditions = {}
        actions = self.policy.apply(
            params["policy"],
            rng,
            observations,
            conditions,
            deterministic,
            repeat=num_samples,
            method=self.policy.ddpm_sample,
        )
        q1 = self.qf.apply(params["qf1"], observations, actions)
        q2 = self.qf.apply(params["qf2"], observations, actions)
        q = jnp.minimum(q1, q2)

        idx = jax.random.categorical(rng, q)
        return actions[jnp.arange(actions.shape[0]), idx]

    @partial(jax.jit, static_argnames=("self", "deterministic", "num_samples"))
    def dpmensemble_act(self, params, rng, observations, deterministic, num_samples):
        rng, key = jax.random.split(rng)
        conditions = {}
        actions = self.policy.apply(
            params["policy"],
            rng,
            observations,
            conditions,
            deterministic,
            repeat=num_samples,
            method=self.policy.dpm_sample,
        )
        q1 = self.qf.apply(params["qf1"], observations, actions)
        q2 = self.qf.apply(params["qf2"], observations, actions)
        q = jnp.minimum(q1, q2)

        idx = jax.random.categorical(rng, q)
        return actions[jnp.arange(actions.shape[0]), idx]

    @partial(jax.jit, static_argnames=("self", "deterministic", "num_samples"))
    def dpm_act(self, params, rng, observations, deterministic, num_samples):
        conditions = {}
        return self.policy.apply(
            params["policy"],
            rng,
            observations,
            conditions,
            deterministic,
            method=self.policy.dpm_sample,
        )

    @partial(jax.jit, static_argnames=("self", "deterministic", "num_samples"))
    def ddim_act(self, params, rng, observations, deterministic, num_samples):
        conditions = {}
        return self.policy.apply(
            params["policy"],
            rng,
            observations,
            conditions,
            deterministic,
            method=self.policy.ddim_sample,
        )

    @partial(jax.jit, static_argnames=("self", "deterministic", "num_samples"))
    def ddpm_act(self, params, rng, observations, deterministic, num_samples):
        conditions = {}
        return self.policy.apply(
            params["policy"],
            rng,
            observations,
            conditions,
            deterministic,
            method=self.policy.ddpm_sample,
        )

    def __call__(self, observations, deterministic=False):
        actions = getattr(self, f"{self.act_method}_act")(
            self.params, next_rng(), observations, deterministic, self.num_samples
        )
        if isinstance(actions, tuple):
            actions = actions[0]
        assert jnp.all(jnp.isfinite(actions))
        return jax.device_get(actions)


class DiffuserPolicy(object):
    def __init__(self, planner, inv_model, reward_model, act_method="ddpm"):
        self.planner = planner
        self.inv_model = inv_model
        self.reward_model = reward_model
        self.act_method = act_method

    def update_params(self, params):
        self.params = params
        return self

    @partial(jax.jit, static_argnames=("self", "deterministic"))
    def ddpm_act(
        self, params, rng, observations, deterministic
    ):  # deterministic is not used
        conditions = {0: observations}
        returns = jnp.ones((observations.shape[0], 1)) * 0.9
        plan_samples = self.planner.apply(
            params["planner"],
            rng,
            conditions=conditions,
            returns=returns,
            method=self.planner.ddpm_sample,
        )
        
        if self.inv_model is not None:
            obs_comb = jnp.concatenate(
                [plan_samples[:, 0], plan_samples[:, 1]], axis=-1
            )
            actions = self.inv_model.apply(
                params["inv_model"],
                obs_comb,
            )
        else:
            actions = plan_samples[:, 0, -self.planner.action_dim :]
        # import ipdb
        # ipdb.set_trace()
        return actions
    
    @partial(jax.jit, static_argnames=("self", "deterministic"))
    def ddpm_gen(
        self, params, rng, observations, deterministic
    ):  # deterministic is not used
        
        result_dict = {}
        for key in range(observations.shape[0]):
            result_dict[key] = observations[key]
    
        import ipdb
        # ipdb.set_trace()
        
        returns = jnp.ones((1, 1)) * 0.9
        
        t = 0.05
        plan_samples = self.planner.apply(
            params["planner"],
            rng,
            conditions=result_dict,
            returns=returns,
            x = observations,
            method=self.planner.gen_ddpm_sample,
            t = t
        )
        
        if self.inv_model is not None:
            obs_comb = jnp.concatenate(
                [plan_samples[:, 0], plan_samples[:, 1]], axis=-1
            )
            ipdb.set_trace()
            actions = self.inv_model.apply(
                params["inv_model"],
                obs_comb,
            )
        else:
            actions = plan_samples[:, :, -self.planner.action_dim :]

        return plan_samples
        
    @partial(jax.jit, static_argnames=("self", "deterministic"))
    def ddpm_gen_act(
        self, params, rng, observations, actions, rewards, deterministic, t_add, t_de
    ):  # deterministic is not used
        
        result_dict = {}
        for key in range(observations.shape[0]):
            result_dict[key] = jnp.concatenate([observations[key], actions[key], rewards[key]], axis=0)
        
        # import ipdb
        # ipdb.set_trace()
        if self.inv_model is None and self.reward_model is None:
            edit_sar = True
        else:
            edit_sar = False
        returns = jnp.ones((1, 1)) * 0.9
        # import ipdb
        # ipdb.set_trace()
        conditions = {0: observations[0]} if not edit_sar else {0: jnp.concatenate([observations[0], actions[0], rewards[0]], axis=0)}
        goal_conditions = {-1: observations[-1]} if not edit_sar else {-1: jnp.concatenate([observations[-1], actions[-1], rewards[-1]], axis=0)}
        # t = 0.05
        plan_samples, plan_noise_samples = self.planner.apply(
            params["planner"],
            rng,
            conditions=conditions,
            goal_conditions=goal_conditions,
            origin_data=result_dict,
            returns=returns,
            x = observations,
            method=self.planner.gen_ddpm_sample,
            t_add = t_add,
            t_de = t_de,
        )
        # ipdb.set_trace()
        # if self.inv_model is not None:
        #     obs_comb = jnp.concatenate(
        #         [plan_samples[:, 0], plan_samples[:, 1]], axis=-1
        #     )
        #     actions = self.inv_model.apply(
        #         params["inv_model"],
        #         obs_comb,
        #     )
        # else:
        #     actions = plan_samples[:, :, -self.planner.action_dim :]
        # ipdb.set_trace()
        # observations = plan_samples[:,:,:self.planner.observation_dim - 1]
        if self.inv_model is None:
            observations = plan_samples[:,:,: -self.planner.action_dim - self.planner.reward_dim]
            actions = plan_samples[:, :, -self.planner.action_dim - self.planner.reward_dim : -self.planner.reward_dim]
            # noisy_observations = plan_noise_samples[:,:,:-self.planner.action_dim]
            # noisy_actions = plan_noise_samples[:, :, -self.planner.action_dim :]
        else:
            observations = plan_samples
            actions = []
            # ipdb.set_trace()
            for i in range(observations.shape[1] - 1):
                obs_comb = jnp.concatenate(
                    [observations[0][i], observations[0][i+1]], axis=-1
                )
                # ipdb.set_trace()
                action = self.inv_model.apply(
                    params["inv_model"],
                    obs_comb,
                )
                actions.append(action)
            # ipdb.set_trace()
            actions.append(actions[-1])
            actions = jnp.array(actions)
            # import numpy as np
            # actions = np.array(actions).reshape((1, observations.shape[0], 1))
            # noisy_actions = None
            # noisy_observations = None

        if self.reward_model is not None:
            # observations = plan_samples
            rewards = []
            # ipdb.set_trace()
            for i in range(observations.shape[1] - 1):
                # import ipdb
                # ipdb.set_trace()
                obs_comb = jnp.concatenate(
                    [observations[0][i], observations[0][i+1], actions[i]], axis=-1
                )
                # ipdb.set_trace()
                reward = self.reward_model.apply(
                    params["reward_model"],
                    obs_comb,
                )
                rewards.append(reward)
            # ipdb.set_trace()
            rewards.append(rewards[-1])
            rewards = jnp.array(rewards)
            # import numpy as np
            # actions = np.array(actions).reshape((1, observations.shape[0], 1))
            # noisy_actions = None
            # noisy_observations = None

        else:
            rewards = plan_samples[:, :, -self.planner.reward_dim :]

        
        return observations, actions, rewards, None, None

    @partial(jax.jit, static_argnames=("self", "deterministic"))
    def ddim_act(
        self, params, rng, observations, deterministic
    ):  # deterministic is not used
        conditions = {0: observations}
        returns = jnp.ones((observations.shape[0], 1)) * 0.9
        plan_samples = self.planner.apply(
            params["planner"],
            rng,
            conditions=conditions,
            returns=returns,
            method=self.planner.ddim_sample,
        )

        if self.inv_model is not None:
            obs_comb = jnp.concatenate(
                [plan_samples[:, 0], plan_samples[:, 1]], axis=-1
            )
            actions = self.inv_model.apply(
                params["inv_model"],
                obs_comb,
            )
        else:
            actions = plan_samples[:, 0, -self.planner.action_dim :]

        return actions

    def __call__(self, observations, actions = None, rewards = None, deterministic=False, choice = 'act', t_add = 0.05, t_de = 0.05):
        import ipdb
        # ipdb.set_trace()
        if actions is None:
            returns = getattr(self, f"{self.act_method}_{choice}")(
                self.params, next_rng(), observations, deterministic
            )
        else:
            returns = getattr(self, f"{self.act_method}_{choice}_act")(
                self.params, next_rng(), observations, actions, rewards, deterministic, t_add, t_de
            )
            import numpy as np
            import ipdb
            # ipdb.set_trace()
            returns = (returns[0], np.array(returns[1]).reshape((1, self.planner.horizon, returns[1].shape[-1])), np.array(returns[2]).reshape((1, self.planner.horizon, 1)), returns[3], returns[4])
        # assert jnp.all(jnp.isfinite(returns))
        return jax.device_get(returns)
