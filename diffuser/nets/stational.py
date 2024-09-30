from functools import partial
from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import repeat

from diffuser.diffusion import GaussianDiffusion, ModelMeanType, _extract_into_tensor
from diffuser.dpm_solver import DPM_Solver, NoiseScheduleVP
from diffuser.nets.helpers import TimeEmbedding, mish, multiple_action_q_function
from utilities.jax_utils import extend_and_repeat


class PolicyNet(nn.Module):
    output_dim: int
    arch: Tuple = (256, 256, 256)
    time_embed_size: int = 16
    act: callable = mish
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, state, rng, action, t):
        if len(t.shape) < len(action.shape) - 1:
            t = repeat(t, "b -> b n", n=action.shape[1])
        time_embed = TimeEmbedding(self.time_embed_size, self.act)(t)
        x = jnp.concatenate([state, action, time_embed], axis=-1)

        for feat in self.arch:
            x = nn.Dense(feat)(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = self.act(x)

        x = nn.Dense(self.output_dim)(x)
        return x


class DiffusionPolicy(nn.Module):
    diffusion: GaussianDiffusion
    observation_dim: int
    action_dim: int
    arch: Tuple = (256, 256, 256)
    time_embed_size: int = 16
    act: callable = mish
    use_layer_norm: bool = False
    use_dpm: bool = False
    sample_method: str = "ddpm"
    dpm_steps: int = 15
    dpm_t_end: float = 0.001

    def setup(self):
        self.base_net = PolicyNet(
            output_dim=self.action_dim,
            arch=self.arch,
            time_embed_size=self.time_embed_size,
            act=self.act,
            use_layer_norm=self.use_layer_norm,
        )

    def __call__(self, rng, observations, conditions, deterministic=False, repeat=None):
        return getattr(self, f"{self.sample_method}_sample")(
            rng, observations, conditions, deterministic, repeat
        )

    def ddpm_sample(
        self, rng, observations, conditions, deterministic=False, repeat=None
    ):
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)

        shape = observations.shape[:-1] + (self.action_dim,)

        return self.diffusion.p_sample_loop(
            rng_key=rng,
            model_forward=partial(self.base_net, observations),
            shape=shape,
            conditions=conditions,
            clip_denoised=True,
        )

    def dpm_sample(
        self, rng, observations, conditions, deterministic=False, repeat=None
    ):
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        noise_clip = True

        shape = observations.shape[:-1] + (self.action_dim,)

        ns = NoiseScheduleVP(
            schedule="discrete", alphas_cumprod=self.diffusion.alphas_cumprod
        )

        def wrap_model(model_fn):
            def wrapped_model_fn(x, t):
                t = (t - 1.0 / ns.total_N) * ns.total_N

                out = model_fn(x, t)
                # add noise clipping
                if noise_clip:
                    t = t.astype(jnp.int32)
                    x_w = _extract_into_tensor(
                        self.diffusion.sqrt_recip_alphas_cumprod, t, x.shape
                    )
                    e_w = _extract_into_tensor(
                        self.diffusion.sqrt_recipm1_alphas_cumprod, t, x.shape
                    )
                    max_value = (self.diffusion.max_value + x_w * x) / e_w
                    min_value = (self.diffusion.min_value + x_w * x) / e_w

                    out = out.clip(min_value, max_value)
                return out

            return wrapped_model_fn

        dpm_sampler = DPM_Solver(
            model_fn=wrap_model(partial(self.base_net, observations)),
            noise_schedule=ns,
            predict_x0=self.diffusion.model_mean_type is ModelMeanType.START_X,
        )
        x = jax.random.normal(rng, shape)
        out = dpm_sampler.sample(x, steps=self.dpm_steps, t_end=self.dpm_t_end)

        return out

    def ddim_sample(
        self, rng, observations, conditions, deterministic=False, repeat=None
    ):
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)

        shape = observations.shape[:-1] + (self.action_dim,)

        return self.diffusion.ddim_sample_loop(
            rng_key=rng,
            model_forward=partial(self.base_net, observations),
            shape=shape,
            conditions=conditions,
            clip_denoised=True,
        )

    def loss(self, rng_key, observations, actions, conditions, ts):
        terms = self.diffusion.training_losses(
            rng_key,
            model_forward=partial(self.base_net, observations),
            x_start=actions,
            conditions=conditions,
            t=ts,
        )
        return terms

    @property
    def max_action(self):
        return self.diffusion.max_value


class Critic(nn.Module):
    observation_dim: int
    action_dim: int
    arch: Tuple = (256, 256, 256)
    act: callable = mish
    use_layer_norm: bool = False
    orthogonal_init: bool = False

    @nn.compact
    @multiple_action_q_function
    def __call__(self, observations, actions):
        x = jnp.concatenate([observations, actions], axis=-1)

        for feat in self.arch:
            if self.orthogonal_init:
                x = nn.Dense(
                    feat,
                    kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2.0)),
                    bias_init=jax.nn.initializers.zeros,
                )(x)
            else:
                x = nn.Dense(feat)(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = self.act(x)

        if self.orthogonal_init:
            x = nn.Dense(
                1,
                kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2.0)),
                bias_init=jax.nn.initializers.zeros,
            )(x)
        else:
            x = nn.Dense(1)(x)
        return jnp.squeeze(x, -1)

    @property
    def input_size(self):
        return self.observation_dim + self.action_dim


class Value(nn.Module):
    observation_dim: int
    arch: Tuple = (256, 256, 256)
    act: callable = mish
    use_layer_norm: bool = False
    orthogonal_init: bool = False

    @nn.compact
    def __call__(self, observations):
        x = observations

        for feat in self.arch:
            if self.orthogonal_init:
                x = nn.Dense(
                    feat,
                    kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2.0)),
                    bias_init=jax.nn.initializers.zeros,
                )(x)
            else:
                x = nn.Dense(feat)(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = self.act(x)

        if self.orthogonal_init:
            x = nn.Dense(
                1,
                kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2.0)),
                bias_init=jax.nn.initializers.zeros,
            )(x)
        else:
            x = nn.Dense(1)(x)
        return jnp.squeeze(x, -1)

    @property
    def input_size(self):
        return self.observation_dim


class InverseDynamic(nn.Module):
    action_dim: int
    hidden_dims: Tuple[int] = (256, 256)

    @nn.compact
    def __call__(self, x):
        import ipdb
        for i in range(len(self.hidden_dims)):
            # ipdb.set_trace()
            x = nn.Dense(self.hidden_dims[i])(x)
            x = nn.relu(x)
        # ipdb.set_trace()
        x = nn.Dense(self.action_dim)(x)
        return x

class RewardDynamic(nn.Module):
    reward_dim: int = 1
    hidden_dims: Tuple[int] = (256, 256)

    @nn.compact
    def __call__(self, x):
        x = nn.LayerNorm()(x)
        for i in range(len(self.hidden_dims)):
            # ipdb.set_trace()
            x = nn.Dense(self.hidden_dims[i])(x)
            x = nn.relu(x)
        # ipdb.set_trace()
        x = nn.Dense(self.reward_dim)(x)
        return x
