import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp

from utilities.jax_utils import extend_and_repeat


class Conv1dBlock(nn.Module):
    out_channels: int
    kernel_size: int
    mish: bool = True
    n_groups: int = 8

    @nn.compact
    def __call__(self, x):
        if self.mish:
            act_fn = mish
        else:
            act_fn = nn.silu

        # NOTE(zbzhu): in flax, conv use the channel last format
        x = nn.Conv(
            self.out_channels, (self.kernel_size,), padding=self.kernel_size // 2
        )(x)
        x = nn.GroupNorm(self.n_groups, epsilon=1e-5)(x)
        return act_fn(x)


class DownSample1d(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.dim, (3,), strides=(2,), padding=1)(x)
        return x


class UpSample1d(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x):
        batch, length, channels = x.shape
        x = jax.image.resize(
            x,
            shape=(batch, length * 2, channels),
            method="nearest",
        )
        x = nn.Conv(self.dim, (3,), strides=(1,), padding=1)(x)
        return x


class GaussianPolicy(nn.Module):
    action_dim: int
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    temperature: float = 1.0

    @nn.compact
    def __call__(self, mean):
        log_stds = self.param("log_stds", nn.initializers.zeros, (self.action_dim,))
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
        return distrax.MultivariateNormalDiag(
            mean, jnp.exp(log_stds * self.temperature)
        )


def multiple_action_q_function(forward):
    def wrapped(self, observations, actions, **kwargs):
        multiple_actions = False
        batch_size = observations.shape[0]
        if actions.ndim == 3 and observations.ndim == 2:
            multiple_actions = True
            observations = extend_and_repeat(observations, 1, actions.shape[1])
            observations = observations.reshape(-1, observations.shape[-1])
            actions = actions.reshape(-1, actions.shape[-1])
        q_values = forward(self, observations, actions, **kwargs)
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values

    return wrapped


def mish(x):
    return x * jnp.tanh(nn.softplus(x))


def sinusoidal_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """

    half = dim // 2
    freqs = jnp.exp(-jnp.log(max_period) * jnp.arange(half, dtype=jnp.float32) / half)
    # args = timesteps[:, None] * freqs[None, :]
    args = jnp.expand_dims(timesteps, axis=-1) * freqs[None, :]
    embd = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    return embd


class TimeEmbedding(nn.Module):
    embed_size: int
    act: callable = mish

    @nn.compact
    def __call__(self, timesteps):
        x = sinusoidal_embedding(timesteps, self.embed_size)
        x = nn.Dense(self.embed_size * 4)(x)
        x = self.act(x)
        x = nn.Dense(self.embed_size)(x)
        return x
    
class StateEmbedding(nn.Module):
    # embed_size: int
    act: callable = mish

    @nn.compact
    def __call__(self, state):
        x = state
        embed_size = x.shape[-1]
        x = nn.Dense(embed_size * 4)(x)
        x = self.act(x)
        x = nn.Dense(embed_size)(x)
        return x
    
class ActionEmbedding(nn.Module):
    # embed_size: int
    act: callable = mish

    @nn.compact
    def __call__(self, action):
        x = action
        embed_size = x.shape[-1]
        # import ipdb
        # ipdb.set_trace()
        x = nn.Dense(embed_size * 4)(x)
        x = self.act(x)
        x = nn.Dense(embed_size)(x)
        return x
    
class TimeStepEmbedding(nn.Module):
    # embed_size: int
    act: callable = mish

    @nn.compact
    def __call__(self, timestep):
        x = timestep
        embed_size = x.shape[-1]
        x = nn.Dense(embed_size * 4)(x)
        x = self.act(x)
        x = nn.Dense(embed_size)(x)
        return x
    
    
class DiTBlock(nn.Module):
    # n_embed: int
    num_heads: int
    dim: int
    # hidden_size: int
    drop_rate: float = 0.0
    mlp_ratio: float = 4.0
    
    def setup(self):
        
        self.ln1 = nn.LayerNorm()
        self.ln2 = nn.LayerNorm()
        self.attn = nn.MultiHeadDotProductAttention(num_heads = self.num_heads, out_features = self.dim)
        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        self.mlp = nn.Sequential([
            nn.Dense(features = mlp_hidden_dim),
            nn.gelu,
            nn.Dense(features = self.dim)]
        )
        self.adaLN_modulation = nn.Sequential([nn.silu,
            nn.Dense(features = 6 * self.dim)
        ])
        
    @nn.compact
    def __call__(self, x, c, mask = None):
        # import ipdb
        # ipdb.set_trace()
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunk(self.adaLN_modulation(c), chunks = 6, dim=-1)
        x = x + gate_msa.reshape(x.shape) * self.attn(modulate(self.ln1(x), shift_msa.reshape(x.shape), scale_msa.reshape(x.shape)), mask = mask)
        x = x + gate_mlp.reshape(x.shape) * self.mlp(modulate(self.ln2(x), shift_mlp.reshape(x.shape), scale_mlp.reshape(x.shape)))
        
        return x
    
class CrossDiTBlock(nn.Module):
    # n_embed: int
    num_heads: int
    dim: int
    # hidden_size: int
    drop_rate: float = 0.0
    mlp_ratio: float = 4.0
    
    def setup(self):
        
        self.ln_attn = nn.LayerNorm()
        self.ln_mlp = nn.LayerNorm()
        self.attn = nn.MultiHeadDotProductAttention(num_heads = self.num_heads, out_features = self.dim)
        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        self.mlp = nn.Sequential([
            nn.Dense(features = mlp_hidden_dim),
            nn.gelu,
            nn.Dense(features = self.dim)]
        )
        self.adaLN_modulation = nn.Sequential([nn.silu,
            nn.Dense(features = 6 * self.dim)
        ])
        
    @nn.compact
    def __call__(self, x1, x2, c, mask = None):
        # x1: q
        # x2: k,v
        # import ipdb
        # ipdb.set_trace()
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunk(self.adaLN_modulation(c), chunks = 6, dim=-1)
        x2 = x2 + gate_msa.reshape(x2.shape) * self.attn(
            inputs_q = x1, 
            inputs_k = modulate(
                self.ln_attn(x2), 
                shift_msa.reshape(x2.shape), 
                scale_msa.reshape(x2.shape)
                ),
            inputs_v = modulate(
                self.ln_attn(x2), 
                shift_msa.reshape(x2.shape), 
                scale_msa.reshape(x2.shape)
                ),
            mask = mask,)
        x2 = x2 + gate_mlp.reshape(x2.shape) * self.mlp(modulate(self.ln_mlp(x2), shift_mlp.reshape(x2.shape), scale_mlp.reshape(x2.shape)))
        
        return x2
    
class NewCrossDiTBlock(nn.Module):
    # n_embed: int
    num_heads: int
    dim: int
    # hidden_size: int
    drop_rate: float = 0.0
    mlp_ratio: float = 4.0
    
    def setup(self):
        
        self.ln_attn = nn.LayerNorm()
        self.ln_mlp = nn.LayerNorm()
        self.attn = nn.MultiHeadDotProductAttention(num_heads = self.num_heads, out_features = self.dim)
        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        self.mlp = nn.Sequential([
            nn.Dense(features = mlp_hidden_dim),
            nn.gelu,
            nn.Dense(features = self.dim)]
        )
        self.adaLN_modulation = nn.Sequential([nn.silu,
            nn.Dense(features = 6 * self.dim)
        ])
        
    @nn.compact
    def __call__(self, x1, x2, c, mask = None):
        # x1: q
        # x2: k,v
        # import ipdb
        # ipdb.set_trace()
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunk(self.adaLN_modulation(c), chunks = 6, dim=-1)
        x1 = x1 + gate_msa.reshape(x1.shape) * self.attn(
            inputs_q = modulate(
                self.ln_attn(x1), 
                shift_msa.reshape(x1.shape), 
                scale_msa.reshape(x1.shape)
                ), 
            inputs_k = x2,
            inputs_v = x2,
            mask = mask,)
        x1 = x1 + gate_mlp.reshape(x1.shape) * self.mlp(modulate(self.ln_mlp(x1), shift_mlp.reshape(x1.shape), scale_mlp.reshape(x1.shape)))
        
        return x1
    
class CrossDiTBlock_LNq(nn.Module):
    # n_embed: int
    num_heads: int
    dim: int
    # hidden_size: int
    drop_rate: float = 0.0
    mlp_ratio: float = 4.0
    
    def setup(self):
        
        self.ln_attn_kv = nn.LayerNorm()
        self.ln_attn_q = nn.LayerNorm()
        self.ln_mlp = nn.LayerNorm()
        self.attn = nn.MultiHeadDotProductAttention(num_heads = self.num_heads, out_features = self.dim)
        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        self.mlp = nn.Sequential([
            nn.Dense(features = mlp_hidden_dim),
            nn.gelu,
            nn.Dense(features = self.dim)]
        )
        self.adaLN_modulation = nn.Sequential([nn.silu,
            nn.Dense(features = 6 * self.dim)
        ])
        
    @nn.compact
    def __call__(self, x1, x2, c, mask = None):
        # x1: q
        # x2: k,v
        # import ipdb
        # ipdb.set_trace()
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunk(self.adaLN_modulation(c), chunks = 6, dim=-1)
        x2 = x2 + gate_msa.reshape(x2.shape) * self.attn(
            inputs_q = self.ln_attn_q(x1), 
            inputs_k = modulate(
                self.ln_attn_kv(x2), 
                shift_msa.reshape(x2.shape), 
                scale_msa.reshape(x2.shape)
                ),
            inputs_v = modulate(
                self.ln_attn_kv(x2), 
                shift_msa.reshape(x2.shape), 
                scale_msa.reshape(x2.shape)
                ),
            mask = mask,)
        x2 = x2 + gate_mlp.reshape(x2.shape) * self.mlp(modulate(self.ln_mlp(x2), shift_mlp.reshape(x2.shape), scale_mlp.reshape(x2.shape)))
        
        return x2
    
class NewCrossDiTBlock_AllAdaLN(nn.Module):
    # n_embed: int
    num_heads: int
    q_dim: int
    kv_dim: int
    # hidden_size: int
    drop_rate: float = 0.0
    mlp_ratio: float = 4.0
    
    def setup(self):
        
        self.ln_attn_kv = nn.LayerNorm()
        self.ln_attn_q = nn.LayerNorm()
        self.ln_mlp = nn.LayerNorm()
        self.attn = nn.MultiHeadDotProductAttention(num_heads = self.num_heads, out_features = self.q_dim)
        mlp_hidden_dim = int(self.q_dim * self.mlp_ratio)
        self.mlp = nn.Sequential([
            nn.Dense(features = mlp_hidden_dim),
            nn.gelu,
            nn.Dense(features = self.q_dim)]
        )
        self.adaLN_modulation_kv = nn.Sequential([nn.silu,
            nn.Dense(features = 2*self.kv_dim)
        ])
        self.adaLN_modulation_q = nn.Sequential([nn.silu,
            nn.Dense(features = 6*self.q_dim)
        ])
        
    @nn.compact
    def __call__(self, x1, x2, c1, c2, mask = None):
        # x1: q
        # x2: k,v
        # import ipdb
        # ipdb.set_trace()
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunk(self.adaLN_modulation_q(c1), chunks=6, dim=-1)
        shift_kv, scale_kv = chunk(self.adaLN_modulation_kv(c2), chunks=2, dim=-1)
        x1 = x1 + gate_msa.reshape(x1.shape) * self.attn(
            inputs_q = modulate(
                self.ln_attn_q(x1), 
                shift_msa.reshape(x1.shape),
                scale_msa.reshape(x1.shape)
                ), 
            inputs_k = modulate(
                self.ln_attn_kv(x2), 
                shift_kv.reshape(x2.shape), 
                scale_kv.reshape(x2.shape)
                ),
            inputs_v = modulate(
                self.ln_attn_kv(x2), 
                shift_kv.reshape(x2.shape), 
                scale_kv.reshape(x2.shape)
                ),
            mask = mask,)
        x1 = x1 + gate_mlp.reshape(x1.shape) * self.mlp(modulate(self.ln_mlp(x1), shift_mlp.reshape(x1.shape), scale_mlp.reshape(x1.shape)))
        
        return x1

class CrossDiTBlock_AllAdaLN(nn.Module):
    # n_embed: int
    num_heads: int
    q_dim: int
    kv_dim: int
    # hidden_size: int
    drop_rate: float = 0.0
    mlp_ratio: float = 4.0
    
    def setup(self):
        
        self.ln_attn_kv = nn.LayerNorm()
        self.ln_attn_q = nn.LayerNorm()
        self.ln_mlp = nn.LayerNorm()
        self.attn = nn.MultiHeadDotProductAttention(num_heads = self.num_heads, out_features = self.kv_dim)
        mlp_hidden_dim = int(self.kv_dim * self.mlp_ratio)
        self.mlp = nn.Sequential([
            nn.Dense(features = mlp_hidden_dim),
            nn.gelu,
            nn.Dense(features = self.kv_dim)]
        )
        self.adaLN_modulation_kv = nn.Sequential([nn.silu,
            nn.Dense(features = 6*self.kv_dim)
        ])
        self.adaLN_modulation_q = nn.Sequential([nn.silu,
            nn.Dense(features = 2*self.q_dim)
        ])
        
    @nn.compact
    def __call__(self, x1, x2, c1, c2, mask = None):
        # x1: q
        # x2: k,v
        # import ipdb
        # ipdb.set_trace()
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunk(self.adaLN_modulation_kv(c2), chunks=6, dim=-1)
        shift_q, scale_q = chunk(self.adaLN_modulation_q(c1), chunks=2, dim=-1)
        x2 = x2 + gate_msa.reshape(x2.shape) * self.attn(
            inputs_q = modulate(
                self.ln_attn_q(x1), 
                shift_q.reshape(x1.shape),
                scale_q.reshape(x1.shape)
                ), 
            inputs_k = modulate(
                self.ln_attn_kv(x2), 
                shift_msa.reshape(x2.shape), 
                scale_msa.reshape(x2.shape)
                ),
            inputs_v = modulate(
                self.ln_attn_kv(x2), 
                shift_msa.reshape(x2.shape), 
                scale_msa.reshape(x2.shape)
                ),
            mask = mask,)
        x2 = x2 + gate_mlp.reshape(x2.shape) * self.mlp(modulate(self.ln_mlp(x2), shift_mlp.reshape(x2.shape), scale_mlp.reshape(x2.shape)))
        
        return x2
    
class CrossDiTBlock_NoAdaLN(nn.Module):
    # n_embed: int
    num_heads: int
    dim: int
    # hidden_size: int
    drop_rate: float = 0.0
    mlp_ratio: float = 4.0
    
    def setup(self):
        
        self.ln_attn_kv = nn.LayerNorm()
        self.ln_attn_q = nn.LayerNorm()
        self.ln_mlp = nn.LayerNorm()
        self.attn = nn.MultiHeadDotProductAttention(num_heads = self.num_heads, out_features = self.dim)
        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        self.mlp = nn.Sequential([
            nn.Dense(features = mlp_hidden_dim),
            nn.gelu,
            nn.Dense(features = self.dim)]
        )
        
    @nn.compact
    def __call__(self, x1, x2, mask = None):
        # x1: q
        # x2: k,v
        # import ipdb
        # ipdb.set_trace()
        x2 = x2 + self.attn(
            inputs_q = self.ln_attn_q(x1), 
            inputs_k = self.ln_attn_kv(x2),
            inputs_v = self.ln_attn_kv(x2),
            mask = mask,)
        x2 = x2 + self.mlp(self.ln_mlp(x2))
        
        return x2

# class CrossDiTBlock(nn.Module):
#     # n_embed: int
#     num_heads: int
#     dim: int
#     # hidden_size: int
#     drop_rate: float = 0.0
#     mlp_ratio: float = 4.0
    
#     def setup(self):
        
#         self.ln_attn_1 = nn.LayerNorm()
#         self.ln_attn_2 = nn.LayerNorm()
#         self.ln_mlp = nn.LayerNorm()
#         self.attn = nn.MultiHeadDotProductAttention(num_heads = self.num_heads, out_features = self.dim)
#         mlp_hidden_dim = int(self.dim * self.mlp_ratio)
#         self.mlp = nn.Sequential([
#             nn.Dense(features = mlp_hidden_dim),
#             nn.gelu,
#             nn.Dense(features = self.dim)]
#         )
#         self.adaLN_modulation = nn.Sequential([nn.silu,
#             nn.Dense(features = 6 * self.dim)
#         ])
        
#     @nn.compact
#     def __call__(self, x1, x2, c, mask = None):
#         # x1: q
#         # x2: k,v
#         # c: c_x1, c_x2
#         # import ipdb
#         # ipdb.set_trace()
#         c_x1, c_x2 = c
#         shift_msa_1, scale_msa_1, gate_msa_1, shift_mlp_1, scale_mlp_1, gate_mlp_1 = chunk(self.adaLN_modulation(c_x1), chunks = 6, dim=-1)
#         shift_msa_2, scale_msa_2, gate_msa_2, shift_mlp_2, scale_mlp_2, gate_mlp_2 = chunk(self.adaLN_modulation(c_x2), chunks = 6, dim=-1)
#         x2 = x2 + gate_msa_2.reshape(x2.shape) * self.attn(
#             inputs_q = modulate(
#                 self.ln_attn_1(x1), 
#                 shift_msa_1.reshape(x1.shape), 
#                 scale_msa_1.reshape(x1.shape)
#                 ), 
#             inputs_k = modulate(
#                 self.ln_attn_2(x2), 
#                 shift_msa_2.reshape(x2.shape), 
#                 scale_msa_2.reshape(x2.shape)
#                 ),
#             inputs_v = modulate(
#                 self.ln_attn_2(x2), 
#                 shift_msa_2.reshape(x2.shape), 
#                 scale_msa_2.reshape(x2.shape)
#                 ),
#             mask = mask,)
#         x2 = x2 + gate_mlp_2.reshape(x2.shape) * self.mlp(modulate(self.ln_mlp(x2), shift_mlp_2.reshape(x2.shape), scale_mlp_2.reshape(x2.shape)))
        
#         return x2

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    dim: int
    # out_channels: int
    # patch_size: int

    def setup(self):
        
        self.norm_final = nn.LayerNorm(self.dim)
        # self.linear = nn.Dense(self.hidden_size, patch_size * patch_size * out_channels, use_bias=True)
        self.adaLN_modulation = nn.Sequential([nn.silu,
            nn.Dense(features = 2 * self.dim)
        ])

    @nn.compact
    def __call__(self, x, c):
        shift, scale = chunk(self.adaLN_modulation(c), chunks = 2, dim = -1)
        x = modulate(self.norm_final(x), shift, scale)
        # x = self.linear(x)
        return x

def chunk(input_array, chunks, dim=0):
    """
    在 JAX 中模拟类似 PyTorch 中 chunk 的功能。
    """
    size = input_array.shape[dim]
    chunk_size = size // chunks
    remainder = size % chunks

    chunks_sizes = [chunk_size] * chunks
    # 将余数分配给前面的 chunk
    for i in range(remainder):
        chunks_sizes[i] += 1

    # 使用数组切片生成分块
    chunks = [input_array[:,:, i * chunk_size:(i+1) * chunk_size] for i, chunk_size in enumerate(chunks_sizes)]

    return chunks

def chunk_diff_sizes(input_array, chunk_sizes):
    """
    在 JAX 中模拟类似 PyTorch 中 chunk 的功能，但允许不均分。
    """
    num_chunks = len(chunk_sizes)

    # 使用数组切片生成分块
    chunks = [input_array[..., sum(chunk_sizes[:i]):sum(chunk_sizes[:i+1])] for i in range(num_chunks)]

    return chunks

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

def ConcateSAR(state, action, reward):
    assert state.shape == action.shape == reward.shape
    concatenate_array = jnp.concatenate((state, action, reward), axis=1)
    res = jnp.zeros(concatenate_array.shape)
    for batch in range(res.shape[0]):
        for i in range(state.shape[1]):
            res = res.at[batch,3*i,:].set(state[batch][i])
        for i in range(action.shape[1]):
            res = res.at[batch,3*i+1,:].set(action[batch][i])
        for i in range(reward.shape[1]):
            res = res.at[batch,3*i+2,:].set(reward[batch][i])
    return res
