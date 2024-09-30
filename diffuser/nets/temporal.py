from functools import partial
from typing import Tuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
from einops.layers.flax import Rearrange

from diffuser.diffusion import GaussianDiffusion, ModelMeanType, _extract_into_tensor
from diffuser.dpm_solver import DPM_Solver, NoiseScheduleVP

from .helpers import Conv1dBlock, DownSample1d, TimeEmbedding, TimeStepEmbedding, StateEmbedding, ActionEmbedding, UpSample1d, mish, DiTBlock, FinalLayer, CrossDiTBlock, CrossDiTBlock_NoAdaLN, CrossDiTBlock_LNq, CrossDiTBlock_AllAdaLN, ConcateSAR, NewCrossDiTBlock, NewCrossDiTBlock_AllAdaLN
import ipdb
from .embedding import PositionalEncoding
# from ..vision_transformer.vit_jax.models_vit import MlpBlock






class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    mlp_dim: int
    out_dim: int
    dropout_rate: float = 0.1
    dtype: type = jnp.float32
   
    @nn.compact
    def __call__(self, inputs, *, deterministic = False):
        """Applies Transformer MlpBlock module."""
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        # import ipdb
        # ipdb.set_trace()
        x = nn.Dense(
            features=self.mlp_dim,
            dtype=self.dtype,)(  # pytype: disable=wrong-arg-types
                inputs)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        output = nn.Dense(
            features=actual_out_dim,
            dtype=self.dtype,)(  # pytype: disable=wrong-arg-types
                x)
        output = nn.Dropout(
            rate=self.dropout_rate)(
                output, deterministic=deterministic)
        return output

class TransformerNetNoMask(nn.Module):
    sample_dim: int
    hidden_size: int = 480
    num_heads: int = 2
    mlp_ratio: float = 4.0
    dim: int = 240
    dim_mults: Tuple[int] = (1, 4, 8)
    returns_condition: bool = False
    condition_dropout: float = 0.1
    kernel_size: int = 5

    def setup(self):
        self.norm1 = nn.LayerNorm(epsilon=1e-6)
        self.attn = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)
        self.norm2 = nn.LayerNorm(epsilon=1e-6)
        self.mlp_hidden_dim = int(self.hidden_size * self.mlp_ratio)
        self.approx_gelu = lambda: nn.gelu()
        self.mlp = MlpBlock(mlp_dim = self.mlp_hidden_dim, out_dim = 24, dropout_rate = 0)
        self.adaLN_modulation = nn.Sequential([nn.silu,
            nn.Dense(features = 6 * self.hidden_size, use_bias = True)
        ])
        self.time_mlp = TimeEmbedding(self.dim)

    @nn.compact
    def __call__(self,
        rng,
        x,
        time,
        returns: jnp.ndarray = None,
        use_dropout: bool = True,
        force_dropout: bool = False,
        gen: bool = False,):
        # num_head shouldn't be too large
        # position embedding
        t = self.time_mlp(time)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunk(self.adaLN_modulation(t), chunks = 6, dim=1)
        # import ipdb
        # ipdb.set_trace()
        x = x + gate_msa.reshape(x.shape) * self.attn(modulate(self.norm1(x), shift_msa.reshape(x.shape), scale_msa.reshape(x.shape)))

        import ipdb
        
        # ipdb.set_trace()
        
        return x


class TransformerNet(nn.Module):
    hidden_size: int = 512
    num_heads: int = 2
    mlp_ratio: float = 4.0
    dim: int = 17
    attn_size: int = 12
    causal_step: int = 1
    inv_dynamics: bool = False
    condition_dropout: float = 0.1
    returns_condition: bool = False
    observation_dim: int = 17
    action_dim: int = 6
    reward_dim: int = 1
    traj_horizon: int = 20
    
    def setup(self):
        
        self.time_mlp = TimeEmbedding(self.traj_horizon * self.dim)
        # import ipdb
        # ipdb.set_trace()
        self.DiTBlocks = [DiTBlock(num_heads=self.num_heads, dim = self.hidden_size, drop_rate=0.5, mlp_ratio=self.mlp_ratio) for _ in range(self.attn_size)]
        self.pos_embed = PositionalEncoding(d_model = self.hidden_size, max_len = 1000)
        self.final_layer = FinalLayer(dim = self.dim)
        self.input_embed = nn.Sequential([
            nn.Dense(features = self.hidden_size)]
        )
        self.output_embed = nn.Sequential([
            nn.Dense(features = self.dim)]
        )
        # self.state_mlp = StateEmbedding()
        # self.action_mlp = ActionEmbedding()
        # self.timestep_mlp = TimeStepEmbedding()
        # self.mask = jnp.tril(jnp.ones((1,1)))
    
    @nn.compact
    def __call__(self,
        rng,
        x,
        time,
        returns: jnp.ndarray = None,
        use_dropout: bool = True,
        force_dropout: bool = False,
        gen: bool = False,):
        # num_head shouldn't be too large
        # position embedding
        # import ipdb
        # ipdb.set_trace()
        embed_t = self.time_mlp(time).reshape(x.shape) # self.traj_horizon * 17
        states = x[:,:,:self.observation_dim] # 10 * self.traj_horizon * 17

        act_fn = mish
        if self.returns_condition:
            returns_mlp = nn.Sequential(
                [
                    nn.Dense(self.traj_horizon * self.dim),
                    act_fn,
                    nn.Dense(self.traj_horizon * self.dim * 4),
                    act_fn,
                    nn.Dense(self.traj_horizon * self.dim),
                ]
            )
            mask_dist = distrax.Bernoulli(probs=1 - self.condition_dropout)
            assert returns is not None
            returns = returns.reshape(-1, 1)
            returns_embed = returns_mlp(returns)
            if use_dropout:
                rng, sample_key = jax.random.split(rng)
                mask = mask_dist.sample(
                    seed=sample_key, sample_shape=(returns_embed.shape[0], 1)
                )
                returns_embed = returns_embed * mask
            if force_dropout:
                returns_embed = returns_embed * 0
            # import ipdb
            # ipdb.set_trace()

            embed_t = jnp.concatenate([embed_t, returns_embed.reshape(embed_t.shape)], axis=-1)

        '''
        # time embedding in one trajectory
        ts = jnp.arange(x.shape[1]).reshape(1,-1)
        ts = jnp.tile(ts, (x.shape[0], 1))
        ts = ts.reshape((ts.shape[0], ts.shape[1], 1))
        # [0, 1, ..., 19]
        '''

        

        '''
        embed_ts = self.timestep_mlp(ts)
        embed_s += embed_ts
        '''

        if not self.inv_dynamics:
            actions = x[:,:,self.observation_dim:]
            
        
        
        # x = jnp.concatenate([embed_s, embed_a], axis = -1)
        x = jnp.concatenate([states, actions], axis = -1) if not self.inv_dynamics else states
        x = self.input_embed(x)
        x = self.pos_embed(x)
        # TODO: every DiT block should have its own modulation
        
        batch_size, seq_len, emb_dim = x.shape

        # TODO: No mask for now
        if self.causal_step == -1:
            mask = None
        else:
            sub_diagonal = jnp.triu(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_), k=-self.causal_step)
            mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_))
            mask = jnp.logical_and(sub_diagonal, mask)
        # import ipdb
        # ipdb.set_trace()
        
        '''
        [
            s_0, a_0,
            s_1, a_1
        ]
        '''
        for block in self.DiTBlocks:
            x = block(x, embed_t, mask=mask)
        
        x = self.output_embed(x)
        x = self.final_layer(x, embed_t)
        return x

class TransformerNetRA_DIM(nn.Module):
    hidden_size: int = 512
    num_heads: int = 2
    mlp_ratio: float = 4.0
    dim: int = 17
    attn_size: int = 12
    causal_step: int = 1
    inv_dynamics: bool = False
    condition_dropout: float = 0.1
    returns_condition: bool = False
    observation_dim: int = 17
    action_dim: int = 6
    reward_dim: int = 1
    traj_horizon: int = 20
    seperate_encoding: bool = False
    embed_ratio: int = 2
    
    def setup(self):
    
        self.state_embed = nn.Sequential([
            nn.Dense(features = self.embed_ratio * self.observation_dim)]
        )
        self.action_embed = nn.Sequential([
            nn.Dense(features = self.embed_ratio * self.action_dim)]
        )
        self.reward_embed = nn.Sequential([
            nn.Dense(features = self.embed_ratio * self.reward_dim)]
        )
        
        dim = self.embed_ratio * self.observation_dim + self.embed_ratio * self.action_dim + self.embed_ratio * self.reward_dim
        self.time_mlp = TimeEmbedding(self.traj_horizon * dim)
        self.DiTBlocks = [DiTBlock(num_heads=self.num_heads, dim = dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio) for _ in range(self.attn_size)]
        self.pos_embed = PositionalEncoding(d_model = dim, max_len = 1000)
        self.final_layer = FinalLayer(dim = dim)
        
        self.state_decoder = nn.Sequential([
            nn.Dense(features = self.observation_dim)]
        )
        self.action_decoder = nn.Sequential([
            nn.Dense(features = self.action_dim)]
        )
        self.reward_decoder = nn.Sequential([
            nn.Dense(features = self.reward_dim)]
        )
    
    @nn.compact
    def __call__(self,
        rng,
        x,
        time,
        returns: jnp.ndarray = None,
        use_dropout: bool = True,
        force_dropout: bool = False,
        gen: bool = False,):
 
        
        states = x[:,:,:self.observation_dim] # 10 * self.traj_horizon * 17
        actions = x[:,:,self.observation_dim:self.observation_dim + self.action_dim]
        rewards = x[:,:,self.observation_dim + self.action_dim:]
                  
        # x = jnp.concatenate([embed_s, embed_a], axis = -1)
        embed_s = self.state_embed(states)
        embed_a = self.action_embed(actions)
        embed_r = self.reward_embed(rewards)
        x = jnp.concatenate([embed_s, embed_a, embed_r], axis = -1)
        
        embed_t = self.time_mlp(time).reshape(x.shape) # self.traj_horizon * 17
        
        x = self.pos_embed(x)
        # TODO: every DiT block should have its own modulation
        
        batch_size, seq_len, emb_dim = x.shape

        # TODO: No mask for now
        if self.causal_step == -1:
            mask = None
        else:
            sub_diagonal = jnp.triu(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_), k=-self.causal_step)
            mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_))
            mask = jnp.logical_and(sub_diagonal, mask)
        # import ipdb
        # ipdb.set_trace()
        
        for block in self.DiTBlocks:
            x = block(x, embed_t, mask=mask)
        x = self.final_layer(x, embed_t)
        
        embed_s = x[:,:,:self.embed_ratio*self.observation_dim]
        embed_a = x[:,:,self.embed_ratio*self.observation_dim:self.embed_ratio*self.observation_dim+self.embed_ratio*self.action_dim]
        embed_r = x[:,:,self.embed_ratio*self.observation_dim+self.embed_ratio*self.action_dim:]
        
        noised_s = self.state_decoder(embed_s)
        noised_a = self.action_decoder(embed_a)
        noised_r = self.reward_decoder(embed_r)
        
        noised_x = jnp.concatenate([noised_s,noised_a,noised_r], axis = -1)
        
        return noised_x

class TransformerNetRA_DIM_cross(nn.Module):
    hidden_size: int = 512
    num_heads: int = 2
    mlp_ratio: float = 4.0
    dim: int = 17
    attn_size: int = 12
    causal_step: int = 1
    inv_dynamics: bool = False
    condition_dropout: float = 0.1
    returns_condition: bool = False
    observation_dim: int = 17
    action_dim: int = 6
    reward_dim: int = 1
    traj_horizon: int = 20
    seperate_encoding: bool = False
    embed_ratio: int = 2
    
    def setup(self):
        self.s_hidden_dim = self.embed_ratio * self.observation_dim
        self.a_hidden_dim = self.embed_ratio * self.action_dim
        self.r_hidden_dim = self.embed_ratio * self.reward_dim
        self.hidden_dim = self.s_hidden_dim + self.a_hidden_dim + self.r_hidden_dim
        
        self.state_embed = nn.Sequential([
            nn.Dense(features = self.s_hidden_dim)]
        )
        self.action_embed = nn.Sequential([
            nn.Dense(features = self.a_hidden_dim)]
        )
        self.reward_embed = nn.Sequential([
            nn.Dense(features = self.r_hidden_dim )]
        )
        
        self.time_mlp = TimeEmbedding(self.traj_horizon * self.hidden_dim)
        self.DiTBlocks = [DiTBlock(num_heads=self.num_heads, dim = self.hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio) for _ in range(self.attn_size)]
        self.SA_R_CrossDiTBlock = CrossDiTBlock_LNq(num_heads=self.num_heads, dim = self.r_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio)
        self.pos_embed_sa = PositionalEncoding(d_model = self.s_hidden_dim + self.a_hidden_dim, max_len = 1000)
        self.pos_embed_r = PositionalEncoding(d_model = self.r_hidden_dim, max_len = 1000)
        self.final_layer = FinalLayer(dim = self.hidden_dim)
        
        self.state_decoder = nn.Sequential([
            nn.Dense(features = self.observation_dim)]
        )
        self.action_decoder = nn.Sequential([
            nn.Dense(features = self.action_dim)]
        )
        self.reward_decoder = nn.Sequential([
            nn.Dense(features = self.reward_dim)]
        )
    
    @nn.compact
    def __call__(self,
        rng,
        x,
        time,
        returns: jnp.ndarray = None,
        use_dropout: bool = True,
        force_dropout: bool = False,
        gen: bool = False,):
 
        
        states = x[:,:,:self.observation_dim] # 10 * self.traj_horizon * 17
        actions = x[:,:,self.observation_dim:self.observation_dim + self.action_dim]
        rewards = x[:,:,self.observation_dim + self.action_dim:]
                  
        # x = jnp.concatenate([embed_s, embed_a], axis = -1)
        embed_s = self.state_embed(states)
        embed_a = self.action_embed(actions)
        embed_r = self.reward_embed(rewards)
        embed_sa = jnp.concatenate([embed_s, embed_a], axis = -1)
        x = jnp.concatenate([embed_s, embed_a, embed_r], axis = -1)
        
        embed_t = self.time_mlp(time).reshape(x.shape) # self.traj_horizon * 17
        sa_embed_t = embed_t[:,:,:self.s_hidden_dim + self.a_hidden_dim] # 10 * self.traj_horizon * 17
        r_embed_t = embed_t[:,:,self.s_hidden_dim + self.a_hidden_dim:]
        embed_sa = self.pos_embed_sa(embed_sa)
        embed_r = self.pos_embed_r(embed_r)
        # TODO: every DiT block should have its own modulation
        
        batch_size, seq_len, emb_dim = embed_sa.shape

        # TODO: No mask for now
        if self.causal_step == -1:
            mask = None
        else:
            sub_diagonal = jnp.triu(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_), k=-self.causal_step)
            mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_))
            mask = jnp.logical_and(sub_diagonal, mask)
        # import ipdb
        # ipdb.set_trace()
        
        for block in self.DiTBlocks:
            embed_sa = block(embed_sa, sa_embed_t, mask=mask)
        embed_r = self.SA_R_CrossDiTBlock(embed_sa, embed_r, r_embed_t, mask)
        x = jnp.concatenate([embed_sa, embed_r], axis = -1)
        x = self.final_layer(x, embed_t)
        
        embed_s = x[:,:,:self.embed_ratio*self.observation_dim]
        embed_a = x[:,:,self.embed_ratio*self.observation_dim:self.embed_ratio*self.observation_dim+self.embed_ratio*self.action_dim]
        embed_r = x[:,:,self.embed_ratio*self.observation_dim+self.embed_ratio*self.action_dim:]
        
        noised_s = self.state_decoder(embed_s)
        noised_a = self.action_decoder(embed_a)
        noised_r = self.reward_decoder(embed_r)
        
        noised_x = jnp.concatenate([noised_s,noised_a,noised_r], axis = -1)
        
        return noised_x    

class TransformerNetRA_DIM_cross_alladaln(nn.Module):
    hidden_size: int = 512
    num_heads: int = 2
    mlp_ratio: float = 4.0
    dim: int = 17
    attn_size: int = 12
    causal_step: int = 1
    inv_dynamics: bool = False
    condition_dropout: float = 0.1
    returns_condition: bool = False
    observation_dim: int = 17
    action_dim: int = 6
    reward_dim: int = 1
    traj_horizon: int = 20
    seperate_encoding: bool = False
    embed_ratio: int = 2
    
    def setup(self):
        self.s_hidden_dim = self.embed_ratio * self.observation_dim
        self.a_hidden_dim = self.embed_ratio * self.action_dim
        self.r_hidden_dim = self.embed_ratio * self.reward_dim
        self.hidden_dim = self.s_hidden_dim + self.a_hidden_dim + self.r_hidden_dim
        
        self.state_embed = nn.Sequential([
            nn.Dense(features = self.s_hidden_dim)]
        )
        self.action_embed = nn.Sequential([
            nn.Dense(features = self.a_hidden_dim)]
        )
        self.reward_embed = nn.Sequential([
            nn.Dense(features = self.r_hidden_dim )]
        )
        
        self.time_mlp = TimeEmbedding(self.traj_horizon * self.hidden_dim)
        self.DiTBlocks = [DiTBlock(num_heads=self.num_heads, dim = self.s_hidden_dim + self.a_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio) for _ in range(self.attn_size)]
        self.SA_R_CrossDiTBlock = CrossDiTBlock_AllAdaLN(num_heads=self.num_heads, kv_dim = self.r_hidden_dim, q_dim = self.s_hidden_dim + self.a_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio)
        self.pos_embed_sa = PositionalEncoding(d_model = self.s_hidden_dim + self.a_hidden_dim, max_len = 1000)
        self.pos_embed_r = PositionalEncoding(d_model = self.r_hidden_dim, max_len = 1000)
        self.final_layer = FinalLayer(dim = self.hidden_dim)
        
        self.state_decoder = nn.Sequential([
            nn.Dense(features = self.observation_dim)]
        )
        self.action_decoder = nn.Sequential([
            nn.Dense(features = self.action_dim)]
        )
        self.reward_decoder = nn.Sequential([
            nn.Dense(features = self.reward_dim)]
        )
    
    @nn.compact
    def __call__(self,
        rng,
        x,
        time,
        returns: jnp.ndarray = None,
        use_dropout: bool = True,
        force_dropout: bool = False,
        gen: bool = False,):
 
        
        states = x[:,:,:self.observation_dim] # 10 * self.traj_horizon * 17
        actions = x[:,:,self.observation_dim:self.observation_dim + self.action_dim]
        rewards = x[:,:,self.observation_dim + self.action_dim:]
                  
        # x = jnp.concatenate([embed_s, embed_a], axis = -1)
        embed_s = self.state_embed(states)
        embed_a = self.action_embed(actions)
        embed_r = self.reward_embed(rewards)
        embed_sa = jnp.concatenate([embed_s, embed_a], axis = -1)
        x = jnp.concatenate([embed_s, embed_a, embed_r], axis = -1)
        
        embed_t = self.time_mlp(time).reshape(x.shape) # self.traj_horizon * 17
        sa_embed_t = embed_t[:,:,:self.s_hidden_dim + self.a_hidden_dim] # 10 * self.traj_horizon * 17
        r_embed_t = embed_t[:,:,self.s_hidden_dim + self.a_hidden_dim:]
        embed_sa = self.pos_embed_sa(embed_sa)
        embed_r = self.pos_embed_r(embed_r)
        # TODO: every DiT block should have its own modulation
        
        batch_size, seq_len, emb_dim = embed_sa.shape

        # TODO: No mask for now
        if self.causal_step == -1:
            mask = None
        else:
            sub_diagonal = jnp.triu(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_), k=-self.causal_step)
            mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_))
            mask = jnp.logical_and(sub_diagonal, mask)
        # import ipdb
        # ipdb.set_trace()
        
        for block in self.DiTBlocks:
            embed_sa = block(embed_sa, sa_embed_t, mask=mask)
        embed_r = self.SA_R_CrossDiTBlock(x1=embed_sa, x2=embed_r, c1=sa_embed_t, c2=r_embed_t, mask=mask)
        x = jnp.concatenate([embed_sa, embed_r], axis = -1)
        x = self.final_layer(x, embed_t)
        
        embed_s = x[:,:,:self.embed_ratio*self.observation_dim]
        embed_a = x[:,:,self.embed_ratio*self.observation_dim:self.embed_ratio*self.observation_dim+self.embed_ratio*self.action_dim]
        embed_r = x[:,:,self.embed_ratio*self.observation_dim+self.embed_ratio*self.action_dim:]
        
        noised_s = self.state_decoder(embed_s)
        noised_a = self.action_decoder(embed_a)
        noised_r = self.reward_decoder(embed_r)
        
        noised_x = jnp.concatenate([noised_s,noised_a,noised_r], axis = -1)
        
        return noised_x

class d(nn.Module):
    hidden_size: int = 512
    num_heads: int = 2
    mlp_ratio: float = 4.0
    dim: int = 17
    attn_size: int = 12
    causal_step: int = 1
    inv_dynamics: bool = False
    condition_dropout: float = 0.1
    returns_condition: bool = False
    observation_dim: int = 17
    action_dim: int = 6
    reward_dim: int = 1
    traj_horizon: int = 20
    seperate_encoding: bool = False
    embed_ratio: int = 2
    
    def setup(self):
        self.s_hidden_dim = self.embed_ratio * self.observation_dim
        self.a_hidden_dim = self.embed_ratio * self.action_dim
        self.r_hidden_dim = self.embed_ratio * self.reward_dim
        self.hidden_dim = self.s_hidden_dim + self.a_hidden_dim + self.r_hidden_dim
        
        self.state_embed = nn.Sequential([
            nn.Dense(features = self.s_hidden_dim)]
        )
        self.action_embed = nn.Sequential([
            nn.Dense(features = self.a_hidden_dim)]
        )
        self.reward_embed = nn.Sequential([
            nn.Dense(features = self.r_hidden_dim )]
        )
        
        self.time_mlp = TimeEmbedding(self.traj_horizon * self.hidden_dim)
        self.DiTBlocks = [DiTBlock(num_heads=self.num_heads, dim = self.hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio) for _ in range(self.attn_size)]
        self.SA_R_CrossDiTBlock = [CrossDiTBlock_LNq(num_heads=self.num_heads, dim = self.r_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio) for _ in range(self.attn_size)]
        self.pos_embed_sa = PositionalEncoding(d_model = self.s_hidden_dim + self.a_hidden_dim, max_len = 1000)
        self.pos_embed_r = PositionalEncoding(d_model = self.r_hidden_dim, max_len = 1000)
        self.final_layer = FinalLayer(dim = self.hidden_dim)
        
        self.state_decoder = nn.Sequential([
            nn.Dense(features = self.observation_dim)]
        )
        self.action_decoder = nn.Sequential([
            nn.Dense(features = self.action_dim)]
        )
        self.reward_decoder = nn.Sequential([
            nn.Dense(features = self.reward_dim)]
        )
    
    @nn.compact
    def __call__(self,
        rng,
        x,
        time,
        returns: jnp.ndarray = None,
        use_dropout: bool = True,
        force_dropout: bool = False,
        gen: bool = False,):
 
        
        states = x[:,:,:self.observation_dim] # 10 * self.traj_horizon * 17
        actions = x[:,:,self.observation_dim:self.observation_dim + self.action_dim]
        rewards = x[:,:,self.observation_dim + self.action_dim:]
                  
        # x = jnp.concatenate([embed_s, embed_a], axis = -1)
        embed_s = self.state_embed(states)
        embed_a = self.action_embed(actions)
        embed_r = self.reward_embed(rewards)
        embed_sa = jnp.concatenate([embed_s, embed_a], axis = -1)
        x = jnp.concatenate([embed_s, embed_a, embed_r], axis = -1)
        
        embed_t = self.time_mlp(time).reshape(x.shape) # self.traj_horizon * 17
        sa_embed_t = embed_t[:,:,:self.s_hidden_dim + self.a_hidden_dim] # 10 * self.traj_horizon * 17
        r_embed_t = embed_t[:,:,self.s_hidden_dim + self.a_hidden_dim:]
        embed_sa = self.pos_embed_sa(embed_sa)
        embed_r = self.pos_embed_r(embed_r)
        # TODO: every DiT block should have its own modulation
        
        batch_size, seq_len, emb_dim = embed_sa.shape

        # TODO: No mask for now
        if self.causal_step == -1:
            mask = None
        else:
            sub_diagonal = jnp.triu(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_), k=-self.causal_step)
            mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_))
            mask = jnp.logical_and(sub_diagonal, mask)
        # import ipdb
        # ipdb.set_trace()
        
        for i in range(self.N_attns):
            embed_sa = self.DiTBlocks[i](embed_sa, sa_embed_t, mask=mask)
            embed_r = self.SA_R_CrossDiTBlock[i](embed_sa, embed_r, r_embed_t, mask)
        x = jnp.concatenate([embed_sa, embed_r], axis = -1)
        x = self.final_layer(x, embed_t)
        
        embed_s = x[:,:,:self.embed_ratio*self.observation_dim]
        embed_a = x[:,:,self.embed_ratio*self.observation_dim:self.embed_ratio*self.observation_dim+self.embed_ratio*self.action_dim]
        embed_r = x[:,:,self.embed_ratio*self.observation_dim+self.embed_ratio*self.action_dim:]
        
        noised_s = self.state_decoder(embed_s)
        noised_a = self.action_decoder(embed_a)
        noised_r = self.reward_decoder(embed_r)
        
        noised_x = jnp.concatenate([noised_s,noised_a,noised_r], axis = -1)
        
        return noised_x

class TransformerNetRA_CrossDIM(nn.Module):
    hidden_size: int = 512
    num_heads: int = 2
    mlp_ratio: float = 4.0
    dim: int = 17
    attn_size: int = 12
    causal_step: int = 1
    inv_dynamics: bool = False
    condition_dropout: float = 0.1
    returns_condition: bool = False
    observation_dim: int = 17
    action_dim: int = 6
    reward_dim: int = 1
    traj_horizon: int = 20
    seperate_encoding: bool = False
    embed_ratio: int = 2
    
    def setup(self):
        self.s_hidden_dim = self.embed_ratio * self.observation_dim
        self.a_hidden_dim = self.embed_ratio * self.action_dim
        self.r_hidden_dim = self.embed_ratio * self.reward_dim
        self.hidden_dim = self.s_hidden_dim + self.a_hidden_dim + self.r_hidden_dim
        self.state_embed = nn.Sequential([
            nn.Dense(features = self.s_hidden_dim)]
        )
        self.action_embed = nn.Sequential([
            nn.Dense(features = self.a_hidden_dim)]
        )
        self.reward_embed = nn.Sequential([
            nn.Dense(features = self.r_hidden_dim)]
        )
        
        # TODO: integrated time embedding
        self.time_mlp = TimeEmbedding(self.traj_horizon * self.hidden_dim)
        # state: self-attn for temporal consistency
        self.S_DiTBlocks = [DiTBlock(num_heads=self.num_heads, dim = self.s_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio) for _ in range(self.attn_size)]
        # action: self-attn for temporal consistency
        self.A_DiTBlocks = [DiTBlock(num_heads=self.num_heads, dim = self.a_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio) for _ in range(self.attn_size)]
        # state: action cross-attends state
        self.AS_CrossDiTBlock = CrossDiTBlock(num_heads=self.num_heads, dim = self.s_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio)
        # action: state cross-attends action
        self.SA_CrossDiTBlock = CrossDiTBlock(num_heads=self.num_heads, dim = self.a_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio)
        # reward: state-action cross-attends reward
        self.SA_R_CrossDiTBlock = CrossDiTBlock(num_heads=self.num_heads, dim = self.r_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio)
        # TODO: integrated position embedding
        self.pos_embed = PositionalEncoding(d_model = self.hidden_dim, max_len = 1000)
        self.final_layer = FinalLayer(dim = self.hidden_dim)
        
        self.state_decoder = nn.Sequential([
            nn.Dense(features = self.observation_dim)]
        )
        self.action_decoder = nn.Sequential([
            nn.Dense(features = self.action_dim)]
        )
        self.reward_decoder = nn.Sequential([
            nn.Dense(features = self.reward_dim)]
        )
    
    @nn.compact
    def __call__(self,
        rng,
        x,
        time,
        returns: jnp.ndarray = None,
        use_dropout: bool = True,
        force_dropout: bool = False,
        gen: bool = False,):
        # import ipdb
        # ipdb.set_trace()
        # Separate Encoder
        states = x[:,:,:self.observation_dim] # 10 * self.traj_horizon * 17
        actions = x[:,:,self.observation_dim:self.observation_dim + self.action_dim]
        rewards = x[:,:,self.observation_dim + self.action_dim:]

        embed_s = self.state_embed(states)
        embed_a = self.action_embed(actions)
        embed_r = self.reward_embed(rewards)
        
        # Latent DiT
        latent_x = jnp.concatenate([embed_s, embed_a, embed_r], axis=-1)
        latent_x = self.pos_embed(latent_x)
        embed_s = latent_x[:,:,:self.s_hidden_dim] # 10 * self.traj_horizon * 17
        embed_a = latent_x[:,:,self.s_hidden_dim:self.s_hidden_dim + self.a_hidden_dim]
        embed_r = latent_x[:,:,self.s_hidden_dim + self.a_hidden_dim:]
        
        embed_t = self.time_mlp(time).reshape(latent_x.shape)
        s_embed_t = embed_t[:,:,:self.s_hidden_dim] # 10 * self.traj_horizon * 17
        a_embed_t = embed_t[:,:,self.s_hidden_dim:self.s_hidden_dim + self.a_hidden_dim]
        r_embed_t = embed_t[:,:,self.s_hidden_dim + self.a_hidden_dim:]
        
        _, seq_len, _ = latent_x.shape

        # TODO: No mask for now
        if self.causal_step == -1:
            mask = None
        else:
            sub_diagonal = jnp.triu(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_), k=-self.causal_step)
            mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_))
            mask = jnp.logical_and(sub_diagonal, mask)
        # import ipdb
        # ipdb.set_trace()
        
        for block in self.S_DiTBlocks:
            embed_s = block(embed_s, s_embed_t, mask)
        for block in self.A_DiTBlocks:
            embed_a = block(embed_a, a_embed_t, mask)
        
        # import ipdb
        # ipdb.set_trace()
        embed_s = self.AS_CrossDiTBlock(embed_a, embed_s, s_embed_t, mask)
        embed_a = self.SA_CrossDiTBlock(embed_s, embed_a, a_embed_t, mask)
        embed_sa = jnp.concatenate([embed_s, embed_a], axis=-1)
        embed_r = self.SA_R_CrossDiTBlock(embed_sa, embed_r, r_embed_t, mask)
        # embed_s = self.AS_CrossDiTBlock(embed_a, embed_s, (a_embed_t, s_embed_t), mask)
        # embed_a = self.SA_CrossDiTBlock(embed_s, embed_a, (s_embed_t, a_embed_t), mask)
        # embed_sa = jnp.concatenate([embed_s, embed_a], axis=-1)
        # # detach sa when diffusing reward
        # embed_r = self.SA_R_CrossDiTBlock(jax.detach(embed_sa), embed_r, (jax.detach(sa_embed_t), r_embed_t), mask)
        
        latent_x = jnp.concatenate([embed_s, embed_a, embed_r], axis=-1)
        latent_x = self.final_layer(latent_x, embed_t)
        
        embed_s = latent_x[:,:,:self.s_hidden_dim]
        embed_a = latent_x[:,:,self.s_hidden_dim:self.s_hidden_dim+self.a_hidden_dim]
        embed_r = latent_x[:,:,self.s_hidden_dim+self.a_hidden_dim:]
        
        # Separate Decoder
        noised_s = self.state_decoder(embed_s)
        noised_a = self.action_decoder(embed_a)
        noised_r = self.reward_decoder(embed_r)
        
        noised_x = jnp.concatenate([noised_s,noised_a,noised_r], axis=-1)
        
        return noised_x


class TransformerNetRA_CrossDIM_LNq(nn.Module):
    hidden_size: int = 512
    num_heads: int = 2
    mlp_ratio: float = 4.0
    dim: int = 17
    attn_size: int = 12
    causal_step: int = 1
    inv_dynamics: bool = False
    condition_dropout: float = 0.1
    returns_condition: bool = False
    observation_dim: int = 17
    action_dim: int = 6
    reward_dim: int = 1
    traj_horizon: int = 20
    seperate_encoding: bool = False
    embed_ratio: int = 2
    
    def setup(self):
        self.s_hidden_dim = self.embed_ratio * self.observation_dim
        self.a_hidden_dim = self.embed_ratio * self.action_dim
        self.r_hidden_dim = self.embed_ratio * self.reward_dim
        self.hidden_dim = self.s_hidden_dim + self.a_hidden_dim + self.r_hidden_dim
        self.state_embed = nn.Sequential([
            nn.Dense(features = self.s_hidden_dim)]
        )
        self.action_embed = nn.Sequential([
            nn.Dense(features = self.a_hidden_dim)]
        )
        self.reward_embed = nn.Sequential([
            nn.Dense(features = self.r_hidden_dim)]
        )
        
        # TODO: integrated time embedding
        self.time_mlp = TimeEmbedding(self.traj_horizon * self.hidden_dim)
        # state: self-attn for temporal consistency
        self.S_DiTBlocks = [DiTBlock(num_heads=self.num_heads, dim = self.s_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio) for _ in range(self.attn_size)]
        # action: self-attn for temporal consistency
        self.A_DiTBlocks = [DiTBlock(num_heads=self.num_heads, dim = self.a_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio) for _ in range(self.attn_size)]
        # state: action cross-attends state
        self.AS_CrossDiTBlock = CrossDiTBlock_LNq(num_heads=self.num_heads, dim = self.s_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio)
        # action: state cross-attends action
        self.SA_CrossDiTBlock = CrossDiTBlock_LNq(num_heads=self.num_heads, dim = self.a_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio)
        # reward: state-action cross-attends reward
        self.SA_R_CrossDiTBlock = CrossDiTBlock_LNq(num_heads=self.num_heads, dim = self.r_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio)
        # TODO: integrated position embedding
        self.pos_embed = PositionalEncoding(d_model = self.hidden_dim, max_len = 1000)
        self.final_layer = FinalLayer(dim = self.hidden_dim)
        
        self.state_decoder = nn.Sequential([
            nn.Dense(features = self.observation_dim)]
        )
        self.action_decoder = nn.Sequential([
            nn.Dense(features = self.action_dim)]
        )
        self.reward_decoder = nn.Sequential([
            nn.Dense(features = self.reward_dim)]
        )
    
    @nn.compact
    def __call__(self,
        rng,
        x,
        time,
        returns: jnp.ndarray = None,
        use_dropout: bool = True,
        force_dropout: bool = False,
        gen: bool = False,):
        # import ipdb
        # ipdb.set_trace()
        # Separate Encoder
        states = x[:,:,:self.observation_dim] # 10 * self.traj_horizon * 17
        actions = x[:,:,self.observation_dim:self.observation_dim + self.action_dim]
        rewards = x[:,:,self.observation_dim + self.action_dim:]

        embed_s = self.state_embed(states)
        embed_a = self.action_embed(actions)
        embed_r = self.reward_embed(rewards)
        
        # Latent DiT
        latent_x = jnp.concatenate([embed_s, embed_a, embed_r], axis=-1)
        latent_x = self.pos_embed(latent_x)
        embed_s = latent_x[:,:,:self.s_hidden_dim] # 10 * self.traj_horizon * 17
        embed_a = latent_x[:,:,self.s_hidden_dim:self.s_hidden_dim + self.a_hidden_dim]
        embed_r = latent_x[:,:,self.s_hidden_dim + self.a_hidden_dim:]
        
        embed_t = self.time_mlp(time).reshape(latent_x.shape)
        s_embed_t = embed_t[:,:,:self.s_hidden_dim] # 10 * self.traj_horizon * 17
        a_embed_t = embed_t[:,:,self.s_hidden_dim:self.s_hidden_dim + self.a_hidden_dim]
        r_embed_t = embed_t[:,:,self.s_hidden_dim + self.a_hidden_dim:]
        
        _, seq_len, _ = latent_x.shape

        # TODO: No mask for now
        if self.causal_step == -1:
            mask = None
        else:
            sub_diagonal = jnp.triu(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_), k=-self.causal_step)
            mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_))
            mask = jnp.logical_and(sub_diagonal, mask)
        # import ipdb
        # ipdb.set_trace()
        
        for block in self.S_DiTBlocks:
            embed_s = block(embed_s, s_embed_t, mask)
        for block in self.A_DiTBlocks:
            embed_a = block(embed_a, a_embed_t, mask)
        
        # import ipdb
        # ipdb.set_trace()
        embed_s = self.AS_CrossDiTBlock(embed_a, embed_s, s_embed_t, mask)
        embed_a = self.SA_CrossDiTBlock(embed_s, embed_a, a_embed_t, mask)
        embed_sa = jnp.concatenate([embed_s, embed_a], axis=-1)
        embed_r = self.SA_R_CrossDiTBlock(embed_sa, embed_r, r_embed_t, mask)
        # embed_s = self.AS_CrossDiTBlock(embed_a, embed_s, (a_embed_t, s_embed_t), mask)
        # embed_a = self.SA_CrossDiTBlock(embed_s, embed_a, (s_embed_t, a_embed_t), mask)
        # embed_sa = jnp.concatenate([embed_s, embed_a], axis=-1)
        # # detach sa when diffusing reward
        # embed_r = self.SA_R_CrossDiTBlock(jax.detach(embed_sa), embed_r, (jax.detach(sa_embed_t), r_embed_t), mask)
        
        latent_x = jnp.concatenate([embed_s, embed_a, embed_r], axis=-1)
        latent_x = self.final_layer(latent_x, embed_t)
        
        embed_s = latent_x[:,:,:self.s_hidden_dim]
        embed_a = latent_x[:,:,self.s_hidden_dim:self.s_hidden_dim+self.a_hidden_dim]
        embed_r = latent_x[:,:,self.s_hidden_dim+self.a_hidden_dim:]
        
        # Separate Decoder
        noised_s = self.state_decoder(embed_s)
        noised_a = self.action_decoder(embed_a)
        noised_r = self.reward_decoder(embed_r)
        
        noised_x = jnp.concatenate([noised_s,noised_a,noised_r], axis=-1)
        
        return noised_x


class TransformerNetRA_CrossSelfDIM_LNq(nn.Module):
    hidden_size: int = 512
    num_heads: int = 2
    mlp_ratio: float = 4.0
    dim: int = 17
    attn_size: int = 12
    causal_step: int = 1
    inv_dynamics: bool = False
    condition_dropout: float = 0.1
    returns_condition: bool = False
    observation_dim: int = 17
    action_dim: int = 6
    reward_dim: int = 1
    traj_horizon: int = 20
    seperate_encoding: bool = False
    embed_ratio: int = 2
    
    def setup(self):
        self.s_hidden_dim = self.embed_ratio * self.observation_dim
        self.a_hidden_dim = self.embed_ratio * self.action_dim
        self.r_hidden_dim = self.embed_ratio * self.reward_dim
        self.hidden_dim = self.s_hidden_dim + self.a_hidden_dim + self.r_hidden_dim
        self.state_embed = nn.Sequential([
            nn.Dense(features = self.s_hidden_dim)]
        )
        self.action_embed = nn.Sequential([
            nn.Dense(features = self.a_hidden_dim)]
        )
        self.reward_embed = nn.Sequential([
            nn.Dense(features = self.r_hidden_dim)]
        )
        
        # TODO: integrated time embedding
        self.time_mlp = TimeEmbedding(self.traj_horizon * self.hidden_dim)
        # state: self-attn for temporal consistency
        self.S_DiTBlocks = [DiTBlock(num_heads=self.num_heads, dim = self.s_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio) for _ in range(self.attn_size)]
        # action: self-attn for temporal consistency
        self.A_DiTBlocks = [DiTBlock(num_heads=self.num_heads, dim = self.a_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio) for _ in range(self.attn_size)]
        # state: action cross-attends state
        self.AS_CrossDiTBlocks = [CrossDiTBlock_LNq(num_heads=self.num_heads, dim = self.s_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio) for _ in range(self.attn_size)]
        # action: state cross-attends action
        self.SA_CrossDiTBlocks = [CrossDiTBlock_LNq(num_heads=self.num_heads, dim = self.a_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio) for _ in range(self.attn_size)]
        # reward: state-action cross-attends reward
        self.SA_R_CrossDiTBlocks = [CrossDiTBlock_LNq(num_heads=self.num_heads, dim = self.r_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio) for _ in range(self.attn_size)]
        # TODO: integrated position embedding
        self.pos_embed = PositionalEncoding(d_model = self.hidden_dim, max_len = 1000)
        self.final_layer = FinalLayer(dim = self.hidden_dim)
        
        self.state_decoder = nn.Sequential([
            nn.Dense(features = self.observation_dim)]
        )
        self.action_decoder = nn.Sequential([
            nn.Dense(features = self.action_dim)]
        )
        self.reward_decoder = nn.Sequential([
            nn.Dense(features = self.reward_dim)]
        )
    
    @nn.compact
    def __call__(self,
        rng,
        x,
        time,
        returns: jnp.ndarray = None,
        use_dropout: bool = True,
        force_dropout: bool = False,
        gen: bool = False,):
        # import ipdb
        # ipdb.set_trace()
        # Separate Encoder
        states = x[:,:,:self.observation_dim] # 10 * self.traj_horizon * 17
        actions = x[:,:,self.observation_dim:self.observation_dim + self.action_dim]
        rewards = x[:,:,self.observation_dim + self.action_dim:]

        embed_s = self.state_embed(states)
        embed_a = self.action_embed(actions)
        embed_r = self.reward_embed(rewards)
        
        # DiT
        latent_x = jnp.concatenate([embed_s, embed_a, embed_r], axis=-1)
        latent_x = self.pos_embed(latent_x)
        embed_s = latent_x[:,:,:self.s_hidden_dim] # 10 * self.traj_horizon * 17
        embed_a = latent_x[:,:,self.s_hidden_dim:self.s_hidden_dim + self.a_hidden_dim]
        embed_r = latent_x[:,:,self.s_hidden_dim + self.a_hidden_dim:]
        
        embed_t = self.time_mlp(time).reshape(latent_x.shape)
        s_embed_t = embed_t[:,:,:self.s_hidden_dim] # 10 * self.traj_horizon * 17
        a_embed_t = embed_t[:,:,self.s_hidden_dim:self.s_hidden_dim + self.a_hidden_dim]
        r_embed_t = embed_t[:,:,self.s_hidden_dim + self.a_hidden_dim:]
        
        _, seq_len, _ = latent_x.shape

        # TODO: No mask for now
        if self.causal_step == -1:
            mask = None
        else:
            sub_diagonal = jnp.triu(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_), k=-self.causal_step)
            mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_))
            mask = jnp.logical_and(sub_diagonal, mask)
        # import ipdb
        # ipdb.set_trace()
        
        for i in range(self.attn_size):
            embed_s = self.S_DiTBlocks[i](embed_s, s_embed_t, mask)
            embed_a = self.A_DiTBlocks[i](embed_a, a_embed_t, mask)
            embed_s = self.AS_CrossDiTBlocks[i](embed_a, embed_s, s_embed_t, mask)
            embed_a = self.SA_CrossDiTBlocks[i](embed_s, embed_a, a_embed_t, mask)
            embed_sa = jnp.concatenate([embed_s, embed_a], axis=-1)
            embed_r = self.SA_R_CrossDiTBlocks[i](embed_sa, embed_r, r_embed_t, mask)
        
        latent_x = jnp.concatenate([embed_s, embed_a, embed_r], axis=-1)
        latent_x = self.final_layer(latent_x, embed_t)
        
        embed_s = latent_x[:,:,:self.s_hidden_dim]
        embed_a = latent_x[:,:,self.s_hidden_dim:self.s_hidden_dim+self.a_hidden_dim]
        embed_r = latent_x[:,:,self.s_hidden_dim+self.a_hidden_dim:]
        
        # Separate Decoder
        noised_s = self.state_decoder(embed_s)
        noised_a = self.action_decoder(embed_a)
        noised_r = self.reward_decoder(embed_r)
        
        noised_x = jnp.concatenate([noised_s,noised_a,noised_r], axis=-1)
        
        return noised_x


class NewTransformerNetRA_CrossDIM_AllAdaLN(nn.Module):
    hidden_size: int = 512
    num_heads: int = 2
    mlp_ratio: float = 4.0
    dim: int = 17
    attn_size: int = 12
    causal_step: int = 1
    inv_dynamics: bool = False
    condition_dropout: float = 0.1
    returns_condition: bool = False
    observation_dim: int = 17
    action_dim: int = 6
    reward_dim: int = 1
    traj_horizon: int = 20
    seperate_encoding: bool = False
    embed_ratio: int = 2
    
    def setup(self):
        self.s_hidden_dim = self.embed_ratio * self.observation_dim
        self.a_hidden_dim = self.embed_ratio * self.action_dim
        self.r_hidden_dim = self.embed_ratio * self.reward_dim
        self.hidden_dim = self.s_hidden_dim + self.a_hidden_dim + self.r_hidden_dim
        self.state_embed = nn.Sequential([
            nn.Dense(features = self.s_hidden_dim)]
        )
        self.action_embed = nn.Sequential([
            nn.Dense(features = self.a_hidden_dim)]
        )
        self.reward_embed = nn.Sequential([
            nn.Dense(features = self.r_hidden_dim)]
        )
        
        # TODO: integrated time embedding
        self.time_mlp = TimeEmbedding(self.traj_horizon * self.hidden_dim)
        # state: self-attn for temporal consistency
        self.S_DiTBlocks = [DiTBlock(num_heads=self.num_heads, dim = self.s_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio) for _ in range(self.attn_size)]
        # action: self-attn for temporal consistency
        self.A_DiTBlocks = [DiTBlock(num_heads=self.num_heads, dim = self.a_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio) for _ in range(self.attn_size)]
        # state: action cross-attends state
        self.AS_CrossDiTBlock = NewCrossDiTBlock_AllAdaLN(num_heads=self.num_heads, q_dim=self.a_hidden_dim, kv_dim=self.s_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio)
        # action: state cross-attends action
        self.SA_CrossDiTBlock = NewCrossDiTBlock_AllAdaLN(num_heads=self.num_heads, q_dim=self.s_hidden_dim, kv_dim=self.a_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio)
        # reward: state-action cross-attends reward
        self.SA_R_CrossDiTBlock = NewCrossDiTBlock_AllAdaLN(num_heads=self.num_heads, q_dim=self.r_hidden_dim, kv_dim=self.s_hidden_dim+self.a_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio)
        # TODO: integrated position embedding
        self.pos_embed = PositionalEncoding(d_model = self.hidden_dim, max_len = 1000)
        self.final_layer = FinalLayer(dim = self.hidden_dim)
        
        self.state_decoder = nn.Sequential([
            nn.Dense(features = self.observation_dim)]
        )
        self.action_decoder = nn.Sequential([
            nn.Dense(features = self.action_dim)]
        )
        self.reward_decoder = nn.Sequential([
            nn.Dense(features = self.reward_dim)]
        )
    
    @nn.compact
    def __call__(self,
        rng,
        x,
        time,
        returns: jnp.ndarray = None,
        use_dropout: bool = True,
        force_dropout: bool = False,
        gen: bool = False,):
        # import ipdb
        # ipdb.set_trace()
        # Separate Encoder
        states = x[:,:,:self.observation_dim] # 10 * self.traj_horizon * 17
        actions = x[:,:,self.observation_dim:self.observation_dim + self.action_dim]
        rewards = x[:,:,self.observation_dim + self.action_dim:]

        embed_s = self.state_embed(states)
        embed_a = self.action_embed(actions)
        embed_r = self.reward_embed(rewards)
        
        # Latent DiT
        latent_x = jnp.concatenate([embed_s, embed_a, embed_r], axis=-1)
        latent_x = self.pos_embed(latent_x)
        embed_s = latent_x[:,:,:self.s_hidden_dim] # 10 * self.traj_horizon * 17
        embed_a = latent_x[:,:,self.s_hidden_dim:self.s_hidden_dim + self.a_hidden_dim]
        embed_r = latent_x[:,:,self.s_hidden_dim + self.a_hidden_dim:]
        
        embed_t = self.time_mlp(time).reshape(latent_x.shape)
        s_embed_t = embed_t[:,:,:self.s_hidden_dim] # 10 * self.traj_horizon * 17
        a_embed_t = embed_t[:,:,self.s_hidden_dim:self.s_hidden_dim + self.a_hidden_dim]
        r_embed_t = embed_t[:,:,self.s_hidden_dim + self.a_hidden_dim:]
        
        _, seq_len, _ = latent_x.shape

        # TODO: No mask for now
        if self.causal_step == -1:
            mask = None
        else:
            sub_diagonal = jnp.triu(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_), k=-self.causal_step)
            mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_))
            mask = jnp.logical_and(sub_diagonal, mask)
        # import ipdb
        # ipdb.set_trace()
        
        for block in self.S_DiTBlocks:
            embed_s = block(embed_s, s_embed_t, mask)
        for block in self.A_DiTBlocks:
            embed_a = block(embed_a, a_embed_t, mask)
        
        # import ipdb
        # ipdb.set_trace()
        new_embed_a = self.AS_CrossDiTBlock(embed_a, embed_s, a_embed_t, s_embed_t, mask)
        new_embed_s = self.SA_CrossDiTBlock(embed_s, embed_a, s_embed_t, a_embed_t, mask)
        embed_sa = jnp.concatenate([new_embed_s, new_embed_a], axis=-1)
        sa_embed_t = jnp.concatenate([s_embed_t, a_embed_t], axis=-1)
        # import ipdb
        # ipdb.set_trace()
        new_embed_r = self.SA_R_CrossDiTBlock(embed_r, embed_sa, r_embed_t, sa_embed_t, mask)
        
        latent_x = jnp.concatenate([new_embed_s, new_embed_a, new_embed_r], axis=-1)
        latent_x = self.final_layer(latent_x, embed_t)
        
        embed_s = latent_x[:,:,:self.s_hidden_dim]
        embed_a = latent_x[:,:,self.s_hidden_dim:self.s_hidden_dim+self.a_hidden_dim]
        embed_r = latent_x[:,:,self.s_hidden_dim+self.a_hidden_dim:]
        
        # Separate Decoder
        noised_s = self.state_decoder(embed_s)
        noised_a = self.action_decoder(embed_a)
        noised_r = self.reward_decoder(embed_r)
        
        noised_x = jnp.concatenate([noised_s,noised_a,noised_r], axis=-1)
        
        return noised_x

class TransformerNetRA_CrossDIM_AllAdaLN(nn.Module):
    hidden_size: int = 512
    num_heads: int = 2
    mlp_ratio: float = 4.0
    dim: int = 17
    attn_size: int = 12
    causal_step: int = 1
    inv_dynamics: bool = False
    condition_dropout: float = 0.1
    returns_condition: bool = False
    observation_dim: int = 17
    action_dim: int = 6
    reward_dim: int = 1
    traj_horizon: int = 20
    seperate_encoding: bool = False
    embed_ratio: int = 2
    
    def setup(self):
        self.s_hidden_dim = self.embed_ratio * self.observation_dim
        self.a_hidden_dim = self.embed_ratio * self.action_dim
        self.r_hidden_dim = self.embed_ratio * self.reward_dim
        self.hidden_dim = self.s_hidden_dim + self.a_hidden_dim + self.r_hidden_dim
        self.state_embed = nn.Sequential([
            nn.Dense(features = self.s_hidden_dim)]
        )
        self.action_embed = nn.Sequential([
            nn.Dense(features = self.a_hidden_dim)]
        )
        self.reward_embed = nn.Sequential([
            nn.Dense(features = self.r_hidden_dim)]
        )
        
        # TODO: integrated time embedding
        self.time_mlp = TimeEmbedding(self.traj_horizon * self.hidden_dim)
        # state: self-attn for temporal consistency
        self.S_DiTBlocks = [DiTBlock(num_heads=self.num_heads, dim = self.s_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio) for _ in range(self.attn_size)]
        # action: self-attn for temporal consistency
        self.A_DiTBlocks = [DiTBlock(num_heads=self.num_heads, dim = self.a_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio) for _ in range(self.attn_size)]
        # state: action cross-attends state
        self.AS_CrossDiTBlock = CrossDiTBlock_AllAdaLN(num_heads=self.num_heads, q_dim=self.a_hidden_dim, kv_dim=self.s_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio)
        # action: state cross-attends action
        self.SA_CrossDiTBlock = CrossDiTBlock_AllAdaLN(num_heads=self.num_heads, q_dim=self.s_hidden_dim, kv_dim=self.a_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio)
        # reward: state-action cross-attends reward
        self.SA_R_CrossDiTBlock = CrossDiTBlock_AllAdaLN(num_heads=self.num_heads, q_dim=self.s_hidden_dim+self.a_hidden_dim, kv_dim=self.r_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio)
        # TODO: integrated position embedding
        self.pos_embed = PositionalEncoding(d_model = self.hidden_dim, max_len = 1000)
        self.final_layer = FinalLayer(dim = self.hidden_dim)
        
        self.state_decoder = nn.Sequential([
            nn.Dense(features = self.observation_dim)]
        )
        self.action_decoder = nn.Sequential([
            nn.Dense(features = self.action_dim)]
        )
        self.reward_decoder = nn.Sequential([
            nn.Dense(features = self.reward_dim)]
        )
    
    @nn.compact
    def __call__(self,
        rng,
        x,
        time,
        returns: jnp.ndarray = None,
        use_dropout: bool = True,
        force_dropout: bool = False,
        gen: bool = False,):
        # import ipdb
        # ipdb.set_trace()
        # Separate Encoder
        states = x[:,:,:self.observation_dim] # 10 * self.traj_horizon * 17
        actions = x[:,:,self.observation_dim:self.observation_dim + self.action_dim]
        rewards = x[:,:,self.observation_dim + self.action_dim:]

        embed_s = self.state_embed(states)
        embed_a = self.action_embed(actions)
        embed_r = self.reward_embed(rewards)
        
        # Latent DiT
        latent_x = jnp.concatenate([embed_s, embed_a, embed_r], axis=-1)
        latent_x = self.pos_embed(latent_x)
        embed_s = latent_x[:,:,:self.s_hidden_dim] # 10 * self.traj_horizon * 17
        embed_a = latent_x[:,:,self.s_hidden_dim:self.s_hidden_dim + self.a_hidden_dim]
        embed_r = latent_x[:,:,self.s_hidden_dim + self.a_hidden_dim:]
        
        embed_t = self.time_mlp(time).reshape(latent_x.shape)
        s_embed_t = embed_t[:,:,:self.s_hidden_dim] # 10 * self.traj_horizon * 17
        a_embed_t = embed_t[:,:,self.s_hidden_dim:self.s_hidden_dim + self.a_hidden_dim]
        r_embed_t = embed_t[:,:,self.s_hidden_dim + self.a_hidden_dim:]
        
        _, seq_len, _ = latent_x.shape

        # TODO: No mask for now
        if self.causal_step == -1:
            mask = None
        else:
            sub_diagonal = jnp.triu(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_), k=-self.causal_step)
            mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_))
            mask = jnp.logical_and(sub_diagonal, mask)
        # import ipdb
        # ipdb.set_trace()
        
        for block in self.S_DiTBlocks:
            embed_s = block(embed_s, s_embed_t, mask)
        for block in self.A_DiTBlocks:
            embed_a = block(embed_a, a_embed_t, mask)
        
        # import ipdb
        # ipdb.set_trace()
        embed_s = self.AS_CrossDiTBlock(embed_a, embed_s, a_embed_t, s_embed_t, mask)
        embed_a = self.SA_CrossDiTBlock(embed_s, embed_a, s_embed_t, a_embed_t, mask)
        embed_sa = jnp.concatenate([embed_s, embed_a], axis=-1)
        sa_embed_t = jnp.concatenate([s_embed_t, a_embed_t], axis=-1)
        embed_r = self.SA_R_CrossDiTBlock(embed_sa, embed_r, sa_embed_t, r_embed_t, mask)
        
        latent_x = jnp.concatenate([embed_s, embed_a, embed_r], axis=-1)
        latent_x = self.final_layer(latent_x, embed_t)
        
        embed_s = latent_x[:,:,:self.s_hidden_dim]
        embed_a = latent_x[:,:,self.s_hidden_dim:self.s_hidden_dim+self.a_hidden_dim]
        embed_r = latent_x[:,:,self.s_hidden_dim+self.a_hidden_dim:]
        
        # Separate Decoder
        noised_s = self.state_decoder(embed_s)
        noised_a = self.action_decoder(embed_a)
        noised_r = self.reward_decoder(embed_r)
        
        noised_x = jnp.concatenate([noised_s,noised_a,noised_r], axis=-1)
        
        return noised_x

class TransformerNetRA_CrossDIM_woAdaLN(nn.Module):
    hidden_size: int = 512
    num_heads: int = 2
    mlp_ratio: float = 4.0
    dim: int = 17
    attn_size: int = 12
    causal_step: int = 1
    inv_dynamics: bool = False
    condition_dropout: float = 0.1
    returns_condition: bool = False
    observation_dim: int = 17
    action_dim: int = 6
    reward_dim: int = 1
    traj_horizon: int = 20
    seperate_encoding: bool = False
    embed_ratio: int = 2
    
    def setup(self):
        self.s_hidden_dim = self.embed_ratio * self.observation_dim
        self.a_hidden_dim = self.embed_ratio * self.action_dim
        self.r_hidden_dim = self.embed_ratio * self.reward_dim
        self.hidden_dim = self.s_hidden_dim + self.a_hidden_dim + self.r_hidden_dim
        self.state_embed = nn.Sequential([
            nn.Dense(features = self.s_hidden_dim)]
        )
        self.action_embed = nn.Sequential([
            nn.Dense(features = self.a_hidden_dim)]
        )
        self.reward_embed = nn.Sequential([
            nn.Dense(features = self.r_hidden_dim)]
        )
        
        # TODO: integrated time embedding
        self.time_mlp = TimeEmbedding(self.traj_horizon * self.hidden_dim)
        # state: self-attn for temporal consistency
        self.S_DiTBlocks = [DiTBlock(num_heads=self.num_heads, dim = self.s_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio) for _ in range(self.attn_size)]
        # action: self-attn for temporal consistency
        self.A_DiTBlocks = [DiTBlock(num_heads=self.num_heads, dim = self.a_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio) for _ in range(self.attn_size)]
        # state: action cross-attends state
        self.AS_CrossDiTBlock = CrossDiTBlock_NoAdaLN(num_heads=self.num_heads, dim = self.s_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio)
        # action: state cross-attends action
        self.SA_CrossDiTBlock = CrossDiTBlock_NoAdaLN(num_heads=self.num_heads, dim = self.a_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio)
        # reward: state-action cross-attends reward
        self.SA_R_CrossDiTBlock = CrossDiTBlock(num_heads=self.num_heads, dim = self.r_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio)
        # TODO: integrated position embedding
        self.pos_embed = PositionalEncoding(d_model = self.hidden_dim, max_len = 1000)
        self.final_layer = FinalLayer(dim = self.hidden_dim)
        
        self.state_decoder = nn.Sequential([
            nn.Dense(features = self.observation_dim)]
        )
        self.action_decoder = nn.Sequential([
            nn.Dense(features = self.action_dim)]
        )
        self.reward_decoder = nn.Sequential([
            nn.Dense(features = self.reward_dim)]
        )
    
    @nn.compact
    def __call__(self,
        rng,
        x,
        time,
        returns: jnp.ndarray = None,
        use_dropout: bool = True,
        force_dropout: bool = False,
        gen: bool = False,):
        # import ipdb
        # ipdb.set_trace()
        # Separate Encoder
        states = x[:,:,:self.observation_dim] # 10 * self.traj_horizon * 17
        actions = x[:,:,self.observation_dim:self.observation_dim + self.action_dim]
        rewards = x[:,:,self.observation_dim + self.action_dim:]

        embed_s = self.state_embed(states)
        embed_a = self.action_embed(actions)
        embed_r = self.reward_embed(rewards)
        
        # Latent DiT
        latent_x = jnp.concatenate([embed_s, embed_a, embed_r], axis=-1)
        latent_x = self.pos_embed(latent_x)
        embed_s = latent_x[:,:,:self.s_hidden_dim] # 10 * self.traj_horizon * 17
        embed_a = latent_x[:,:,self.s_hidden_dim:self.s_hidden_dim + self.a_hidden_dim]
        embed_r = latent_x[:,:,self.s_hidden_dim + self.a_hidden_dim:]
        
        embed_t = self.time_mlp(time).reshape(latent_x.shape)
        s_embed_t = embed_t[:,:,:self.s_hidden_dim] # 10 * self.traj_horizon * 17
        a_embed_t = embed_t[:,:,self.s_hidden_dim:self.s_hidden_dim + self.a_hidden_dim]
        r_embed_t = embed_t[:,:,self.s_hidden_dim + self.a_hidden_dim:]
        
        _, seq_len, _ = latent_x.shape

        # TODO: No mask for now
        if self.causal_step == -1:
            mask = None
        else:
            sub_diagonal = jnp.triu(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_), k=-self.causal_step)
            mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_))
            mask = jnp.logical_and(sub_diagonal, mask)
        # import ipdb
        # ipdb.set_trace()
        
        for block in self.S_DiTBlocks:
            embed_s = block(embed_s, s_embed_t, mask)
        for block in self.A_DiTBlocks:
            embed_a = block(embed_a, a_embed_t, mask)
        
        # import ipdb
        # ipdb.set_trace()
        embed_s = self.AS_CrossDiTBlock(embed_a, embed_s, mask)
        embed_a = self.SA_CrossDiTBlock(embed_s, embed_a, mask)
        embed_sa = jnp.concatenate([embed_s, embed_a], axis=-1)
        embed_r = self.SA_R_CrossDiTBlock(embed_sa, embed_r, r_embed_t, mask)
        
        latent_x = jnp.concatenate([embed_s, embed_a, embed_r], axis=-1)
        latent_x = self.final_layer(latent_x, embed_t)
        
        embed_s = latent_x[:,:,:self.s_hidden_dim]
        embed_a = latent_x[:,:,self.s_hidden_dim:self.s_hidden_dim+self.a_hidden_dim]
        embed_r = latent_x[:,:,self.s_hidden_dim+self.a_hidden_dim:]
        
        # Separate Decoder
        noised_s = self.state_decoder(embed_s)
        noised_a = self.action_decoder(embed_a)
        noised_r = self.reward_decoder(embed_r)
        
        noised_x = jnp.concatenate([noised_s,noised_a,noised_r], axis=-1)
        
        return noised_x


class TransformerNetRA_CrossDIM_SAttnrew(nn.Module):
    hidden_size: int = 512
    num_heads: int = 2
    mlp_ratio: float = 4.0
    dim: int = 17
    attn_size: int = 12
    causal_step: int = 1
    inv_dynamics: bool = False
    condition_dropout: float = 0.1
    returns_condition: bool = False
    observation_dim: int = 17
    action_dim: int = 6
    reward_dim: int = 1
    traj_horizon: int = 20
    seperate_encoding: bool = False
    embed_ratio: int = 2
    
    def setup(self):
        self.s_hidden_dim = self.embed_ratio * self.observation_dim
        self.a_hidden_dim = self.embed_ratio * self.action_dim
        self.r_hidden_dim = self.embed_ratio * self.reward_dim
        self.hidden_dim = self.s_hidden_dim + self.a_hidden_dim + self.r_hidden_dim
        self.state_embed = nn.Sequential([
            nn.Dense(features = self.s_hidden_dim)]
        )
        self.action_embed = nn.Sequential([
            nn.Dense(features = self.a_hidden_dim)]
        )
        self.reward_embed = nn.Sequential([
            nn.Dense(features = self.r_hidden_dim)]
        )
        
        # TODO: integrated time embedding
        self.time_mlp = TimeEmbedding(self.traj_horizon * self.hidden_dim)
        # state: self-attn for temporal consistency
        self.S_DiTBlocks = [DiTBlock(num_heads=self.num_heads, dim = self.s_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio) for _ in range(self.attn_size)]
        # action: self-attn for temporal consistency
        self.A_DiTBlocks = [DiTBlock(num_heads=self.num_heads, dim = self.a_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio) for _ in range(self.attn_size)]
        self.R_DiTBlock = DiTBlock(num_heads=self.num_heads, dim = self.r_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio)
        # state: action cross-attends state
        self.AS_CrossDiTBlock = CrossDiTBlock(num_heads=self.num_heads, dim = self.s_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio)
        # action: state cross-attends action
        self.SA_CrossDiTBlock = CrossDiTBlock(num_heads=self.num_heads, dim = self.a_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio)
        # reward: state-action cross-attends reward
        self.SA_R_CrossDiTBlock = CrossDiTBlock(num_heads=self.num_heads, dim = self.r_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio)
        # TODO: integrated position embedding
        self.pos_embed = PositionalEncoding(d_model = self.hidden_dim, max_len = 1000)
        self.final_layer = FinalLayer(dim = self.hidden_dim)
        
        self.state_decoder = nn.Sequential([
            nn.Dense(features = self.observation_dim)]
        )
        self.action_decoder = nn.Sequential([
            nn.Dense(features = self.action_dim)]
        )
        self.reward_decoder = nn.Sequential([
            nn.Dense(features = self.reward_dim)]
        )
    
    @nn.compact
    def __call__(self,
        rng,
        x,
        time,
        returns: jnp.ndarray = None,
        use_dropout: bool = True,
        force_dropout: bool = False,
        gen: bool = False,):
        # import ipdb
        # ipdb.set_trace()
        # Separate Encoder
        states = x[:,:,:self.observation_dim] # 10 * self.traj_horizon * 17
        actions = x[:,:,self.observation_dim:self.observation_dim + self.action_dim]
        rewards = x[:,:,self.observation_dim + self.action_dim:]

        embed_s = self.state_embed(states)
        embed_a = self.action_embed(actions)
        embed_r = self.reward_embed(rewards)
        
        # Latent DiT
        latent_x = jnp.concatenate([embed_s, embed_a, embed_r], axis=-1)
        latent_x = self.pos_embed(latent_x)
        embed_s = latent_x[:,:,:self.s_hidden_dim] # 10 * self.traj_horizon * 17
        embed_a = latent_x[:,:,self.s_hidden_dim:self.s_hidden_dim + self.a_hidden_dim]
        embed_r = latent_x[:,:,self.s_hidden_dim + self.a_hidden_dim:]
        
        embed_t = self.time_mlp(time).reshape(latent_x.shape)
        act_fn = mish
        if self.returns_condition:
            returns_mlp = nn.Sequential(
                [
                    nn.Dense(self.traj_horizon * self.hidden_dim),
                    act_fn,
                    nn.Dense(self.traj_horizon * self.hidden_dim * 4),
                    act_fn,
                    nn.Dense(self.traj_horizon * self.hidden_dim),
                ]
            )
            mask_dist = distrax.Bernoulli(probs=1 - self.condition_dropout)
            assert returns is not None
            returns = returns.reshape(-1, 1)
            returns_embed = returns_mlp(returns)
            if use_dropout:
                rng, sample_key = jax.random.split(rng)
                mask = mask_dist.sample(
                    seed=sample_key, sample_shape=(returns_embed.shape[0], 1)
                )
                returns_embed = returns_embed * mask
            if force_dropout:
                returns_embed = returns_embed * 0
            # import ipdb
            # ipdb.set_trace()

            embed_t = jnp.concatenate([embed_t, returns_embed.reshape(embed_t.shape)], axis=-1)
        s_embed_t = embed_t[:,:,:self.s_hidden_dim] # 10 * self.traj_horizon * 17
        a_embed_t = embed_t[:,:,self.s_hidden_dim:self.s_hidden_dim + self.a_hidden_dim]
        r_embed_t = embed_t[:,:,self.s_hidden_dim + self.a_hidden_dim:]
        
        _, seq_len, _ = latent_x.shape

        # TODO: No mask for now
        if self.causal_step == -1:
            mask = None
        else:
            sub_diagonal = jnp.triu(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_), k=-self.causal_step)
            mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_))
            mask = jnp.logical_and(sub_diagonal, mask)
        # import ipdb
        # ipdb.set_trace()
        
        for block in self.S_DiTBlocks:
            embed_s = block(embed_s, s_embed_t, mask)
        for block in self.A_DiTBlocks:
            embed_a = block(embed_a, a_embed_t, mask)
        embed_r = self.R_DiTBlock(embed_r, r_embed_t, mask)
        
        # import ipdb
        # ipdb.set_trace()
        embed_s = self.AS_CrossDiTBlock(embed_a, embed_s, s_embed_t, mask)
        embed_a = self.SA_CrossDiTBlock(embed_s, embed_a, a_embed_t, mask)
        embed_sa = jnp.concatenate([embed_s, embed_a], axis=-1)
        embed_r = self.SA_R_CrossDiTBlock(embed_sa, embed_r, r_embed_t, mask)
        
        latent_x = jnp.concatenate([embed_s, embed_a, embed_r], axis=-1)
        latent_x = self.final_layer(latent_x, embed_t)
        
        embed_s = latent_x[:,:,:self.s_hidden_dim]
        embed_a = latent_x[:,:,self.s_hidden_dim:self.s_hidden_dim+self.a_hidden_dim]
        embed_r = latent_x[:,:,self.s_hidden_dim+self.a_hidden_dim:]
        
        # Separate Decoder
        noised_s = self.state_decoder(embed_s)
        noised_a = self.action_decoder(embed_a)
        noised_r = self.reward_decoder(embed_r)
        
        noised_x = jnp.concatenate([noised_s,noised_a,noised_r], axis=-1)
        
        return noised_x
    

class new_arch_TransformerNetRA_CrossDIM_SAttnrew(nn.Module):
    hidden_size: int = 512
    num_heads: int = 2
    mlp_ratio: float = 4.0
    dim: int = 17
    attn_size: int = 12
    causal_step: int = 1
    inv_dynamics: bool = False
    condition_dropout: float = 0.1
    returns_condition: bool = False
    observation_dim: int = 17
    action_dim: int = 6
    reward_dim: int = 1
    traj_horizon: int = 20
    seperate_encoding: bool = False
    embed_ratio: int = 2
    
    def setup(self):
        self.s_hidden_dim = self.embed_ratio * self.observation_dim
        self.a_hidden_dim = self.embed_ratio * self.action_dim
        self.r_hidden_dim = self.embed_ratio * self.reward_dim
        self.hidden_dim = self.s_hidden_dim + self.a_hidden_dim + self.r_hidden_dim
        self.state_embed = nn.Sequential([
            nn.Dense(features = self.s_hidden_dim)]
        )
        self.action_embed = nn.Sequential([
            nn.Dense(features = self.a_hidden_dim)]
        )
        self.reward_embed = nn.Sequential([
            nn.Dense(features = self.r_hidden_dim)]
        )
        
        # TODO: integrated time embedding
        self.time_mlp = TimeEmbedding(self.traj_horizon * self.hidden_dim)
        # state: self-attn for temporal consistency
        self.S_DiTBlocks = [DiTBlock(num_heads=self.num_heads, dim = self.s_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio) for _ in range(self.attn_size)]
        # action: self-attn for temporal consistency
        self.A_DiTBlocks = [DiTBlock(num_heads=self.num_heads, dim = self.a_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio) for _ in range(self.attn_size)]
        self.R_DiTBlock = DiTBlock(num_heads=self.num_heads, dim = self.r_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio)
        # state: action cross-attends state
        self.AS_CrossDiTBlock = NewCrossDiTBlock(num_heads=self.num_heads, dim = self.s_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio)
        # action: state cross-attends action
        self.SA_CrossDiTBlock = NewCrossDiTBlock(num_heads=self.num_heads, dim = self.a_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio)
        # reward: state-action cross-attends reward
        self.SA_R_CrossDiTBlock = NewCrossDiTBlock(num_heads=self.num_heads, dim = self.r_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio)
        # TODO: integrated position embedding
        self.pos_embed = PositionalEncoding(d_model = self.hidden_dim, max_len = 1000)
        self.final_layer = FinalLayer(dim = self.hidden_dim)
        
        self.state_decoder = nn.Sequential([
            nn.Dense(features = self.observation_dim)]
        )
        self.action_decoder = nn.Sequential([
            nn.Dense(features = self.action_dim)]
        )
        self.reward_decoder = nn.Sequential([
            nn.Dense(features = self.reward_dim)]
        )
    
    @nn.compact
    def __call__(self,
        rng,
        x,
        time,
        returns: jnp.ndarray = None,
        use_dropout: bool = True,
        force_dropout: bool = False,
        gen: bool = False,):
        # import ipdb
        # ipdb.set_trace()
        # Separate Encoder
        states = x[:,:,:self.observation_dim] # 10 * self.traj_horizon * 17
        actions = x[:,:,self.observation_dim:self.observation_dim + self.action_dim]
        rewards = x[:,:,self.observation_dim + self.action_dim:]

        embed_s = self.state_embed(states)
        embed_a = self.action_embed(actions)
        embed_r = self.reward_embed(rewards)
        
        # Latent DiT
        latent_x = jnp.concatenate([embed_s, embed_a, embed_r], axis=-1)
        latent_x = self.pos_embed(latent_x)
        embed_s = latent_x[:,:,:self.s_hidden_dim] # 10 * self.traj_horizon * 17
        embed_a = latent_x[:,:,self.s_hidden_dim:self.s_hidden_dim + self.a_hidden_dim]
        embed_r = latent_x[:,:,self.s_hidden_dim + self.a_hidden_dim:]
        
        embed_t = self.time_mlp(time).reshape(latent_x.shape)
        act_fn = mish
        if self.returns_condition:
            returns_mlp = nn.Sequential(
                [
                    nn.Dense(self.traj_horizon * self.hidden_dim),
                    act_fn,
                    nn.Dense(self.traj_horizon * self.hidden_dim * 4),
                    act_fn,
                    nn.Dense(self.traj_horizon * self.hidden_dim),
                ]
            )
            mask_dist = distrax.Bernoulli(probs=1 - self.condition_dropout)
            assert returns is not None
            returns = returns.reshape(-1, 1)
            returns_embed = returns_mlp(returns)
            if use_dropout:
                rng, sample_key = jax.random.split(rng)
                mask = mask_dist.sample(
                    seed=sample_key, sample_shape=(returns_embed.shape[0], 1)
                )
                returns_embed = returns_embed * mask
            if force_dropout:
                returns_embed = returns_embed * 0
            # import ipdb
            # ipdb.set_trace()

            embed_t = jnp.concatenate([embed_t, returns_embed.reshape(embed_t.shape)], axis=-1)
        s_embed_t = embed_t[:,:,:self.s_hidden_dim] # 10 * self.traj_horizon * 17
        a_embed_t = embed_t[:,:,self.s_hidden_dim:self.s_hidden_dim + self.a_hidden_dim]
        r_embed_t = embed_t[:,:,self.s_hidden_dim + self.a_hidden_dim:]
        
        _, seq_len, _ = latent_x.shape

        # TODO: No mask for now
        if self.causal_step == -1:
            mask = None
        else:
            sub_diagonal = jnp.triu(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_), k=-self.causal_step)
            mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_))
            mask = jnp.logical_and(sub_diagonal, mask)
        # import ipdb
        # ipdb.set_trace()
        
        for block in self.S_DiTBlocks:
            embed_s = block(embed_s, s_embed_t, mask)
        for block in self.A_DiTBlocks:
            embed_a = block(embed_a, a_embed_t, mask)
        embed_r = self.R_DiTBlock(embed_r, r_embed_t, mask)
        
        # import ipdb
        # ipdb.set_trace()
        embed_s = self.AS_CrossDiTBlock(embed_s, embed_a, s_embed_t, mask)
        embed_a = self.SA_CrossDiTBlock(embed_a, embed_s, a_embed_t, mask)
        embed_sa = jnp.concatenate([embed_s, embed_a], axis=-1)
        embed_r = self.SA_R_CrossDiTBlock(embed_r, embed_sa, r_embed_t, mask)
        
        latent_x = jnp.concatenate([embed_s, embed_a, embed_r], axis=-1)
        latent_x = self.final_layer(latent_x, embed_t)
        
        embed_s = latent_x[:,:,:self.s_hidden_dim]
        embed_a = latent_x[:,:,self.s_hidden_dim:self.s_hidden_dim+self.a_hidden_dim]
        embed_r = latent_x[:,:,self.s_hidden_dim+self.a_hidden_dim:]
        
        # Separate Decoder
        noised_s = self.state_decoder(embed_s)
        noised_a = self.action_decoder(embed_a)
        noised_r = self.reward_decoder(embed_r)
        
        noised_x = jnp.concatenate([noised_s,noised_a,noised_r], axis=-1)
        
        return noised_x
    
class TransformerNetRA_CrossDIM_SAttnrew_NOSE(nn.Module):
    hidden_size: int = 512
    num_heads: int = 2
    mlp_ratio: float = 4.0
    dim: int = 17
    attn_size: int = 12
    causal_step: int = 1
    inv_dynamics: bool = False
    condition_dropout: float = 0.1
    returns_condition: bool = False
    observation_dim: int = 17
    action_dim: int = 6
    reward_dim: int = 1
    traj_horizon: int = 20
    seperate_encoding: bool = False
    embed_ratio: int = 2
    
    def setup(self):
        self.s_hidden_dim = self.embed_ratio * self.observation_dim
        self.a_hidden_dim = self.embed_ratio * self.action_dim
        self.r_hidden_dim = self.embed_ratio * self.reward_dim
        self.hidden_dim = self.s_hidden_dim + self.a_hidden_dim + self.r_hidden_dim
        self.embed = nn.Sequential([
            nn.Dense(features = self.hidden_dim)]
        )
        # self.action_embed = nn.Sequential([
        #     nn.Dense(features = self.a_hidden_dim)]
        # )
        # self.reward_embed = nn.Sequential([
        #     nn.Dense(features = self.r_hidden_dim)]
        # )
        
        # TODO: integrated time embedding
        self.time_mlp = TimeEmbedding(self.traj_horizon * self.hidden_dim)
        # state: self-attn for temporal consistency
        self.S_DiTBlocks = [DiTBlock(num_heads=self.num_heads, dim = self.s_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio) for _ in range(self.attn_size)]
        # action: self-attn for temporal consistency
        self.A_DiTBlocks = [DiTBlock(num_heads=self.num_heads, dim = self.a_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio) for _ in range(self.attn_size)]
        self.R_DiTBlock = DiTBlock(num_heads=self.num_heads, dim = self.r_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio)
        # state: action cross-attends state
        self.AS_CrossDiTBlock = CrossDiTBlock(num_heads=self.num_heads, dim = self.s_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio)
        # action: state cross-attends action
        self.SA_CrossDiTBlock = CrossDiTBlock(num_heads=self.num_heads, dim = self.a_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio)
        # reward: state-action cross-attends reward
        self.SA_R_CrossDiTBlock = CrossDiTBlock(num_heads=self.num_heads, dim = self.r_hidden_dim, drop_rate=0.5, mlp_ratio=self.mlp_ratio)
        # TODO: integrated position embedding
        self.pos_embed = PositionalEncoding(d_model = self.hidden_dim, max_len = 1000)
        self.final_layer = FinalLayer(dim = self.hidden_dim)
        
        # self.state_decoder = nn.Sequential([
        #     nn.Dense(features = self.observation_dim)]
        # )
        # self.action_decoder = nn.Sequential([
        #     nn.Dense(features = self.action_dim)]
        # )
        # self.reward_decoder = nn.Sequential([
        #     nn.Dense(features = self.reward_dim)]
        # )
        self.decoder = nn.Sequential([
            nn.Dense(features = self.observation_dim+self.action_dim+self.reward_dim)]
        )
    
    @nn.compact
    def __call__(self,
        rng,
        x,
        time,
        returns: jnp.ndarray = None,
        use_dropout: bool = True,
        force_dropout: bool = False,
        gen: bool = False,):
        # import ipdb
        # ipdb.set_trace()
        # Separate Encoder
        # states = x[:,:,:self.observation_dim] # 10 * self.traj_horizon * 17
        # actions = x[:,:,self.observation_dim:self.observation_dim + self.action_dim]
        # rewards = x[:,:,self.observation_dim + self.action_dim:]

        # embed_s = self.state_embed(states)
        # embed_a = self.action_embed(actions)
        # embed_r = self.reward_embed(rewards)
        
        # # Latent DiT
        # latent_x = jnp.concatenate([embed_s, embed_a, embed_r], axis=-1)
        latent_x = self.embed(x)
        latent_x = self.pos_embed(latent_x)
        embed_s = latent_x[:,:,:self.s_hidden_dim] # 10 * self.traj_horizon * 17
        embed_a = latent_x[:,:,self.s_hidden_dim:self.s_hidden_dim + self.a_hidden_dim]
        embed_r = latent_x[:,:,self.s_hidden_dim + self.a_hidden_dim:]
        
        embed_t = self.time_mlp(time).reshape(latent_x.shape)
        s_embed_t = embed_t[:,:,:self.s_hidden_dim] # 10 * self.traj_horizon * 17
        a_embed_t = embed_t[:,:,self.s_hidden_dim:self.s_hidden_dim + self.a_hidden_dim]
        r_embed_t = embed_t[:,:,self.s_hidden_dim + self.a_hidden_dim:]
        
        _, seq_len, _ = latent_x.shape

        # TODO: No mask for now
        if self.causal_step == -1:
            mask = None
        else:
            sub_diagonal = jnp.triu(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_), k=-self.causal_step)
            mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_))
            mask = jnp.logical_and(sub_diagonal, mask)
        # import ipdb
        # ipdb.set_trace()
        
        for block in self.S_DiTBlocks:
            embed_s = block(embed_s, s_embed_t, mask)
        for block in self.A_DiTBlocks:
            embed_a = block(embed_a, a_embed_t, mask)
        embed_r = self.R_DiTBlock(embed_r, r_embed_t, mask)
        
        # import ipdb
        # ipdb.set_trace()
        embed_s = self.AS_CrossDiTBlock(embed_a, embed_s, s_embed_t, mask)
        embed_a = self.SA_CrossDiTBlock(embed_s, embed_a, a_embed_t, mask)
        embed_sa = jnp.concatenate([embed_s, embed_a], axis=-1)
        embed_r = self.SA_R_CrossDiTBlock(embed_sa, embed_r, r_embed_t, mask)
        
        latent_x = jnp.concatenate([embed_s, embed_a, embed_r], axis=-1)
        latent_x = self.final_layer(latent_x, embed_t)
        
        # embed_s = latent_x[:,:,:self.s_hidden_dim]
        # embed_a = latent_x[:,:,self.s_hidden_dim:self.s_hidden_dim+self.a_hidden_dim]
        # embed_r = latent_x[:,:,self.s_hidden_dim+self.a_hidden_dim:]
        
        # # Separate Decoder
        # noised_s = self.state_decoder(embed_s)
        # noised_a = self.action_decoder(embed_a)
        # noised_r = self.reward_decoder(embed_r)
        
        # noised_x = jnp.concatenate([noised_s,noised_a,noised_r], axis=-1)
        noised_x = self.decoder(latent_x)
        
        return noised_x

class TransformerNetRA(nn.Module):
    hidden_size: int = 512
    num_heads: int = 2
    mlp_ratio: float = 4.0
    dim: int = 17
    attn_size: int = 12
    causal_step: int = 1
    inv_dynamics: bool = False
    condition_dropout: float = 0.1
    returns_condition: bool = False
    observation_dim: int = 17
    action_dim: int = 6
    reward_dim: int = 1
    traj_horizon: int = 20
    seperate_encoding: bool = False
    
    def setup(self):
        
        self.time_mlp = TimeEmbedding(self.traj_horizon * self.dim)
        # import ipdb
        # ipdb.set_trace()
        self.DiTBlocks = [DiTBlock(num_heads=self.num_heads, dim = self.hidden_size, drop_rate=0.5, mlp_ratio=self.mlp_ratio) for _ in range(self.attn_size)]
        self.pos_embed = PositionalEncoding(d_model = self.hidden_size, max_len = 1000)
        self.final_layer = FinalLayer(dim = self.dim)
       
        if self.seperate_encoding:
            self.state_embed = nn.Sequential([
                nn.Dense(features = self.hidden_size)]
            )
            self.action_embed = nn.Sequential([
                nn.Dense(features = self.hidden_size)]
            )
            self.reward_embed = nn.Sequential([
                nn.Dense(features = self.hidden_size)]
            )
        else:
            self.input_embed = nn.Sequential([
                nn.Dense(features = self.hidden_size)]
            )
        self.output_embed = nn.Sequential([
            nn.Dense(features = self.dim)]
        )
        # self.state_mlp = StateEmbedding()
        # self.action_mlp = ActionEmbedding()
        # self.timestep_mlp = TimeStepEmbedding()
        # self.mask = jnp.tril(jnp.ones((1,1)))
    
    @nn.compact
    def __call__(self,
        rng,
        x,
        time,
        returns: jnp.ndarray = None,
        use_dropout: bool = True,
        force_dropout: bool = False,
        gen: bool = False,):
        # num_head shouldn't be too large
        # position embedding
        # import ipdb
        # ipdb.set_trace()
        embed_t = self.time_mlp(time).reshape(x.shape) # self.traj_horizon * 17
        states = x[:,:,:self.observation_dim] # 10 * self.traj_horizon * 17
        actions = x[:,:,self.observation_dim:self.observation_dim + self.action_dim]
        rewards = x[:,:,self.observation_dim + self.action_dim:]

        act_fn = mish
        if self.returns_condition:
            returns_mlp = nn.Sequential(
                [
                    nn.Dense(self.traj_horizon * self.dim),
                    act_fn,
                    nn.Dense(self.traj_horizon * self.dim * 4),
                    act_fn,
                    nn.Dense(self.traj_horizon * self.dim),
                ]
            )
            mask_dist = distrax.Bernoulli(probs=1 - self.condition_dropout)
            assert returns is not None
            returns = returns.reshape(-1, 1)
            returns_embed = returns_mlp(returns)
            if use_dropout:
                rng, sample_key = jax.random.split(rng)
                mask = mask_dist.sample(
                    seed=sample_key, sample_shape=(returns_embed.shape[0], 1)
                )
                returns_embed = returns_embed * mask
            if force_dropout:
                returns_embed = returns_embed * 0
            # import ipdb
            # ipdb.set_trace()

            embed_t = jnp.concatenate([embed_t, returns_embed.reshape(embed_t.shape)], axis=-1)

        '''
        # time embedding in one trajectory
        ts = jnp.arange(x.shape[1]).reshape(1,-1)
        ts = jnp.tile(ts, (x.shape[0], 1))
        ts = ts.reshape((ts.shape[0], ts.shape[1], 1))
        # [0, 1, ..., 19]
        '''

        

        '''
        embed_ts = self.timestep_mlp(ts)
        embed_s += embed_ts
        '''
            
        
        # x = jnp.concatenate([embed_s, embed_a], axis = -1)
        if self.seperate_encoding:
            embed_s = self.state_embed(states)
            embed_a = self.action_embed(actions)
            embed_r = self.reward_embed(rewards)
            x = embed_s + embed_a + embed_r
        else:
            x = jnp.concatenate([states, actions, rewards], axis = -1)
            x = self.input_embed(x)
        x = self.pos_embed(x)
        # TODO: every DiT block should have its own modulation
        
        batch_size, seq_len, emb_dim = x.shape

        # TODO: No mask for now
        if self.causal_step == -1:
            mask = None
        else:
            sub_diagonal = jnp.triu(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_), k=-self.causal_step)
            mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_))
            mask = jnp.logical_and(sub_diagonal, mask)
        # import ipdb
        # ipdb.set_trace()
        
        '''
        [
            s_0, a_0,
            s_1, a_1
        ]
        '''
        for block in self.DiTBlocks:
            x = block(x, embed_t, mask=mask)
        
        x = self.output_embed(x)
        x = self.final_layer(x, embed_t)
        return x

class TransformerNetRA_SD(nn.Module):
    hidden_size: int = 512
    num_heads: int = 2
    mlp_ratio: float = 4.0
    dim: int = 17
    attn_size: int = 12
    causal_step: int = 1
    inv_dynamics: bool = False
    condition_dropout: float = 0.1
    returns_condition: bool = False
    observation_dim: int = 17
    action_dim: int = 6
    reward_dim: int = 1
    traj_horizon: int = 20
    seperate_encoding: bool = False
    
    def setup(self):
        
        self.time_mlp = TimeEmbedding(self.traj_horizon * self.dim)
        # import ipdb
        # ipdb.set_trace()
        self.DiTBlocks = [DiTBlock(num_heads=self.num_heads, dim = 3 * self.hidden_size, drop_rate=0.5, mlp_ratio=self.mlp_ratio) for _ in range(self.attn_size)]
        self.pos_embed = PositionalEncoding(d_model = 3 * self.hidden_size, max_len = 1000)
        self.final_layer = FinalLayer(dim = self.dim)
       
        if self.seperate_encoding:
            self.state_embed = nn.Sequential([
                nn.Dense(features = self.hidden_size)]
            )
            self.action_embed = nn.Sequential([
                nn.Dense(features = self.hidden_size)]
            )
            self.reward_embed = nn.Sequential([
                nn.Dense(features = self.hidden_size)]
            )
            self.state_decoder = nn.Sequential([
                nn.Dense(features = self.observation_dim)
            ])
            self.action_decoder = nn.Sequential([
                nn.Dense(features = self.action_dim)
            ])
            self.reward_decoder = nn.Sequential([
                nn.Dense(features = self.reward_dim)
            ])
        else:
            self.input_embed = nn.Sequential([
                nn.Dense(features = self.hidden_size)]
            )
        self.output_embed = nn.Sequential([
            nn.Dense(features = self.dim)]
        )
        # self.state_mlp = StateEmbedding()
        # self.action_mlp = ActionEmbedding()
        # self.timestep_mlp = TimeStepEmbedding()
        # self.mask = jnp.tril(jnp.ones((1,1)))
    
    @nn.compact
    def __call__(self,
        rng,
        x,
        time,
        returns: jnp.ndarray = None,
        use_dropout: bool = True,
        force_dropout: bool = False,
        gen: bool = False,):
        # num_head shouldn't be too large
        # position embedding
        # import ipdb
        # ipdb.set_trace()
        embed_t = self.time_mlp(time).reshape(x.shape) # self.traj_horizon * 17
        states = x[:,:,:self.observation_dim] # 10 * self.traj_horizon * 17
        actions = x[:,:,self.observation_dim:self.observation_dim + self.action_dim]
        rewards = x[:,:,self.observation_dim + self.action_dim:]

        act_fn = mish
        if self.returns_condition:
            returns_mlp = nn.Sequential(
                [
                    nn.Dense(self.traj_horizon * self.dim),
                    act_fn,
                    nn.Dense(self.traj_horizon * self.dim * 4),
                    act_fn,
                    nn.Dense(self.traj_horizon * self.dim),
                ]
            )
            mask_dist = distrax.Bernoulli(probs=1 - self.condition_dropout)
            assert returns is not None
            returns = returns.reshape(-1, 1)
            returns_embed = returns_mlp(returns)
            if use_dropout:
                rng, sample_key = jax.random.split(rng)
                mask = mask_dist.sample(
                    seed=sample_key, sample_shape=(returns_embed.shape[0], 1)
                )
                returns_embed = returns_embed * mask
            if force_dropout:
                returns_embed = returns_embed * 0
            # import ipdb
            # ipdb.set_trace()

            embed_t = jnp.concatenate([embed_t, returns_embed.reshape(embed_t.shape)], axis=-1)

        '''
        # time embedding in one trajectory
        ts = jnp.arange(x.shape[1]).reshape(1,-1)
        ts = jnp.tile(ts, (x.shape[0], 1))
        ts = ts.reshape((ts.shape[0], ts.shape[1], 1))
        # [0, 1, ..., 19]
        '''

        

        '''
        embed_ts = self.timestep_mlp(ts)
        embed_s += embed_ts
        '''
            
        
        # x = jnp.concatenate([embed_s, embed_a], axis = -1)
        if self.seperate_encoding:
            embed_s = self.state_embed(states)
            embed_a = self.action_embed(actions)
            embed_r = self.reward_embed(rewards)
            x = jnp.concatenate([embed_s, embed_a, embed_r], axis = -1)
        else:
            x = jnp.concatenate([states, actions, rewards], axis = -1)
            x = self.input_embed(x)
        x = self.pos_embed(x)
        # TODO: every DiT block should have its own modulation
        
        batch_size, seq_len, emb_dim = x.shape

        # TODO: No mask for now
        if self.causal_step == -1:
            mask = None
        else:
            sub_diagonal = jnp.triu(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_), k=-self.causal_step)
            mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_))
            mask = jnp.logical_and(sub_diagonal, mask)
        # import ipdb
        # ipdb.set_trace()
        
        '''
        [
            s_0, a_0,
            s_1, a_1
        ]
        '''
        for block in self.DiTBlocks:
            x = block(x, embed_t, mask=mask)
        
        embed_s = x[:,:,:self.hidden_size]
        embed_a = x[:,:,self.hidden_size:2*self.hidden_size]
        embed_r = x[:,:,2*self.hidden_size:]
        
        if self.seperate_encoding:
            embed_s = self.state_decoder(embed_s)
            embed_a = self.action_decoder(embed_a)
            embed_r = self.reward_decoder(embed_r)
            x = jnp.concatenate([embed_s, embed_a, embed_r], axis = -1)
            
        x = self.output_embed(x)
        # x = self.final_layer(x, embed_t)
        return x

class TransformerNetRA_SA(nn.Module):
    hidden_size: int = 512
    num_heads: int = 2
    mlp_ratio: float = 4.0
    dim: int = 17
    attn_size: int = 12
    causal_step: int = 1
    inv_dynamics: bool = False
    condition_dropout: float = 0.1
    returns_condition: bool = False
    observation_dim: int = 17
    action_dim: int = 6
    reward_dim: int = 1
    traj_horizon: int = 20
    seperate_encoding: bool = False
    
    def setup(self):
        
        self.time_mlp = TimeEmbedding(3 * self.traj_horizon * self.dim)
        # import ipdb
        # ipdb.set_trace()
        self.DiTBlocks = [DiTBlock(num_heads=self.num_heads, dim = self.hidden_size, drop_rate=0.5, mlp_ratio=self.mlp_ratio) for _ in range(self.attn_size)]
        self.pos_embed = PositionalEncoding(d_model = self.hidden_size, max_len = 1000)
        self.final_layer = FinalLayer(dim = self.dim)
       
        if self.seperate_encoding:
            self.state_embed = nn.Sequential([
                nn.Dense(features = self.hidden_size)]
            )
            self.action_embed = nn.Sequential([
                nn.Dense(features = self.hidden_size)]
            )
            self.reward_embed = nn.Sequential([
                nn.Dense(features = self.hidden_size)]
            )
            # self.state_decoder = nn.Sequential([
            #     nn.Dense(features = self.observation_dim)
            # ])
            # self.action_decoder = nn.Sequential([
            #     nn.Dense(features = self.action_dim)
            # ])
            # self.reward_decoder = nn.Sequential([
            #     nn.Dense(features = self.reward_dim)
            # ])
        else:
            self.input_embed = nn.Sequential([
                nn.Dense(features = self.hidden_size)]
            )
        self.output_embed = nn.Sequential([
            nn.Dense(features = self.dim)]
        )
        # self.state_mlp = StateEmbedding()
        # self.action_mlp = ActionEmbedding()
        # self.timestep_mlp = TimeStepEmbedding()
        # self.mask = jnp.tril(jnp.ones((1,1)))
    
    @nn.compact
    def __call__(self,
        rng,
        x,
        time,
        returns: jnp.ndarray = None,
        use_dropout: bool = True,
        force_dropout: bool = False,
        gen: bool = False,):
        
        embed_t = self.time_mlp(time).reshape((x.shape[0], -1, x.shape[-1])) # self.traj_horizon * 17
        states = x[:,:,:self.observation_dim] # 10 * self.traj_horizon * 17
        actions = x[:,:,self.observation_dim:self.observation_dim + self.action_dim]
        rewards = x[:,:,self.observation_dim + self.action_dim:]

        act_fn = mish
        if self.returns_condition:
            returns_mlp = nn.Sequential(
                [
                    nn.Dense(self.traj_horizon * self.dim),
                    act_fn,
                    nn.Dense(self.traj_horizon * self.dim * 4),
                    act_fn,
                    nn.Dense(self.traj_horizon * self.dim),
                ]
            )
            mask_dist = distrax.Bernoulli(probs=1 - self.condition_dropout)
            assert returns is not None
            returns = returns.reshape(-1, 1)
            returns_embed = returns_mlp(returns)
            if use_dropout:
                rng, sample_key = jax.random.split(rng)
                mask = mask_dist.sample(
                    seed=sample_key, sample_shape=(returns_embed.shape[0], 1)
                )
                returns_embed = returns_embed * mask
            if force_dropout:
                returns_embed = returns_embed * 0
            # import ipdb
            # ipdb.set_trace()

            embed_t = jnp.concatenate([embed_t, returns_embed.reshape(embed_t.shape)], axis=-1)

        '''
        # time embedding in one trajectory
        ts = jnp.arange(x.shape[1]).reshape(1,-1)
        ts = jnp.tile(ts, (x.shape[0], 1))
        ts = ts.reshape((ts.shape[0], ts.shape[1], 1))
        # [0, 1, ..., 19]
        '''

        

        '''
        embed_ts = self.timestep_mlp(ts)
        embed_s += embed_ts
        '''
            
        horizon = states.shape[1]
        # x = jnp.concatenate([embed_s, embed_a], axis = -1)
        if self.seperate_encoding:
            embed_s = self.state_embed(states)
            embed_a = self.action_embed(actions)
            embed_r = self.reward_embed(rewards)
            # import ipdb
            # ipdb.set_trace()
            # x = ConcateSAR(embed_s, embed_a, embed_r)
            x = jnp.concatenate([embed_s, embed_a, embed_r], axis = 1)
            # x = embed_s + embed_a + embed_r
        else:
            x = jnp.concatenate([states, actions, rewards], axis = -1)
            x = self.input_embed(x)
        x = self.pos_embed(x)
        # TODO: every DiT block should have its own modulation
        
        batch_size, seq_len, emb_dim = x.shape

        # TODO: No mask for now
        if self.causal_step == -1:
            mask = None
        else:
            sub_diagonal = jnp.triu(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_), k=-self.causal_step)
            mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_))
            mask = jnp.logical_and(sub_diagonal, mask)
        # import ipdb
        # ipdb.set_trace()
        
        '''
        [
            s_0, a_0,
            s_1, a_1
        ]
        '''
        
        for block in self.DiTBlocks:
            # import ipdb
            # ipdb.set_trace()
            x = block(x, embed_t, mask=mask)
            
            
        embed_s = x[:,:horizon,:]
        embed_a = x[:,horizon:2*horizon,:]
        embed_r = x[:,2*horizon:,:]

        # import ipdb
        # ipdb.set_trace()
        # for b in range(batch_size):
        #     for i in range(embed_s.shape[1]):
        #         embed_s = embed_s.at[b,i,:].set(x[b,3*i,:])
        #         embed_a = embed_a.at[b,i,:].set(x[b,3*i+1,:])
        #         embed_r = embed_r.at[b,i,:].set(x[b,3*i+2,:])
        
        x = embed_s + embed_a + embed_r        
        x = self.output_embed(x)
        
            
        
        # x = self.final_layer(x, embed_t)
        return x


class NewTransformerNetRA(nn.Module):
    hidden_size: int = 512
    num_heads: int = 2
    mlp_ratio: float = 4.0
    dim: int = 17
    attn_size: int = 12
    causal_step: int = 1
    inv_dynamics: bool = False
    condition_dropout: float = 0.1
    returns_condition: bool = False
    observation_dim: int = 17
    action_dim: int = 6
    reward_dim: int = 1
    traj_horizon: int = 20
    seperate_encoding: bool = False
    
    def setup(self):
        
        self.time_mlp = TimeEmbedding(3 * self.traj_horizon * self.dim)
        # import ipdb
        # ipdb.set_trace()
        self.DiTBlocks = [DiTBlock(num_heads=self.num_heads, dim = self.hidden_size, drop_rate=0.5, mlp_ratio=self.mlp_ratio) for _ in range(self.attn_size)]
        self.pos_embed = PositionalEncoding(d_model = self.hidden_size, max_len = 1000)
        self.final_layer = FinalLayer(dim = self.dim)
       
        if self.seperate_encoding:
            self.state_embed = nn.Sequential([
                nn.Dense(features = self.hidden_size)]
            )
            self.action_embed = nn.Sequential([
                nn.Dense(features = self.hidden_size)]
            )
            self.reward_embed = nn.Sequential([
                nn.Dense(features = self.hidden_size)]
            )
            self.state_decoder = nn.Sequential([
                nn.Dense(features = self.observation_dim)
            ])
            self.action_decoder = nn.Sequential([
                nn.Dense(features = self.action_dim)
            ])
            self.reward_decoder = nn.Sequential([
                nn.Dense(features = self.reward_dim)
            ])
        else:
            self.input_embed = nn.Sequential([
                nn.Dense(features = self.hidden_size)]
            )
            self.output_embed = nn.Sequential([
                nn.Dense(features = self.dim)]
            )
        # self.state_mlp = StateEmbedding()
        # self.action_mlp = ActionEmbedding()
        # self.timestep_mlp = TimeStepEmbedding()
        # self.mask = jnp.tril(jnp.ones((1,1)))
    
    @nn.compact
    def __call__(self,
        rng,
        x,
        time,
        returns: jnp.ndarray = None,
        use_dropout: bool = True,
        force_dropout: bool = False,
        gen: bool = False,):
        # num_head shouldn't be too large
        # position embedding
        # import ipdb
        # ipdb.set_trace()
        embed_t = self.time_mlp(time).reshape((x.shape[0], -1, x.shape[-1])) # self.traj_horizon * 17
        states = x[:,:,:self.observation_dim] # 10 * self.traj_horizon * 17
        actions = x[:,:,self.observation_dim:self.observation_dim + self.action_dim]
        rewards = x[:,:,self.observation_dim + self.action_dim:]

        act_fn = mish
        if self.returns_condition:
            returns_mlp = nn.Sequential(
                [
                    nn.Dense(self.traj_horizon * self.dim),
                    act_fn,
                    nn.Dense(self.traj_horizon * self.dim * 4),
                    act_fn,
                    nn.Dense(self.traj_horizon * self.dim * 3),
                ]
            )
            mask_dist = distrax.Bernoulli(probs=1 - self.condition_dropout)
            assert returns is not None
            returns = returns.reshape(-1, 1)
            returns_embed = returns_mlp(returns)
            if use_dropout:
                rng, sample_key = jax.random.split(rng)
                mask = mask_dist.sample(
                    seed=sample_key, sample_shape=(returns_embed.shape[0], 1)
                )
                returns_embed = returns_embed * mask
            if force_dropout:
                returns_embed = returns_embed * 0
            # import ipdb
            # ipdb.set_trace()

            embed_t = jnp.concatenate([embed_t, returns_embed.reshape(embed_t.shape)], axis=-1)

        '''
        # time embedding in one trajectory
        ts = jnp.arange(x.shape[1]).reshape(1,-1)
        ts = jnp.tile(ts, (x.shape[0], 1))
        ts = ts.reshape((ts.shape[0], ts.shape[1], 1))
        # [0, 1, ..., 19]
        '''

        

        '''
        embed_ts = self.timestep_mlp(ts)
        embed_s += embed_ts
        '''
            
        horizon = states.shape[1]
        # x = jnp.concatenate([embed_s, embed_a], axis = -1)
        if self.seperate_encoding:
            embed_s = self.state_embed(states)
            embed_a = self.action_embed(actions)
            embed_r = self.reward_embed(rewards)
            # import ipdb
            # ipdb.set_trace()
            # x = ConcateSAR(embed_s, embed_a, embed_r)
            x = jnp.concatenate([embed_s, embed_a, embed_r], axis = 1)
            # x = embed_s + embed_a + embed_r
        else:
            x = jnp.concatenate([states, actions, rewards], axis = -1)
            x = self.input_embed(x)
        x = self.pos_embed(x)
        # TODO: every DiT block should have its own modulation
        
        batch_size, seq_len, emb_dim = x.shape

        # TODO: No mask for now
        if self.causal_step == -1:
            mask = None
        else:
            sub_diagonal = jnp.triu(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_), k=-self.causal_step)
            mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_))
            mask = jnp.logical_and(sub_diagonal, mask)
        # import ipdb
        # ipdb.set_trace()
        
        '''
        [
            s_0, a_0,
            s_1, a_1
        ]
        '''
        
        for block in self.DiTBlocks:
            # import ipdb
            # ipdb.set_trace()
            x = block(x, embed_t, mask=mask)
            
            
        embed_s = x[:,:horizon,:]
        embed_a = x[:,horizon:2*horizon,:]
        embed_r = x[:,2*horizon:,:]

        # import ipdb
        # ipdb.set_trace()
        # for b in range(batch_size):
        #     for i in range(embed_s.shape[1]):
        #         embed_s = embed_s.at[b,i,:].set(x[b,3*i,:])
        #         embed_a = embed_a.at[b,i,:].set(x[b,3*i+1,:])
        #         embed_r = embed_r.at[b,i,:].set(x[b,3*i+2,:])
        if not self.seperate_encoding:
            x = embed_s + embed_a + embed_r        
            x = self.output_embed(x)
        else:
            embed_s = self.state_decoder(embed_s)
            embed_a = self.action_decoder(embed_a)
            embed_r = self.reward_decoder(embed_r)
            x = jnp.concatenate([embed_s, embed_a, embed_r], axis = -1)
            
        
        # x = self.final_layer(x, embed_t)
        return x

class CrossTransformerNet(nn.Module):
    hidden_size: int = 512
    num_heads: int = 2
    mlp_ratio: float = 4.0
    dim: int = 17
    # attn_size: int = 12
    causal_step: int = 1
    # inv_dynamics: bool = False
    condition_dropout: float = 0.1
    # returns_condition: bool = False
    observation_dim: int = 17
    action_dim: int = 6
    reward_dim: int = 1
    traj_horizon: int = 20
    # seperate_encoding: bool = False
    
    def setup(self):
        
        self.time_mlp = TimeEmbedding(self.traj_horizon * self.dim)
        # import ipdb
        # ipdb.set_trace()
        self.S_DiTBlock = DiTBlock(num_heads=self.num_heads, dim = self.hidden_size, drop_rate=0.5, mlp_ratio=self.mlp_ratio)
        self.A_DiTBlock = DiTBlock(num_heads=self.num_heads, dim = self.hidden_size, drop_rate=0.5, mlp_ratio=self.mlp_ratio)
        # self.R_DiTBlock = DiTBlock(num_heads=self.num_heads, dim = self.hidden_size, drop_rate=0.5, mlp_ratio=self.mlp_ratio)
        self.SA_CrossDiTBlock = CrossDiTBlock(num_heads=self.num_heads, dim = self.hidden_size, drop_rate=0.5, mlp_ratio=self.mlp_ratio)
        self.AS_CrossDiTBlock = CrossDiTBlock(num_heads=self.num_heads, dim = self.hidden_size, drop_rate=0.5, mlp_ratio=self.mlp_ratio)
        self.SA_R_CrossDiTBlock = CrossDiTBlock(num_heads=self.num_heads, dim = self.hidden_size, drop_rate=0.5, mlp_ratio=self.mlp_ratio)


        self.pos_embed = PositionalEncoding(d_model = self.hidden_size, max_len = 1000)
        self.final_layer = FinalLayer(dim = self.dim)
       
        
        self.state_embed = nn.Sequential([
            nn.Dense(features = self.hidden_size)]
        )
        self.action_embed = nn.Sequential([
            nn.Dense(features = self.hidden_size)]
        )
        self.reward_embed = nn.Sequential([
            nn.Dense(features = self.hidden_size)]
        )

        self.output_embed = nn.Sequential([
            nn.Dense(features = self.dim)]
        )
        # self.state_mlp = StateEmbedding()
        # self.action_mlp = ActionEmbedding()
        # self.timestep_mlp = TimeStepEmbedding()
        # self.mask = jnp.tril(jnp.ones((1,1)))
    
    @nn.compact
    def __call__(self,
        rng,
        x,
        time,
        returns: jnp.ndarray = None,
        use_dropout: bool = True,
        force_dropout: bool = False,
        gen: bool = False,
        ):

        embed_t = self.time_mlp(time).reshape(x.shape) # self.traj_horizon * 17
        states = x[:,:,:self.observation_dim] # 10 * self.traj_horizon * 17
        actions = x[:,:,self.observation_dim:self.observation_dim + self.action_dim]
        rewards = x[:,:,self.observation_dim + self.action_dim:]
            
        
        # x = jnp.concatenate([embed_s, embed_a], axis = -1)
        
        embed_s = self.pos_embed(self.state_embed(states))
        embed_a = self.pos_embed(self.action_embed(actions))
        embed_r = self.pos_embed(self.reward_embed(rewards))
        
        _, seq_len, _ = x.shape

        if self.causal_step == -1:
            mask = None
        else:
            # sub_diagonal = jnp.triu(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_), k=-self.causal_step)
            sub_diagonal = jnp.triu(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_))
            mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_))
            mask = jnp.logical_and(sub_diagonal, mask)

        embed_s = self.S_DiTBlock(embed_s, embed_t)
        embed_a = self.A_DiTBlock(embed_a, embed_t)
        # embed_r = self.R_DiTBlock(embed_r, embed_t)
        
        embed_s = self.AS_CrossDiTBlock(embed_a, embed_s, embed_t, mask)
        embed_a = self.SA_CrossDiTBlock(embed_s, embed_a, embed_t, mask)

        # ipdb.set_trace()
        embed_r = self.SA_R_CrossDiTBlock(embed_s + embed_a, embed_r, embed_t)

        # ipdb.set_trace()
        x = jnp.concatenate([embed_s, embed_a, embed_r], axis = -1)

        x = self.output_embed(x)
        x = self.final_layer(x, embed_t)
        return x



class CrossTransformerNet_v2(nn.Module):
    hidden_size: int = 512
    num_heads: int = 2
    mlp_ratio: float = 4.0
    dim: int = 17
    # attn_size: int = 12
    causal_step: int = 1
    # inv_dynamics: bool = False
    condition_dropout: float = 0.1
    # returns_condition: bool = False
    observation_dim: int = 17
    action_dim: int = 6
    reward_dim: int = 1
    traj_horizon: int = 20
    # seperate_encoding: bool = False
    
    def setup(self):
        
        self.time_mlp = TimeEmbedding(self.traj_horizon * self.dim)
        # import ipdb
        # ipdb.set_trace()
        self.S_DiTBlock = DiTBlock(num_heads=self.num_heads, dim = self.hidden_size, drop_rate=0.5, mlp_ratio=self.mlp_ratio)
        self.A_DiTBlock = DiTBlock(num_heads=self.num_heads, dim = self.hidden_size, drop_rate=0.5, mlp_ratio=self.mlp_ratio)
        self.R_DiTBlock = DiTBlock(num_heads=self.num_heads, dim = self.hidden_size, drop_rate=0.5, mlp_ratio=self.mlp_ratio)
        self.SA_CrossDiTBlock = CrossDiTBlock(num_heads=self.num_heads, dim = self.hidden_size, drop_rate=0.5, mlp_ratio=self.mlp_ratio)
        self.AS_CrossDiTBlock = CrossDiTBlock(num_heads=self.num_heads, dim = self.hidden_size, drop_rate=0.5, mlp_ratio=self.mlp_ratio)
        self.SA_R_CrossDiTBlock = CrossDiTBlock(num_heads=self.num_heads, dim = self.hidden_size, drop_rate=0.5, mlp_ratio=self.mlp_ratio)


        self.pos_embed = PositionalEncoding(d_model = self.hidden_size, max_len = 1000)
        self.final_layer = FinalLayer(dim = self.dim)
       
        
        self.state_embed = nn.Sequential([
            nn.Dense(features = self.hidden_size)]
        )
        self.action_embed = nn.Sequential([
            nn.Dense(features = self.hidden_size)]
        )
        self.reward_embed = nn.Sequential([
            nn.Dense(features = self.hidden_size)]
        )

        self.output_embed = nn.Sequential([
            nn.Dense(features = self.dim)]
        )
        # self.state_mlp = StateEmbedding()
        # self.action_mlp = ActionEmbedding()
        # self.timestep_mlp = TimeStepEmbedding()
        # self.mask = jnp.tril(jnp.ones((1,1)))
    
    @nn.compact
    def __call__(self,
        rng,
        x,
        time,
        returns: jnp.ndarray = None,
        use_dropout: bool = True,
        force_dropout: bool = False,
        gen: bool = False,
        ):

        embed_t = self.time_mlp(time).reshape(x.shape) # self.traj_horizon * 17
        states = x[:,:,:self.observation_dim] # 10 * self.traj_horizon * 17
        actions = x[:,:,self.observation_dim:self.observation_dim + self.action_dim]
        rewards = x[:,:,self.observation_dim + self.action_dim:]
            
        
        # x = jnp.concatenate([embed_s, embed_a], axis = -1)
        
        embed_s = self.pos_embed(self.state_embed(states))
        embed_a = self.pos_embed(self.action_embed(actions))
        embed_r = self.pos_embed(self.reward_embed(rewards))
        
        _, seq_len, _ = x.shape

        if self.causal_step == -1:
            mask = None
        else:
            # sub_diagonal = jnp.triu(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_), k=-self.causal_step)
            sub_diagonal = jnp.triu(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_))
            mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_))
            mask = jnp.logical_and(sub_diagonal, mask)

        
        embed_s = self.AS_CrossDiTBlock(embed_a, embed_s, embed_t, mask)
        embed_a = self.SA_CrossDiTBlock(embed_s, embed_a, embed_t, mask)

        # ipdb.set_trace()
        embed_r = self.SA_R_CrossDiTBlock(embed_s + embed_a, embed_r, embed_t)
        
        embed_s = self.S_DiTBlock(embed_s, embed_t)
        embed_a = self.A_DiTBlock(embed_a, embed_t)
        # embed_r = self.R_DiTBlock(embed_r, embed_t)

        # ipdb.set_trace()
        x = jnp.concatenate([embed_s, embed_a, embed_r], axis = -1)

        x = self.output_embed(x)
        x = self.final_layer(x, embed_t)
        return x


class CrossTransformerNet_v3(nn.Module):
    hidden_size: int = 512
    num_heads: int = 2
    mlp_ratio: float = 4.0
    dim: int = 17
    # attn_size: int = 12
    causal_step: int = 1
    # inv_dynamics: bool = False
    condition_dropout: float = 0.1
    # returns_condition: bool = False
    observation_dim: int = 17
    action_dim: int = 6
    reward_dim: int = 1
    traj_horizon: int = 20
    # seperate_encoding: bool = False
    
    def setup(self):
        
        self.time_mlp = TimeEmbedding(self.traj_horizon * self.dim)
        # import ipdb
        # ipdb.set_trace()
        
        self.SA_CrossDiTBlock = CrossDiTBlock(num_heads=self.num_heads, dim = self.hidden_size, drop_rate=0.5, mlp_ratio=self.mlp_ratio)
        self.AS_CrossDiTBlock = CrossDiTBlock(num_heads=self.num_heads, dim = self.hidden_size, drop_rate=0.5, mlp_ratio=self.mlp_ratio)
        self.SA_R_CrossDiTBlock = CrossDiTBlock(num_heads=self.num_heads, dim = self.hidden_size, drop_rate=0.5, mlp_ratio=self.mlp_ratio)


        self.pos_embed = PositionalEncoding(d_model = self.hidden_size, max_len = 1000)
        self.final_layer = FinalLayer(dim = self.dim)
       
        
        self.state_embed = nn.Sequential([
            nn.Dense(features = self.hidden_size)]
        )
        self.action_embed = nn.Sequential([
            nn.Dense(features = self.hidden_size)]
        )
        self.reward_embed = nn.Sequential([
            nn.Dense(features = self.hidden_size)]
        )

        self.output_embed = nn.Sequential([
            nn.Dense(features = self.dim)]
        )
        # self.state_mlp = StateEmbedding()
        # self.action_mlp = ActionEmbedding()
        # self.timestep_mlp = TimeStepEmbedding()
        # self.mask = jnp.tril(jnp.ones((1,1)))
    
    @nn.compact
    def __call__(self,
        rng,
        x,
        time,
        returns: jnp.ndarray = None,
        use_dropout: bool = True,
        force_dropout: bool = False,
        gen: bool = False,
        ):

        embed_t = self.time_mlp(time).reshape(x.shape) # self.traj_horizon * 17
        states = x[:,:,:self.observation_dim] # 10 * self.traj_horizon * 17
        actions = x[:,:,self.observation_dim:self.observation_dim + self.action_dim]
        rewards = x[:,:,self.observation_dim + self.action_dim:]
            
        
        # x = jnp.concatenate([embed_s, embed_a], axis = -1)
        
        embed_s = self.pos_embed(self.state_embed(states))
        embed_a = self.pos_embed(self.action_embed(actions))
        embed_r = self.pos_embed(self.reward_embed(rewards))
        
        _, seq_len, _ = x.shape

        if self.causal_step == -1:
            mask = None
        else:
            # sub_diagonal = jnp.triu(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_), k=-self.causal_step)
            sub_diagonal = jnp.triu(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_))
            mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_))
            mask = jnp.logical_and(sub_diagonal, mask)

        
        embed_s = self.AS_CrossDiTBlock(embed_a, embed_s, embed_t, mask)
        embed_a = self.SA_CrossDiTBlock(embed_s, embed_a, embed_t, mask)

        # ipdb.set_trace()
        embed_r = self.SA_R_CrossDiTBlock(embed_s + embed_a, embed_r, embed_t)
        
        # embed_r = self.R_DiTBlock(embed_r, embed_t)

        # ipdb.set_trace()
        x = jnp.concatenate([embed_s, embed_a, embed_r], axis = -1)

        x = self.output_embed(x)
        x = self.final_layer(x, embed_t)
        return x

class ResidualTemporalBlock(nn.Module):
    out_channels: int
    kernel_size: int
    mish: bool = True

    @nn.compact
    def __call__(self, x, t):
        if self.mish:
            act_fn = mish
        else:
            act_fn = nn.silu

        time_mlp = nn.Sequential(
            [
                act_fn,
                nn.Dense(self.out_channels),
                Rearrange("batch f -> batch 1 f"),
            ]
        )
        # ipdb.set_trace()
        out = Conv1dBlock(self.out_channels, self.kernel_size, self.mish)(x) + time_mlp(
            t
        ) # t.shape == [1, 256] ?
        out = Conv1dBlock(self.out_channels, self.kernel_size, self.mish)(out)

        if x.shape[-1] == self.out_channels:
            return out
        else:
            return out + nn.Conv(self.out_channels, (1,))(x)

# class TransformerNet(nn.Module):
#     sample_dim: int
#     dim: int = 240
#     dim_mults: Tuple[int] = (1, 4, 8)
#     returns_condition: bool = False
#     condition_dropout: float = 0.1
#     kernel_size: int = 5

#     def setup(self):
#         self.dims = dims = [
#             self.sample_dim,
#             *map(lambda m: self.dim * m, self.dim_mults),
#         ]
#         self.in_out = list(zip(dims[:-1], dims[1:]))
#         self.attention = AttentionBlock()
#         self.time_mlp = TimeEmbedding(self.dim)
#         print(f"[ diffuser/nets/temporal.py ] Channel dimensions: {self.in_out}")
    
#     @nn.compact
#     def __call__(
#         self,
#         rng,
#         x,
#         time,
#         returns: jnp.ndarray = None,
#         use_dropout: bool = True,
#         force_dropout: bool = False,
#         gen: bool = False,
#     ):
#         # ipdb.set_trace()
#         t = self.time_mlp(time).reshape(x.shape)
#         x = self.attention(x)
#         x = t * x
#         x = nn.Sequential(
#             [
#                 Conv1dBlock(self.dim, kernel_size=self.kernel_size, mish=True),
#                 nn.Conv(self.sample_dim, (1,)),
#             ]
#         )(x)

#         return x
        

class TemporalUnet(nn.Module):
    hidden_size: int = 512
    num_heads: int = 2
    mlp_ratio: float = 4.0
    dim: int = 17
    attn_size: int = 12
    causal_step: int = 1
    inv_dynamics: bool = False
    condition_dropout: float = 0.1
    returns_condition: bool = False
    observation_dim: int = 17
    action_dim: int = 6
    reward_dim: int = 1
    traj_horizon: int = 20
    seperate_encoding: bool = False
    embed_ratio: int = 2
    dim_mults: Tuple[int] = (1, 2, 4)
    kernel_size: int = 5
    

    def setup(self):
        self.dims = dims = [
            self.dim,
            *map(lambda m: self.hidden_size * m, self.dim_mults),
        ]
        self.in_out = list(zip(dims[:-1], dims[1:]))
        print(f"[ diffuser/nets/temporal.py ] Channel dimensions: {self.in_out}")

    @nn.compact
    def __call__(
        self,
        rng,
        x,
        time,
        returns: jnp.ndarray = None,
        use_dropout: bool = True,
        force_dropout: bool = False,
        gen: bool = False,
    ):
        act_fn = mish
        # ipdb.set_trace()
        time_mlp = TimeEmbedding(self.dim)
        if self.returns_condition:
            returns_mlp = nn.Sequential(
                [
                    nn.Dense(self.dim),
                    act_fn,
                    nn.Dense(self.dim * 4),
                    act_fn,
                    nn.Dense(self.dim),
                ]
            )
            mask_dist = distrax.Bernoulli(probs=1 - self.condition_dropout)

        t = time_mlp(time)
        if self.returns_condition:
            assert returns is not None
            returns = returns.reshape(-1, 1)
            returns_embed = returns_mlp(returns)
            if use_dropout:
                rng, sample_key = jax.random.split(rng)
                mask = mask_dist.sample(
                    seed=sample_key, sample_shape=(returns_embed.shape[0], 1)
                )
                returns_embed = returns_embed * mask
            if force_dropout:
                returns_embed = returns_embed * 0
            t = jnp.concatenate([t, returns_embed], axis=-1)

        h = []
        num_resolutions = len(self.in_out)
        for ind, (_, dim_out) in enumerate(self.in_out):
            is_last = ind >= (num_resolutions - 1)

            x = ResidualTemporalBlock(
                dim_out,
                kernel_size=self.kernel_size,
                mish=True,
            )(x, t)
            # x = AttentionBlock()(x)
            x = ResidualTemporalBlock(
                dim_out,
                kernel_size=self.kernel_size,
                mish=True,
            )(x, t)
            # x = AttentionBlock()(x)
            h.append(x)

            if not is_last:
                x = DownSample1d(dim_out)(x)

        mid_dim = self.dims[-1]
        # ipdb.set_trace()
        x = ResidualTemporalBlock(
            mid_dim,
            kernel_size=self.kernel_size,
            mish=True,
        )(x, t)
        # x = AttentionBlock()(x)
        x = ResidualTemporalBlock(
            mid_dim,
            kernel_size=self.kernel_size,
            mish=True,
        )(x, t)
        # x = AttentionBlock()(x)
        for ind, (dim_in, _) in enumerate(reversed(self.in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            x = jnp.concatenate([x, h.pop()], axis=-1)
            x = ResidualTemporalBlock(
                dim_in,
                kernel_size=self.kernel_size,
                mish=True,
            )(x, t)
            # x = AttentionBlock()(x)
            x = ResidualTemporalBlock(
                dim_in,
                kernel_size=self.kernel_size,
                mish=True,
            )(x, t)
            # x = AttentionBlock()(x)

            if not is_last:
                x = UpSample1d(dim_in)(x)

        x = nn.Sequential(
            [
                Conv1dBlock(self.dim, kernel_size=self.kernel_size, mish=True),
                nn.Conv(self.dim, (1,)),
            ]
        )(x)

        return x


class DiffusionPlanner(nn.Module):
    cfgs: dict
    diffusion: GaussianDiffusion
    sample_dim: int
    observation_dim: int
    action_dim: int
    reward_dim: int
    horizon: int
    dim: int
    dim_mults: Tuple[int]
    returns_condition: bool = False
    condition_dropout: float = 0.25
    kernel_size: int = 5
    sample_method: str = "ddpm"
    dpm_steps: int = 15
    dpm_t_end: float = 0.001
    inv_dynamics: bool = False
    reward_dynamics: bool = True
    use_condition: bool = False
    use_goal_condition: bool = False
    seperate_encoding: bool = False

    def setup(self):
        # self.base_net = TransformerNet(
        #     sample_dim=self.sample_dim,
        #     dim=360,
        #     dim_mults=self.dim_mults,
        #     returns_condition=self.returns_condition,
        #     condition_dropout=self.condition_dropout,
        #     kernel_size=self.kernel_size,
        # )
        if self.cfgs.algo_cfg.edit_sar:
            # import ipdb
            # ipdb.set_trace()
            if self.cfgs.algo_cfg.use_cross:
                if self.cfgs.algo_cfg.version == 'v1':
                    self.base_net = CrossTransformerNet(
                    hidden_size = self.cfgs['hidden_size'].value,
                    num_heads = self.cfgs['num_heads'].value,
                    dim = self.sample_dim,
                    causal_step=self.cfgs['causal_step'].value,
                    condition_dropout = self.condition_dropout,
                    observation_dim=self.observation_dim,
                    action_dim=self.action_dim,
                    traj_horizon=self.horizon,  
                    )
                elif self.cfgs.algo_cfg.version == 'v2':
                    self.base_net = CrossTransformerNet_v2(
                    hidden_size = self.cfgs['hidden_size'].value,
                    num_heads = self.cfgs['num_heads'].value,
                    dim = self.sample_dim,
                    causal_step=self.cfgs['causal_step'].value,
                    condition_dropout = self.condition_dropout,
                    observation_dim=self.observation_dim,
                    action_dim=self.action_dim,
                    traj_horizon=self.horizon,  
                    )
                elif self.cfgs.algo_cfg.version == 'v3':
                    self.base_net = CrossTransformerNet_v3(
                    hidden_size = self.cfgs['hidden_size'].value,
                    num_heads = self.cfgs['num_heads'].value,
                    dim = self.sample_dim,
                    causal_step=self.cfgs['causal_step'].value,
                    condition_dropout = self.condition_dropout,
                    observation_dim=self.observation_dim,
                    action_dim=self.action_dim,
                    traj_horizon=self.horizon,  
                    )
                elif self.cfgs.algo_cfg.version == 'v4':
                    self.base_net = TransformerNetRA_CrossDIM(
                        hidden_size = self.cfgs['hidden_size'].value,
                        dim = self.sample_dim,
                        observation_dim=self.observation_dim,
                        action_dim=self.action_dim,
                        traj_horizon=self.horizon,
                        num_heads = self.cfgs['num_heads'].value,
                        attn_size = self.cfgs['N_attns'].value,
                        causal_step=self.cfgs['causal_step'].value,
                        inv_dynamics = self.inv_dynamics,
                        returns_condition=self.returns_condition,
                        condition_dropout = self.condition_dropout,
                        seperate_encoding = self.seperate_encoding,
                        embed_ratio = self.cfgs['embed_ratio'].value,
                    )
                elif self.cfgs.algo_cfg.version == 'v5':
                    self.base_net = TransformerNetRA_CrossDIM_woAdaLN(
                        hidden_size = self.cfgs['hidden_size'].value,
                        dim = self.sample_dim,
                        observation_dim=self.observation_dim,
                        action_dim=self.action_dim,
                        traj_horizon=self.horizon,
                        num_heads = self.cfgs['num_heads'].value,
                        attn_size = self.cfgs['N_attns'].value,
                        causal_step=self.cfgs['causal_step'].value,
                        inv_dynamics = self.inv_dynamics,
                        returns_condition=self.returns_condition,
                        condition_dropout = self.condition_dropout,
                        seperate_encoding = self.seperate_encoding,
                        embed_ratio = self.cfgs['embed_ratio'].value,
                    )
                elif self.cfgs.algo_cfg.version == 'v6':
                    self.base_net = TransformerNetRA_CrossDIM_LNq(
                        hidden_size = self.cfgs['hidden_size'].value,
                        dim = self.sample_dim,
                        observation_dim=self.observation_dim,
                        action_dim=self.action_dim,
                        traj_horizon=self.horizon,
                        num_heads = self.cfgs['num_heads'].value,
                        attn_size = self.cfgs['N_attns'].value,
                        causal_step=self.cfgs['causal_step'].value,
                        inv_dynamics = self.inv_dynamics,
                        returns_condition=self.returns_condition,
                        condition_dropout = self.condition_dropout,
                        seperate_encoding = self.seperate_encoding,
                        embed_ratio = self.cfgs['embed_ratio'].value,
                    )
                elif self.cfgs.algo_cfg.version == 'v7':
                    self.base_net = TransformerNetRA_CrossDIM_SAttnrew(
                        hidden_size = self.cfgs['hidden_size'].value,
                        dim = self.sample_dim,
                        observation_dim=self.observation_dim,
                        action_dim=self.action_dim,
                        traj_horizon=self.horizon,
                        num_heads = self.cfgs['num_heads'].value,
                        attn_size = self.cfgs['N_attns'].value,
                        causal_step=self.cfgs['causal_step'].value,
                        inv_dynamics = self.inv_dynamics,
                        returns_condition=self.returns_condition,
                        condition_dropout = self.condition_dropout,
                        seperate_encoding = self.seperate_encoding,
                        embed_ratio = self.cfgs['embed_ratio'].value,
                    )
                elif self.cfgs.algo_cfg.version == 'v8':
                    self.base_net = TransformerNetRA_CrossSelfDIM_LNq(
                        hidden_size = self.cfgs['hidden_size'].value,
                        dim = self.sample_dim,
                        observation_dim=self.observation_dim,
                        action_dim=self.action_dim,
                        traj_horizon=self.horizon,
                        num_heads = self.cfgs['num_heads'].value,
                        attn_size = self.cfgs['N_attns'].value,
                        causal_step=self.cfgs['causal_step'].value,
                        inv_dynamics = self.inv_dynamics,
                        returns_condition=self.returns_condition,
                        condition_dropout = self.condition_dropout,
                        seperate_encoding = self.seperate_encoding,
                        embed_ratio = self.cfgs['embed_ratio'].value,
                    )
                elif self.cfgs.algo_cfg.version == 'v9':
                    self.base_net = TransformerNetRA_CrossDIM_AllAdaLN(
                        hidden_size = self.cfgs['hidden_size'].value,
                        dim = self.sample_dim,
                        observation_dim=self.observation_dim,
                        action_dim=self.action_dim,
                        traj_horizon=self.horizon,
                        num_heads = self.cfgs['num_heads'].value,
                        attn_size = self.cfgs['N_attns'].value,
                        causal_step=self.cfgs['causal_step'].value,
                        inv_dynamics = self.inv_dynamics,
                        returns_condition=self.returns_condition,
                        condition_dropout = self.condition_dropout,
                        seperate_encoding = self.seperate_encoding,
                        embed_ratio = self.cfgs['embed_ratio'].value,
                    )
                elif self.cfgs.algo_cfg.version == 'v10':
                    self.base_net = TransformerNetRA_CrossDIM_SAttnrew_NOSE(
                        hidden_size = self.cfgs['hidden_size'].value,
                        dim = self.sample_dim,
                        observation_dim=self.observation_dim,
                        action_dim=self.action_dim,
                        traj_horizon=self.horizon,
                        num_heads = self.cfgs['num_heads'].value,
                        attn_size = self.cfgs['N_attns'].value,
                        causal_step=self.cfgs['causal_step'].value,
                        inv_dynamics = self.inv_dynamics,
                        returns_condition=self.returns_condition,
                        condition_dropout = self.condition_dropout,
                        seperate_encoding = self.seperate_encoding,
                        embed_ratio = self.cfgs['embed_ratio'].value,
                    )
                elif self.cfgs.algo_cfg.version == 'v11':
                    self.base_net = new_arch_TransformerNetRA_CrossDIM_SAttnrew(
                        hidden_size = self.cfgs['hidden_size'].value,
                        dim = self.sample_dim,
                        observation_dim=self.observation_dim,
                        action_dim=self.action_dim,
                        traj_horizon=self.horizon,
                        num_heads = self.cfgs['num_heads'].value,
                        attn_size = self.cfgs['N_attns'].value,
                        causal_step=self.cfgs['causal_step'].value,
                        inv_dynamics = self.inv_dynamics,
                        returns_condition=self.returns_condition,
                        condition_dropout = self.condition_dropout,
                        seperate_encoding = self.seperate_encoding,
                        embed_ratio = self.cfgs['embed_ratio'].value,
                    )
                elif self.cfgs.algo_cfg.version == 'v12':
                    self.base_net = NewTransformerNetRA_CrossDIM_AllAdaLN(
                        hidden_size = self.cfgs['hidden_size'].value,
                        dim = self.sample_dim,
                        observation_dim=self.observation_dim,
                        action_dim=self.action_dim,
                        traj_horizon=self.horizon,
                        num_heads = self.cfgs['num_heads'].value,
                        attn_size = self.cfgs['N_attns'].value,
                        causal_step=self.cfgs['causal_step'].value,
                        inv_dynamics = self.inv_dynamics,
                        returns_condition=self.returns_condition,
                        condition_dropout = self.condition_dropout,
                        seperate_encoding = self.seperate_encoding,
                        embed_ratio = self.cfgs['embed_ratio'].value,
                    )
            else:
                if self.cfgs.algo_cfg.version == 'v0':
                    self.base_net = TransformerNetRA(
                        hidden_size = self.cfgs['hidden_size'].value,
                        dim = self.sample_dim,
                        observation_dim=self.observation_dim,
                        action_dim=self.action_dim,
                        traj_horizon=self.horizon,
                        num_heads = self.cfgs['num_heads'].value,
                        attn_size = self.cfgs['N_attns'].value,
                        causal_step=self.cfgs['causal_step'].value,
                        inv_dynamics = self.inv_dynamics,
                        returns_condition=self.returns_condition,
                        condition_dropout = self.condition_dropout,
                        seperate_encoding = self.seperate_encoding,
                    )
                elif self.cfgs.algo_cfg.version == 'v1':
                    self.base_net = NewTransformerNetRA(
                        hidden_size = self.cfgs['hidden_size'].value,
                        dim = self.sample_dim,
                        observation_dim=self.observation_dim,
                        action_dim=self.action_dim,
                        traj_horizon=self.horizon,
                        num_heads = self.cfgs['num_heads'].value,
                        attn_size = self.cfgs['N_attns'].value,
                        causal_step=self.cfgs['causal_step'].value,
                        inv_dynamics = self.inv_dynamics,
                        returns_condition=self.returns_condition,
                        condition_dropout = self.condition_dropout,
                        seperate_encoding = self.seperate_encoding,
                    )
                elif self.cfgs.algo_cfg.version == 'v2':
                    self.base_net = TransformerNetRA_SD(
                        hidden_size = self.cfgs['hidden_size'].value,
                        dim = self.sample_dim,
                        observation_dim=self.observation_dim,
                        action_dim=self.action_dim,
                        traj_horizon=self.horizon,
                        num_heads = self.cfgs['num_heads'].value,
                        attn_size = self.cfgs['N_attns'].value,
                        causal_step=self.cfgs['causal_step'].value,
                        inv_dynamics = self.inv_dynamics,
                        returns_condition=self.returns_condition,
                        condition_dropout = self.condition_dropout,
                        seperate_encoding = self.seperate_encoding,
                    )
                elif self.cfgs.algo_cfg.version == 'v3':
                    self.base_net = TransformerNetRA_SA(
                        hidden_size = self.cfgs['hidden_size'].value,
                        dim = self.sample_dim,
                        observation_dim=self.observation_dim,
                        action_dim=self.action_dim,
                        traj_horizon=self.horizon,
                        num_heads = self.cfgs['num_heads'].value,
                        attn_size = self.cfgs['N_attns'].value,
                        causal_step=self.cfgs['causal_step'].value,
                        inv_dynamics = self.inv_dynamics,
                        returns_condition=self.returns_condition,
                        condition_dropout = self.condition_dropout,
                        seperate_encoding = self.seperate_encoding,
                    )
                elif self.cfgs.algo_cfg.version == 'v4':
                    self.base_net = TransformerNetRA_DIM(
                        hidden_size = self.cfgs['hidden_size'].value,
                        dim = self.sample_dim,
                        observation_dim=self.observation_dim,
                        action_dim=self.action_dim,
                        traj_horizon=self.horizon,
                        num_heads = self.cfgs['num_heads'].value,
                        attn_size = self.cfgs['N_attns'].value,
                        causal_step=self.cfgs['causal_step'].value,
                        inv_dynamics = self.inv_dynamics,
                        returns_condition=self.returns_condition,
                        condition_dropout = self.condition_dropout,
                        seperate_encoding = self.seperate_encoding,
                        embed_ratio = self.cfgs['embed_ratio'].value,
                    )
                elif self.cfgs.algo_cfg.version == 'v5':
                    self.base_net = TransformerNetRA_DIM_cross(
                        hidden_size = self.cfgs['hidden_size'].value,
                        dim = self.sample_dim,
                        observation_dim=self.observation_dim,
                        action_dim=self.action_dim,
                        traj_horizon=self.horizon,
                        num_heads = self.cfgs['num_heads'].value,
                        attn_size = self.cfgs['N_attns'].value,
                        causal_step=self.cfgs['causal_step'].value,
                        inv_dynamics = self.inv_dynamics,
                        returns_condition=self.returns_condition,
                        condition_dropout = self.condition_dropout,
                        seperate_encoding = self.seperate_encoding,
                        embed_ratio = self.cfgs['embed_ratio'].value,
                    )
                elif self.cfgs.algo_cfg.version == 'v6':
                    self.base_net = TransformerNetRA_DIM_selfcross(
                        hidden_size = self.cfgs['hidden_size'].value,
                        dim = self.sample_dim,
                        observation_dim=self.observation_dim,
                        action_dim=self.action_dim,
                        traj_horizon=self.horizon,
                        num_heads = self.cfgs['num_heads'].value,
                        attn_size = self.cfgs['N_attns'].value,
                        causal_step=self.cfgs['causal_step'].value,
                        inv_dynamics = self.inv_dynamics,
                        returns_condition=self.returns_condition,
                        condition_dropout = self.condition_dropout,
                        seperate_encoding = self.seperate_encoding,
                        embed_ratio = self.cfgs['embed_ratio'].value,
                    )
                elif self.cfgs.algo_cfg.version == 'v7':
                    self.base_net = TransformerNetRA_DIM_cross_alladaln(
                        hidden_size = self.cfgs['hidden_size'].value,
                        dim = self.sample_dim,
                        observation_dim=self.observation_dim,
                        action_dim=self.action_dim,
                        traj_horizon=self.horizon,
                        num_heads = self.cfgs['num_heads'].value,
                        attn_size = self.cfgs['N_attns'].value,
                        causal_step=self.cfgs['causal_step'].value,
                        inv_dynamics = self.inv_dynamics,
                        returns_condition=self.returns_condition,
                        condition_dropout = self.condition_dropout,
                        seperate_encoding = self.seperate_encoding,
                        embed_ratio = self.cfgs['embed_ratio'].value,
                    )
                elif self.cfgs.algo_cfg.version == 'v8':
                    self.base_net = TemporalUnet(
                        hidden_size = self.cfgs['hidden_size'].value,
                        dim = self.sample_dim,
                        observation_dim=self.observation_dim,
                        action_dim=self.action_dim,
                        traj_horizon=self.horizon,
                        num_heads = self.cfgs['num_heads'].value,
                        attn_size = self.cfgs['N_attns'].value,
                        causal_step=self.cfgs['causal_step'].value,
                        inv_dynamics = self.inv_dynamics,
                        returns_condition=self.returns_condition,
                        condition_dropout = self.condition_dropout,
                        seperate_encoding = self.seperate_encoding,
                        embed_ratio = self.cfgs['embed_ratio'].value,
                    )
        else:
            self.base_net = TransformerNet(
                hidden_size = self.cfgs['hidden_size'].value,
                dim = self.sample_dim,
                observation_dim=self.observation_dim,
                action_dim=self.action_dim,
                num_heads = self.cfgs['num_heads'].value,
                attn_size = self.cfgs['N_attns'].value,
                causal_step=self.cfgs['causal_step'].value,
                inv_dynamics = self.inv_dynamics,
                returns_condition=self.returns_condition,
                condition_dropout = self.condition_dropout,
            )

    def ddpm_sample(self, rng, conditions, deterministic=False, returns=None, x = None):
        import ipdb
        ipdb.set_trace()
        batch_size = list(conditions.values())[0].shape[0] if len(list(conditions.values())[0].shape) > 1 else 1
        
        return self.diffusion.p_sample_loop_jit(
            rng_key=rng,
            model_forward=self.base_net,
            shape=(batch_size, self.horizon, self.sample_dim),
            conditions=conditions,
            condition_dim=self.sample_dim - self.action_dim,
            returns=returns,
            clip_denoised=True,
            x = x,
        )
        
    def gen_ddpm_sample(self, rng, conditions, goal_conditions, origin_data, deterministic=False, returns=None, x = None, t_add = 1, t_de = 1):
        # import ipdb
        # ipdb.set_trace()
        # batch_size = origin_data[0].shape[0]
        
        return self.diffusion.gen_p_sample_loop_jit(
            rng_key=rng,
            model_forward=self.base_net,
            shape=(1, len(origin_data), self.sample_dim),
            conditions=conditions,
            goal_conditions=goal_conditions,
            origin_data=origin_data,
            condition_dim=self.sample_dim,
            returns=returns,
            clip_denoised=True,
            x = None,
            t_add = t_add,
            t_de = t_de,
        )

    def dpm_sample(self, rng, samples, conditions, deterministic=False, returns=None):
        raise NotImplementedError
        noise_clip = True
        ns = NoiseScheduleVP(
            schedule="discrete", alphas_cumprod=self.diffusion.alphas_cumprod
        )

        def wrap_model(model_fn):
            def wrapped_model_fn(x, t, returns=None):
                t = (t - 1.0 / ns.total_N) * ns.total_N

                out = model_fn(rng, x, t, returns=returns)
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
            model_fn=wrap_model(partial(self.base_net, samples, returns=returns)),
            noise_schedule=ns,
            predict_x0=self.diffusion.model_mean_type is ModelMeanType.START_X,
        )
        x = jax.random.normal(rng, samples.shape)
        out = dpm_sampler.sample(x, steps=self.dpm_steps, t_end=self.dpm_t_end)

        return out

    def ddim_sample(self, rng, conditions, deterministic=False, returns=None, x = None):
        # expect a loop-jitted version of ddim_sample_loop, otherwise it's too slow
        raise NotImplementedError
        batch_size = list(conditions.items())[0].shape[0]
        return self.diffusion.ddim_sample_loop(
            rng_key=rng,
            model_forward=self.base_net,
            shape=(batch_size, self.horizon, self.sample_dim),
            conditions=conditions,
            returns=returns,
            clip_denoised=True,
            x = x,
        )

    def __call__(
        self, rng, conditions, deterministic=False, returns=None
    ):
        return getattr(self, f"{self.sample_method}_sample")(
            rng, conditions, deterministic, returns
        )

    # @partial(jax.jit())
    def loss(self, rng_key, samples, conditions, goal_conditions, ts, returns=None):
        # import ipdb
        # ipdb.set_trace()
        if not self.use_condition:
            conditions = None
        if not self.use_goal_condition:
            goal_conditions = None
        terms = self.diffusion.training_losses(
            rng_key,
            model_forward=self.base_net,
            x_start=samples,
            conditions=conditions,
            goal_conditions=goal_conditions,
            condition_dim=self.sample_dim,
            returns=returns,
            t=ts,
        )
        return terms
