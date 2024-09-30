import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn

class PositionalEncoding(nn.Module):
    d_model: int
    max_len: int

    def setup(self):

        position = jnp.arange(self.max_len).reshape(-1, 1)
        div_term = jnp.exp(jnp.arange(0, self.d_model, 2) * (-jnp.log(10000.0) / self.d_model))
        pe = jnp.zeros((1, self.max_len, self.d_model))
        pe = pe.at[0, :, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[0, :, 1::2].set(jnp.cos(position * div_term))
        # import ipdb  
        # ipdb.set_trace()
        self.pe = pe

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Arguments:
            x: jnp.ndarray, shape [batch_size, seq_len, embedding_dim]
        """
        seq_len = x.shape[1]
        x = x + self.pe[:,:seq_len,:]
        # import ipdb
        # ipdb.set_trace()
        return x

# 示例用法
d_model = 512
max_len = 5000
pos_enc = PositionalEncoding(d_model=d_model, max_len=max_len)

# 假设输入是形状为 [seq_len, batch_size, embedding_dim] 的张量
rng_key = random.PRNGKey(0)
seq_len = 20
batch_size = 10
embedding_dim = d_model
input_data = jnp.ones((seq_len, batch_size, embedding_dim))
rngs = random.key(0)
params = pos_enc.init(rngs, input_data)
output = pos_enc.apply(params, input_data)

print("Output shape:", output.shape)

