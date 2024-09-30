import jax.numpy as jnp
import jax
# import jax.random as random
class OUNoise:
    def __init__(self, shape, mu=0, theta=0.15, sigma=0.2):
        self.length = shape[0]
        self.feature_dim = shape[1]
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.rng_key = jax.random.PRNGKey(0)
        self.rng_key, self.sample_key = jax.random.split(self.rng_key)
        self.state = jax.random.normal(self.sample_key, shape=(self.feature_dim,), dtype=jnp.float32) * sigma + mu
        self.noise = []
        self.reset()

    def reset(self):
        self.rng_key, self.sample_key = jax.random.split(self.rng_key)
        self.state = jax.random.normal(self.sample_key, shape=(self.feature_dim,), dtype=jnp.float32) * self.sigma + self.mu
        self.noise = []

    def gen_noise(self, t=1):
        x = self.state
        total_noise = jnp.zeros((self.length, self.feature_dim))
        for j in range(t):
            self.noise = []
            self.noise.append(x)
            for i in range(self.length - 1):
                self.rng_key, self.sample_key = jax.random.split(self.rng_key)
                dx = self.theta * (self.mu - x) + self.sigma * jax.random.normal(self.sample_key, shape=(self.feature_dim,), dtype=jnp.float32)
                x = x + dx
                self.noise.append(x)
            self.noise = jnp.array(self.noise).reshape((self.length, self.feature_dim))
            total_noise += self.noise
        return total_noise


    
if __name__ == '__main__':
    OUGen = OUNoise(shape=(20,17))
    noise = OUGen.gen_noise(t=10)
    print(type(noise))
