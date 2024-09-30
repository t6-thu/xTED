import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Normal
import wandb
import gym
# import d4rl
import math
from tqdm import tqdm
import os
import numpy as np



from Network.Actor_Critic_net import LOG_STD_MAX, LOG_STD_MIN


class Dynamics(nn.Module):
    def __init__(self, num_state, num_action, num_hidden, dropout=False, device="cuda"):
        super(Dynamics, self).__init__()
        self.device = device
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(p=0.2)

        self.fc1 = nn.Linear(num_state + num_action, num_hidden)
        torch.nn.init.orthogonal_(self.fc1.weight.data, gain=math.sqrt(2.0))

        self.fc2 = nn.Linear(num_hidden, num_hidden)
        torch.nn.init.orthogonal_(self.fc2.weight.data, gain=math.sqrt(2.0))
        
        self.fc3 = nn.Linear(num_hidden, num_hidden)
        torch.nn.init.orthogonal_(self.fc3.weight.data, gain=math.sqrt(2.0))

        self.mu_head = nn.Linear(num_hidden, num_state)
        torch.nn.init.orthogonal_(self.mu_head.weight.data, gain=1e-2)

        self.sigma_head = nn.Linear(num_hidden, num_state)
        torch.nn.init.orthogonal_(self.sigma_head.weight.data, gain=1e-2)

    def get_loss(self, s, a, ds):
        if isinstance(s, np.ndarray):
            s = torch.tensor(s, dtype=torch.float).to(self.device)
        if isinstance(a, np.ndarray):
            a = torch.tensor(a, dtype=torch.float).to(self.device)
        if isinstance(ds, np.ndarray):
            ds = torch.tensor(ds, dtype=torch.float).to(self.device)

        x = torch.cat((s,a), 1)
        if self.dropout:
            x = F.relu(self.dropout_layer(self.fc1(x)))
            x = F.relu(self.dropout_layer(self.fc2(x)))
            x = F.relu(self.dropout_layer(self.fc3(x)))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
        mu = self.mu_head(x)
        log_sigma = self.sigma_head(x)

        y = ds.to(self.device)

        log_sigma = torch.clamp(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        self.sigma = torch.exp(log_sigma)

        a_distribution = Normal(mu, self.sigma)
        logp_pi = a_distribution.log_prob(y)

        return -logp_pi.mean()

    def get_next_state(self, s, a, num):
        if isinstance(s, np.ndarray):
            s = torch.tensor(s, dtype=torch.float).to(self.device)
        if isinstance(a, np.ndarray):
            a = torch.tensor(a, dtype=torch.float).to(self.device)
        
        x = torch.cat((s,a), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_sigma = self.sigma_head(x)

        log_sigma = torch.clip(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        self.sigma = torch.exp(log_sigma)
        distribution = Normal(mu, self.sigma)

        # TODO sample many times for each state-action pair
        states = distribution.sample((num,)) + s
        
        return states

    def get_dstate_deter(self, s, a):
        if isinstance(s, np.ndarray):
            s = torch.tensor(s, dtype=torch.float).to(self.device)
        if isinstance(a, np.ndarray):
            a = torch.tensor(a, dtype=torch.float).to(self.device)

        x = torch.cat((s,a), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)

        return mu
    
# initialize dynamics model
dynamics_model = Dynamics(num_state, num_action, 256, FLAGS.device).to(FLAGS.device)
model_optimizer = torch.optim.Adam(dynamics_model.parameters(), lr=FLAGS.model_lr)
for n in range(FLAGS.model_train_epoch):
    real_obs, real_action, real_next_obs = replay_buffer.sample(FLAGS.batch_size, scope="real", type="sas").values()
    minus_logp_pi = dynamics_model.get_loss(real_obs, real_action, real_next_obs - real_obs)
    model_optimizer.zero_grad()
    minus_logp_pi.backward()
    model_optimizer.step()
    if n % 100 == 0:
        metrics = {}
        metrics['model_loss'] = minus_logp_pi.cpu().detach().numpy().item()
        wandb_logger.log(metrics)
