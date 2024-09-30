import torch
import torch.nn as nn
import ipdb
import os
import wandb
import pickle
import numpy as np
import random
import math
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Normal
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
LOG_STD_MIN = -5
LOG_STD_MAX = 10

device = 'cuda:0'
# torch.device = device
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
        # ipdb.set_trace()
        states = torch.mean(distribution.sample((num,)) + s, dim = 0)
        
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
    
base_path = '/dysData/cqm/cqm/ddit/logs/2024-06-07-16-21-24_halfcheetah-medium-replay-v2_Gen_Nattn2_tadd0.5_tde0.5_gensourcesim_ntype=gaussian_cs-1_hori=20_dgap=frictionx0.25_crossTrue_Version=sar_dim_cross_aln_v7-er16_initcTrue_goalcFalse_returncFalse_seTrue_r10.0_sim0.0_et1_sd42_haha/diffuser_inv_d4rl/halfcheetah-medium-replay-v2/h_20-r_1200.0/100/SDEdit'



EditedFilePath = os.path.join(base_path, 'Edited.pkl')
unEditedFilePath = os.path.join(base_path, 'unEdited.pkl')
realFilePath = os.path.join(base_path, 'train_dataset.pkl')
with open(EditedFilePath, 'rb') as f:
    # 使用pickle.load从文件中加载数据
    EditedData = pickle.load(f)
# with open(tarFilePath, 'rb') as f:
#     tarData = pickle.load(f)
with open(unEditedFilePath, 'rb') as f:
    UnEditedData = pickle.load(f)
with open(realFilePath, 'rb') as f:
    realData = pickle.load(f)
new_realData = {
    'observations' : None,
    'actions' : None,
    'rewards' : None,
}
ipdb.set_trace()
num_state = realData['observations'].shape[-1]
num_action = realData['actions'].shape[-1]
for key in ['observations', 'actions', 'rewards']:
    for i in range(realData[key].shape[0]):
        # import ipdb
        # ipdb.set_trace()
        try:
            if new_realData[key] is not None:
                new_realData[key] = np.concatenate([new_realData[key], realData[key][i][:EditedData[0][key][0].shape[0] * (realData['traj_lengths'][i] // EditedData[0][key][0].shape[0])].reshape((-1,EditedData[0][key][0].shape[0],EditedData[0][key][0].shape[1]))], axis = 0)
            else:
                new_realData[key] = realData[key][i][:EditedData[0][key][0].shape[0] * (realData['traj_lengths'][i] // EditedData[0][key][0].shape[0])].reshape((-1,EditedData[0][key][0].shape[0],EditedData[0][key][0].shape[1]))
        except:
            ipdb.set_trace()
# ipdb.set_trace()
realData = new_realData
length = len(EditedData) if len(EditedData) < len(realData['observations']) else  len(realData['observations'])
sas_EditedData = []
sas_UnEditedData = []
sas_realData = []
for i in range(length - 1):
    # ipdb.set_trace()
    try:
        # ipdb.set_trace()
        for j in range(EditedData[i]['observations'][0].shape[0] - 1):
            sas_EditedData.append((EditedData[i]['observations'][0][j], EditedData[i]['actions'][0][j],  EditedData[i]['observations'][0][j+1]))
            sas_UnEditedData.append((UnEditedData[i]['observations'][j], UnEditedData[i]['actions'][j],  UnEditedData[i]['observations'][j+1]))
            sas_realData.append((realData['observations'][i][j], realData['actions'][i][j], realData['observations'][i][j+1]))
    except:
        ipdb.set_trace()
  
# ipdb.set_trace()    
# initialize dynamics model
batch_size = 256
model_lr = 1e-4
dynamics_model = Dynamics(num_state, num_action, 256, device).to(device)
model_optimizer = torch.optim.Adam(dynamics_model.parameters(), lr=model_lr)

def Sample(sas_list, batch_size):
    data = random.sample(sas_realData, batch_size)
    obs = []
    act = []
    next_obs = []
    for d in data:
        obs.append(d[0])
        act.append(d[1])
        next_obs.append(d[2])
    # ipdb.set_trace()
    obs = np.array(obs)
    act = np.array(act)
    next_obs = np.array(next_obs)
    obs = torch.FloatTensor(obs).to(device)
    act = torch.FloatTensor(act).to(device)
    next_obs = torch.FloatTensor(next_obs).to(device)
    return obs, act, next_obs
    
    
run = wandb.init(
    # Set the project where this run will be logged
    project="dynamic_model",
    name='test',
)

Edit_obs, Edit_action, Edit_next_obs = Sample(sas_EditedData, batch_size)
unEdit_obs, unEdit_action, unEdit_next_obs = Sample(sas_UnEditedData, batch_size)
try:
    # ipdb.set_trace()
    raise error
    # dynamics_model.load_state_dict(torch.load('model_weights_100k_wkmr.pth'))
except:
    for n in tqdm(range(100000)):


        real_obs, real_action, real_next_obs = Sample(sas_realData, batch_size)
        # ipdb.set_trace() 
        minus_logp_pi = dynamics_model.get_loss(real_obs, real_action, real_next_obs - real_obs)
        model_optimizer.zero_grad()
        minus_logp_pi.backward()
        model_optimizer.step()
        if n % 100 == 0:
            metrics = {}
            Edit_minus_logp_pi = dynamics_model.get_loss(Edit_obs, Edit_action, Edit_next_obs - Edit_obs)
            unEdit_minus_logp_pi = dynamics_model.get_loss(unEdit_obs, unEdit_action, unEdit_next_obs - unEdit_obs)

            metrics['model_loss'] = minus_logp_pi.cpu().detach().numpy().item()
            metrics['Edit_loss'] = Edit_minus_logp_pi.cpu().detach().numpy().item()
            metrics['unEdit_loss'] = unEdit_minus_logp_pi.cpu().detach().numpy().item()
            wandb.log(metrics)

    torch.save(dynamics_model.state_dict(), 'model_weights_100k_wkmr.pth')
dynamics_model.eval()
mse_loss = nn.MSELoss()

def CalMSE(data, batch_size):
    obs, action, next_obs = Sample(data, batch_size)
    pred_next_obs = dynamics_model.get_next_state(obs, action, num = 1 )
    loss = torch.mean((pred_next_obs - next_obs) ** 2, dim=-1)
    # ipdb.set_trace()
    # if loss > 50:
    #     ipdb.set_trace()
    return loss

real_loss = CalMSE(sas_realData, 1000).cpu().numpy()
Edit_loss = CalMSE(sas_EditedData, 1000).cpu().numpy()
unEdit_loss = CalMSE(sas_UnEditedData, 1000).cpu().numpy()

# 使用 seaborn 画直方图和核密度图
# ipdb.set_trace()
sns.histplot(real_loss, color='blue')  # kde=True 会同时绘制核密度估计图
sns.histplot(Edit_loss, color='red')
sns.histplot(unEdit_loss, color='green')
plt.savefig('dynamic_distribution.png')
