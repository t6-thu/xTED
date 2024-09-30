import pickle
import ipdb
import numpy as np
import wandb
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from data.d4rl import get_dataset
import gym
import os
from env.maze_continual import Maze
from k import mean_ks_statistic

import seaborn as sns
# run = wandb.init(
#     # Set the project where this run will be logged
#     project="SDEdit_test",
#     name='0.1',
# )
color = {
    'dark_red': '#DC4437',
    'bright_blue': '#4385F5',
    'gray': "#B2B2B2",
    'light_blue': '#A0C2FF',
    'bright_yellow': '#F5B400',
    'dark_green': "#109D59",
}

pca = PCA(n_components=2)

# base_path_t2 = '/dysData/DISCOVER/nhy/cqm/ddit/logs/2024-05-12-06-51-02_halfcheetah-medium-replay-v2_Gen_Nattn2_tadd0.5_tde0.5_gensourcesim_ntype=gaussian_cs-1_hori=20_dgap=thigh_sizex2.0_crossTrue_Version=sar_dim_cross_aln_v7-er16_initcTrue_goalcFalse_returncFalse_seTrue_r10.0_sim1.0_et1_sd42_haha/diffuser_inv_d4rl/halfcheetah-medium-replay-v2/h_20-r_1200.0/100/SDEdit'
base_path = '/dysData/DISCOVER/nhy/cqm/ddit/logs/2024-05-31-07-26-12_walker2d-medium-replay-v2_Gen_Nattn2_tadd0.5_tde0.5_gensourcesim_ntype=gaussian_cs-1_hori=20_dgap=gravityx2.0_crossTrue_Version=sar_dim_cross_aln_v7-er16_initcTrue_goalcFalse_returncFalse_seTrue_r15.0_sim10.0_et1_sd42_alladaln/diffuser_inv_d4rl/walker2d-medium-replay-v2/h_20-r_1200.0/100/SDEdit'
# base_path_f025 = '/dysData/DISCOVER/nhy/cqm/ddit/logs/2024-05-12-06-50-54_halfcheetah-medium-replay-v2_Gen_Nattn2_tadd0.5_tde0.5_gensourcesim_ntype=gaussian_cs-1_hori=20_dgap=frictionx0.25_crossTrue_Version=sar_dim_cross_aln_v7-er16_initcTrue_goalcFalse_returncFalse_seTrue_r10.0_sim1.0_et1_sd42_haha/diffuser_inv_d4rl/halfcheetah-medium-replay-v2/h_20-r_1200.0/100/SDEdit'
# base_path_j1 = '/dysData/DISCOVER/nhy/cqm/ddit/logs/2024-05-14-11-57-21_halfcheetah-medium-replay-v2_Gen_Nattn2_tadd0.5_tde0.5_gensourcesim_ntype=gaussian_cs-1_hori=20_dgap=joint_noisex1.0_crossTrue_Version=sar_dim_cross_v5-er16_initcTrue_goalcFalse_returncFalse_seTrue_r10.0_sim1.0_et1_sd42_haha/diffuser_inv_d4rl/halfcheetah-medium-replay-v2/h_20-r_1200.0/100/SDEdit'

# base_paths = [base_path_g2, base_path_t2, base_path_f025, base_path_j1]
j = 0
fig = plt.figure(figsize=(8, 8))
gs = GridSpec(4, 4, figure=fig)


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
z_EditedData = []
z_UnEditedData = []
z_realData = []
al_EditedData = []
al_UnEditedData = []
al_realData = []
for i in range(25):
    # ipdb.set_trace()
    z_EditedData.append(EditedData[i]['actions'][0,:,0])
    al_EditedData.append(EditedData[i]['rewards'][0,:,0])
    z_UnEditedData.append(UnEditedData[i]['actions'][:,0])
    al_UnEditedData.append(UnEditedData[i]['rewards'][:,0])
    z_realData.append(realData['actions'][i][:,0])
    al_realData.append(realData['rewards'][i][:,0])

z_EditedData = np.array(z_EditedData).reshape(-1)
al_EditedData = np.array(al_EditedData).reshape(-1)
z_UnEditedData = np.array(z_UnEditedData).reshape(-1)
al_UnEditedData = np.array(al_UnEditedData).reshape(-1)
z_realData = np.array(z_realData).reshape(-1)
al_realData = np.array(al_realData).reshape(-1)

ax_main = fig.add_subplot(gs[1:4, 0:3])
ax_main.spines['top'].set_visible(False)
ax_main.spines['right'].set_visible(False)
ax_main.scatter(z_EditedData, al_EditedData, color = color['dark_green'], s = 10., alpha=1.)
ax_main.scatter(z_UnEditedData, al_UnEditedData, color = color['dark_red'], s = 10., alpha=1.)
ax_main.scatter(z_realData, al_realData, color = color['bright_blue'], s = 10., alpha=1.)
# ax_main.set(xlabel='X data', ylabel='Y data')
ax_main.tick_params(axis='both', which='major', labelsize=24)
ax_main.set_xlabel('Torque Applied on Thigh Rotor', fontsize=24)
ax_main.set_ylabel('Reward', fontsize=24)
ax_main.set_xlim([-1.25,1.25])
ax_main.set_ylim([-1,7])
ax_main.grid(True)  # 默认显示主网格线
ax_main.grid(which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)  # 自定义网格线样式


ax_xDist = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
ax_xDist.set_axis_off()
sns.kdeplot(z_EditedData, ax=ax_xDist, color = color['dark_green'], fill=True)
sns.kdeplot(z_UnEditedData, ax=ax_xDist, color = color['dark_red'], fill=True)
sns.kdeplot(z_realData, ax=ax_xDist, color = color['bright_blue'], fill=True)
ax_xDist.set(xlabel='', ylabel='Density')
ax_xDist.xaxis.set_visible(False)

ax_yDist = fig.add_subplot(gs[1:4, 3], sharey=ax_main)
ax_yDist.set_axis_off()
sns.kdeplot(al_EditedData, ax=ax_yDist, vertical=True, color = color['dark_green'], fill=True)
sns.kdeplot(al_UnEditedData, ax=ax_yDist, vertical=True, color = color['dark_red'], fill=True)
sns.kdeplot(al_realData, ax=ax_yDist, vertical=True, color = color['bright_blue'], fill=True)
ax_yDist.set(xlabel='Density', ylabel='')
ax_yDist.yaxis.set_visible(False)
plt.tight_layout()
plt.savefig('z_pdf_4_walker.png',dpi=400)        
