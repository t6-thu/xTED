import pickle
import ipdb
import numpy as np
import wandb
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from data.d4rl import get_dataset
import gym
import os
from env.maze_continual import Maze
from k import mean_ks_statistic
import matplotlib.lines as mlines
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
# base_path_g2 = '/dysData/DISCOVER/nhy/cqm/ddit/logs/2024-05-10-06-55-21_halfcheetah-medium-replay-v2_Gen_Nattn2_tadd0.5_tde0.5_gensourcesim_ntype=gaussian_cs-1_hori=20_dgap=gravityx2.0_crossTrue_Version=sar_dim_cross_aln_v7-er16_initcTrue_goalcFalse_returncFalse_seTrue_r10.0_sim1.0_et1_sd42_haha/diffuser_inv_d4rl/halfcheetah-medium-replay-v2/h_20-r_1200.0/100/SDEdit'
# base_path_f025 = '/dysData/DISCOVER/nhy/cqm/ddit/logs/2024-05-12-06-50-54_halfcheetah-medium-replay-v2_Gen_Nattn2_tadd0.5_tde0.5_gensourcesim_ntype=gaussian_cs-1_hori=20_dgap=frictionx0.25_crossTrue_Version=sar_dim_cross_aln_v7-er16_initcTrue_goalcFalse_returncFalse_seTrue_r10.0_sim1.0_et1_sd42_haha/diffuser_inv_d4rl/halfcheetah-medium-replay-v2/h_20-r_1200.0/100/SDEdit'
# base_path_j1 = '/dysData/DISCOVER/nhy/cqm/ddit/logs/2024-05-14-11-57-21_halfcheetah-medium-replay-v2_Gen_Nattn2_tadd0.5_tde0.5_gensourcesim_ntype=gaussian_cs-1_hori=20_dgap=joint_noisex1.0_crossTrue_Version=sar_dim_cross_v5-er16_initcTrue_goalcFalse_returncFalse_seTrue_r10.0_sim1.0_et1_sd42_haha/diffuser_inv_d4rl/halfcheetah-medium-replay-v2/h_20-r_1200.0/100/SDEdit'
base_path = '/dysData/DISCOVER/nhy/cqm/ddit/logs/2024-06-08-13-47-13_walker2d-medium-expert-v2_Gen_Nattn2_tadd0.5_tde0.5_gensourcesim_ntype=gaussian_cs-1_hori=20_dgap=gravityx2.0_crossTrue_Version=sar_dim_cross_aln_v7-er16_initcTrue_goalcFalse_returncFalse_seTrue_r100.0_sim0.0_et1_sd42_alladaln/diffuser_inv_d4rl/walker2d-medium-expert-v2/h_20-r_1200.0/100/SDEdit'

# base_paths = [base_path_g2, base_path_t2, base_path_f025, base_path_j1]
j = 0
fig, axes = plt.subplots(1,1,figsize=(8, 8))



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
sars_EditedData = []
sars_UnEditedData = []
sars_realData = []
for i in range(length - 1):
    # ipdb.set_trace()
    try:
        sars_EditedData.append(np.concatenate([EditedData[i]['observations'][0], EditedData[i]['actions'][0], EditedData[i]['rewards'][0], EditedData[i+1]['observations'][0],], axis = -1))
        sars_UnEditedData.append(np.concatenate([UnEditedData[i]['observations'], UnEditedData[i]['actions'], UnEditedData[i]['rewards'], UnEditedData[i+1]['observations'],], axis = -1))
        sars_realData.append(np.concatenate([realData['observations'][i], realData['actions'][i], realData['rewards'][i], realData['observations'][i+1],], axis = -1))
    except:
        ipdb.set_trace()
for i in range(25):
    # ipdb.set_trace()
    pca_Edited = pca.fit_transform(sars_EditedData[i])
    pca_UnEdited = pca.fit_transform(sars_UnEditedData[i])
    pca_real = pca.fit_transform(sars_realData[i])
    axes.scatter(pca_Edited[:,0], pca_Edited[:,1], color=color['dark_green'], s = 10.0, alpha=1.)
    axes.scatter(pca_UnEdited[:,0], pca_UnEdited[:,1], color = color['dark_red'], s = 10.0, alpha=1.)
    axes.scatter(pca_real[:,0], pca_real[:,1], color=color['bright_blue'], s = 10.0, alpha=1.)
    # plt.legend()
    axes.plot()
    # plt.savefig(os.path.join(base_path, f'{i}.png'))
    # plt.close()
pca_Edited = pca.fit_transform(sars_EditedData[-1])
pca_UnEdited = pca.fit_transform(sars_UnEditedData[-1])
pca_real = pca.fit_transform(sars_realData[-1])
axes.scatter(pca_Edited[:,0], pca_Edited[:,1], color=color['dark_green'], s = 10.0, alpha=1.,)
axes.scatter(pca_UnEdited[:,0], pca_UnEdited[:,1], color=color['dark_red'], s = 10.0, alpha=1.,)
axes.scatter(pca_real[:,0], pca_real[:,1], color=color['bright_blue'], s = 10.0, alpha=1.,)
axes.grid(True)  # 默认显示主网格线
axes.grid(which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)  # 自定义网格线样式
axes.tick_params(axis='both', which='major', labelsize=24)
# axes.legend()
# axes[j].savefig(f'distribution{j}.png')
# j += 1    
# plt.close()

# plt.legend(ncol=3, frameon=False, bbox_to_anchor=(0.8,-0), fontsize=24)
# 创建自定义图例句柄
axes.scatter([], [], color=color['dark_green'], s = 40.0, alpha=1., label = 'Src.(Edited)')
axes.scatter([], [], color=color['dark_red'], s = 40.0, alpha=1., label = 'Src.(Unedited)')
axes.scatter([], [], color=color['bright_blue'], s = 40.0, alpha=1., label = 'Tgt.')
# axes.text(-0.05, 1.05, '(a)', transform=axes.transAxes, fontsize=24, verticalalignment='top', horizontalalignment='left')
plt.legend(fontsize=20)
plt.title('WK-MR', fontsize=28)
plt.savefig('distribution_wk.png', dpi=400)