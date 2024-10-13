import argparse
import importlib
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ['CUDA_VISIBLE_DEVICES']='1'
import sys

import absl
import ipdb
from utilities.utils import define_flags_with_default
import ipdb
import numpy as np
import random
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument("--N_attns", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--theta", type=float, default=0.15)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--causal_step", type=int, default=-1) # -1 means no mask
    parser.add_argument("--config", type=str, default='configs/diffuser_inv_walker/diffuser_inv_walker_mdexpert.py')
    parser.add_argument("-g", type=int, default=0)
    parser.add_argument('--run_name', type=str, required = True)
    # parser.add_argument('--env_name', type=str, required = True)
    parser.add_argument('--ratio', type=float, required=False)
    parser.add_argument('--gen_trajs', action='store_true')
    parser.add_argument('--opt', default='train', choices=['train', 'gen'])
    parser.add_argument('--t_add', default=0.5, type=float)
    parser.add_argument('--t_de', default=0.5, type=float)
    parser.add_argument('--edit_time', default=1, type=int)
    parser.add_argument('--gen_source', default='sim', choices=['target', 'sim'])
    parser.add_argument('--ckpt_path', type=str, default='/home/chenqm/projects/ddit/logs/2024-03-30-15-52-57_hopper-medium-expert-v2_N_attn6_hiddensize512_num_heads2_causal_step-1_hopper_cfg_test/diffuser_inv_d4rl/hopper-medium-expert-v2/h_20-r_1200.0/100/checkpoints/model_0')
    parser.add_argument('--noise_type', type=str, default='gaussian', choices=['gaussian', 'ou', 'pure'])
    parser.add_argument('--use_condition', action='store_true')
    parser.add_argument('--use_goal_condition', action='store_true')
    parser.add_argument('--returns_condition', action='store_true')
    # parser.add_argument('--edit_sar', action='store_true')
    parser.add_argument('--net_arch', type=str, default='cross_alladaln_v9', choices=['sar_tmp_v8', 'inv', 'sar_v0', 'self_cross_v1', 'cross_self_v2', 'cross_v3', 'cross_dim_v4', 'cross_woAdaLN_v5', 'cross_LNq_v6', 'cross_SArew_v7', 'cross_blocks_v8', 'cross_alladaln_v9', 'inv_v0', 'sar_v1', 'sar_sd_v2', 'sar_sa_v3', 'sar_dim_v4', 'sar_dim_cross_v5', 'sar_dim_scross_v6', 'sar_dim_cross_aln_v7', 'sar_dim_cross_aln_nose_v10', 'sar_dim_v7', 'sar_dim_cross_v11', 'sar_dim_cross_v12'])
    # parser.add_argument('--use_inv_dynamic', action='store_true')
    # parser.add_argument('--use_reward_dynamic', action='store_true')
    parser.add_argument('--seperate_encoding', action='store_true')
    parser.add_argument('--sim_ratio', type=float, default=1.)
    parser.add_argument('--max_gen_traj_length', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--variety_degree', type=float, default=2.)
    parser.add_argument('--dynamic', type=str, default='gravity', choices=['gravity', 'friction', 'thigh_size', 'torso_length', 'joint_noise'])
    parser.add_argument('--std', type=float, default=0.)
    parser.add_argument('--horizon', type=int, default=20)
    parser.add_argument('--embed_ratio', type=int, default=16)
    parser.add_argument('--real_ratio', type=float, default = 1.)
    args, unknown_flags = parser.parse_known_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # ipdb.set_trace()
    from utilities.utils import import_file
    assert args.hidden_size % args.num_heads == 0
    config = getattr(import_file(args.config, "default_config"), "get_config")()
    # ipdb.set_trace()
    config['horizon'] = args.horizon
    config['algo_cfg']['horizon'] = args.horizon
    config['gen_num'] = int(args.max_gen_traj_length / args.horizon)
    args.max_gen_traj_length = config['horizon'] * int(args.max_gen_traj_length // config['horizon'])

    edit_sar = True if not args.net_arch == 'inv' else False
    use_inv_dynamic = True if args.net_arch == 'inv' else False
    use_reward_dynamic = True if args.net_arch == 'inv' else False
    use_cross = True if 'cross' in args.net_arch else False
    
    
    # version = args.net_arch.split('_')[-1]
   
    config['algo_cfg']['version'] = args.net_arch.split('_')[-1]

    config['gen_trajs'] = args.gen_trajs
    if args.ratio is not None:
        config['ratio'] = args.ratio
    data_ratio = config['ratio']
    config['generator_class'] = "Generator"
    config['variety_degree'] = args.variety_degree
    config['N_attns'] = args.N_attns
    config['num_heads'] = args.num_heads
    config['hidden_size'] = args.hidden_size
    config['embed_ratio'] = args.embed_ratio
    config['causal_step'] = args.causal_step
    config['ckpt_path'] = args.ckpt_path
    config['theta'] = args.theta
    config['mu'] = args.mu
    config['sigma'] = args.sigma
    config['noise_type'] = args.noise_type
    config['use_condition'] = args.use_condition
    config['use_goal_condition'] = args.use_goal_condition
    config['returns_condition'] = args.returns_condition
    config['algo_cfg']['edit_sar'] = edit_sar
    config['algo_cfg']['use_cross'] = use_cross
    config['use_inv_dynamic'] = use_inv_dynamic
    config['use_reward_dynamic'] = use_reward_dynamic
    config['seperate_encoding'] = args.seperate_encoding
    # config['gen_n_trajs'] = args.gen_n_trajs
    config['max_gen_traj_length'] = args.max_gen_traj_length
    config['batch_size'] = args.batch_size
    config['edit_time'] = args.edit_time
    config['sim_ratio'] = args.sim_ratio
    config['dynamic'] = args.dynamic
    config['max_traj_length'] = config['horizon'] * int(config['max_traj_length'] // config['horizon'])
    config['std'] = args.std
    config['real_ratio'] = args.real_ratio
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    config = define_flags_with_default(**config)
    
    
    # ipdb.set_trace()
    absl.flags.FLAGS(sys.argv[:1] + unknown_flags)

    env = config.get('env')
    if not args.gen_trajs:
        # ipdb.set_trace()

        run_name = f'{env}_Nattn{args.N_attns}_hs{args.hidden_size}_n_heads{args.num_heads}_cs{args.causal_step}_hori={args.horizon}_cross{use_cross}_Version={args.net_arch}-er{args.embed_ratio}_constraint{args.use_condition}_goalc{args.use_goal_condition}_returnc{args.returns_condition}_se{args.seperate_encoding}_bs{args.batch_size}_r{data_ratio}_s{args.seed}_{args.run_name}'
    else:
        run_name = f'{env}_Gen_Nattn{args.N_attns}_tadd{args.t_add}_tde{args.t_de}_gensource{args.gen_source}_ntype={args.noise_type}_cs{args.causal_step}_hori={args.horizon}_dgap={args.dynamic}x{args.variety_degree}_Version={args.net_arch}-er{args.embed_ratio}_initc{args.use_condition}_goalc{args.use_goal_condition}_returnc{args.returns_condition}_se{args.seperate_encoding}_r{data_ratio}_sim{args.sim_ratio}_et{args.edit_time}_sd{args.seed}_{args.run_name}'

    trainer = getattr(
        importlib.import_module("diffuser.trainer"), absl.flags.FLAGS.trainer
    )(config, run_name = run_name)
    
    # ipdb.set_trace()
    
    trainer.train(opt = args.opt, t_add = args.t_add, t_de = args.t_de, source = args.gen_source)


if __name__ == "__main__":
    main()
    # print('jajaj')
