from ml_collections import ConfigDict

from utilities.utils import WandBLogger


def get_config():
    config = ConfigDict()
    config.exp_name = "dql_d4rl"
    config.log_dir_format = (
        "{exp_name}/{env}/lr_{algo_cfg.lr}/{seed}"
    )

    config.trainer = "DiffusionQLTrainer"
    config.type = "model-free"

    config.env = "walker2d-medium-replay-v2"
    config.dataset = "d4rl"
    config.dataset_class = "QLearningDataset"
    config.use_padding = True
    config.normalizer = "LimitsNormalizer"
    config.max_traj_length = 1000
    config.horizon = 1
    config.returns_scale = 1.0
    config.include_returns = False
    config.termination_penalty = 0.0

    config.seed = 100
    config.batch_size = 256
    config.reward_scale = 1
    config.reward_bias = 0
    config.clip_action = 0.999
    config.encoder_arch = "64-64"
    config.policy_arch = "256-256-256"
    config.qf_arch = "256-256-256"
    config.orthogonal_init = False
    config.policy_log_std_multiplier = 1.0
    config.policy_log_std_offset = -1.0

    config.n_epochs = 1000
    config.n_train_step_per_epoch = 1000

    config.evaluator_class = "OnlineEvaluator"
    config.eval_period = 100
    config.eval_n_trajs = 10
    config.num_eval_envs = 10
    config.eval_env_seed = 0

    config.qf_layer_norm = False
    config.policy_layer_norm = False
    config.activation = "mish"
    config.obs_norm = False
    config.act_method = "ddpmensemble"
    config.sample_method = "ddpm"
    config.policy_temp = 1.0
    config.norm_reward = False

    config.save_period = 0
    config.logging = WandBLogger.get_default_config()

    config.algo_cfg = ConfigDict()
    config.algo_cfg.discount = 0.99
    config.algo_cfg.tau = 0.005
    config.algo_cfg.policy_tgt_freq = 5
    config.algo_cfg.sample_temperature = 1.0
    config.algo_cfg.num_timesteps = 100
    config.algo_cfg.schedule_name = "linear"
    config.algo_cfg.time_embed_size = 16
    config.algo_cfg.alpha = 2.0  # NOTE 0.25 in diffusion rl but 2.5 in td3
    config.algo_cfg.use_pred_astart = True
    config.algo_cfg.max_q_backup = False
    config.algo_cfg.max_q_backup_topk = 1
    config.algo_cfg.max_q_backup_samples = 10
    config.algo_cfg.nstep = 1

    # learning related
    config.algo_cfg.lr = 3e-4
    config.algo_cfg.diff_coef = 1.0
    config.algo_cfg.guide_coef = 1.0
    config.algo_cfg.lr_decay = False
    config.algo_cfg.lr_decay_steps = 1000000
    config.algo_cfg.max_grad_norm = 0.0
    config.algo_cfg.weight_decay = 0.0

    config.algo_cfg.loss_type = "TD3"
    config.algo_cfg.sample_logp = False

    config.algo_cfg.adv_norm = False
    # CRR-related hps
    config.algo_cfg.sample_actions = 10
    config.algo_cfg.crr_ratio_upper_bound = 20
    config.algo_cfg.crr_beta = 1.0
    config.algo_cfg.crr_weight_mode = "mle"
    config.algo_cfg.fixed_std = True
    config.algo_cfg.crr_multi_sample_mse = False
    config.algo_cfg.crr_avg_fn = "mean"
    config.algo_cfg.crr_fn = "exp"

    # IQL-related hps
    config.algo_cfg.expectile = 0.7
    config.algo_cfg.awr_temperature = 3.0

    # for dpm-solver
    config.algo_cfg.dpm_steps = 15
    config.algo_cfg.dpm_t_end = 0.001

    # useless
    config.algo_cfg.target_entropy = -1

    return config
