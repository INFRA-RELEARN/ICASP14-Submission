import numpy as np

n_envs = 32

config = {
    # General parameters
    'env_id': 'HieracicalGammaProcessSystem2',
    'algorithm': 'PPO',  
    'artifact_dir': 'artifacts/runs/',
    'num_envs':  n_envs,
    'timesteps': 25_000_000,
    'eval_freq': 65536,
    'eval_seed': 4096,
    'eval_episodes': 32,
    'checkpoint_freq': 65536 * 10,


    ### Algorithm specific parameters
    'learning_rate': 5e-5,        # The learning rate, it can be a function of the current progress remaining (from 1 to 0)
                     #lambda x: (1-x) * 0.003 + 0.0003,
    'n_steps': 2048,                # The number of steps to run for each environment per update (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel) NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
    'batch_size': 256,              # Minibatch size
    'n_epochs': 10,                 # Number of epoch when optimizing the surrogate loss
    'gamma': 1.0,                   # Discount factor
    'gae_lambda': 0.91,             # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    'clip_range': 0.165,            # Clipping parameter, it can be a function of the current progress remaining (from 1 to 0).
    'clip_range_vf': None,          # Clipping parameter for the value function, it can be a function of the current progress remaining (from 1 to 0). This is a parameter specific to the OpenAI implementation. If None is passed (default), no clipping will be done on the value function. IMPORTANT: this clipping depends on the reward scaling.
    'normalize_advantage': True,    # Whether to normalize or not the advantage
    'ent_coef': 0.009,              # Entropy coefficient for the loss calculation
    'vf_coef': 0.639,               # Value function coefficient for the loss calculation
    'max_grad_norm': 0.56,          # The maximum value for the gradient clipping
    'use_sde': False,               # Whether to use generalized State Dependent Exploration (gSDE) instead of action noise exploration (default: False)
    'sde_sample_freq': -1,          # Sample a new noise matrix every n steps when using gSDE Default: -1 (only sample at the beginning of the rollout)
    'target_kl': None,              # Limit the KL divergence between updates, because the clipping is not enough to prevent large update see issue #213
    'verbose': 2,                   # Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for debug messages
    'seed': None,                   # Seed for the pseudo random generators
    'device': 'auto',               # Device (cpu, cuda, …) on which the code should be run. Setting it to auto, the code will be run on the GPU if possible.
    '_init_setup_model': True,      # Whether or not to build the network at the creation of the instance
    'tensorboard_log': None,        # the log location for tensorboard (if None, no tensorboard logging) 

    'PPO args': ['learning_rate', 'n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda', 'clip_range', 'clip_range_vf', 'normalize_advantage', 'ent_coef', 'vf_coef', 'max_grad_norm', 'use_sde', 'sde_sample_freq', 'target_kl', 'tensorboard_log', 'verbose', 'seed', 'device', '_init_setup_model'],

    # Policy specific parameters
    'policy_net_arch_l1': 80,
    'policy_net_arch_l2': 57,
    'policy_net_arch_l3': 0,

    # Environment specific parameters
    'env_kwargs': {
        'lam': 10,
        'b': 2,
        'theta': np.log(1+pow(0.01 / 10,2)/pow(2e-2 / 10,2)),
        'mu': np.log(pow(2e-2 / 10,2)/np.sqrt(pow(2e-2 / 10,2)+pow(0.01 / 10,2))),
        'delta_t': 0.1,
        'inspection_cost': 5,
        'repair_cost': 50,
        'repair_effect': 0.5,
        'replacement_cost': 500,
        'failure_cost': 1000,
        'failure_threshold': 25,
        'initial_state': 0.0,
        'time_horizon': 200,
        'steps_per_observation': 1,
    },

    'env_params': ['env_id','env_kwargs'],
    'env_config_passthrough': ['lam', 'b', 'theta', 'mu', 'delta_t', 'inspection_cost', 'repair_cost', 'repair_effect', 'replacement_cost', 'failure_cost', 'failure_threshold', 'initial_state', 'time_horizon', 'steps_per_observation'],

    'vec_normalize': True,
    'vec_normalize_kwargs': {
        'norm_obs': True,
        'norm_reward': True,
        'training': True,
    },
    'vec_normalize_load_path': None, #'artifacts/normalization/best_model_norm.pkl',
}
