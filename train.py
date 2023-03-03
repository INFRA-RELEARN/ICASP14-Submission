import os
from datetime import datetime
import argparse
import yaml
import numpy as np
import torch as th
import wandb

from utils.training_callbacks import SaveNormalizeOnBestEvalRewardCallback, CheckpointCallback
from utils.wandb_logger import WandBLogger

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.logger import Logger,CSVOutputFormat
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor

from wandb.integration.sb3 import WandbCallback

from environments.utils import get_env
from config import config

from pprint import pprint

def get_ppo_kwargs(hyper_dict):
    return { ppo_arg: hyper_dict[ppo_arg] for ppo_arg in hyper_dict['PPO args'] }

def get_policy_kwargs(hyper_dict):
    net_arch = [hyper_dict['policy_net_arch_l1'], hyper_dict['policy_net_arch_l2'], hyper_dict['policy_net_arch_l3']]
    net_arch = [int(x) for x in net_arch if int(x) > 0]
    
    policy_kwargs = dict(
        net_arch= net_arch,
    )

    return policy_kwargs

def get_training_setup_ppo(hyper_dict):
    n_envs = hyper_dict['num_envs']

    for id in hyper_dict['env_config_passthrough']:
        if id in hyper_dict:
            hyper_dict['env_kwargs'][id] = hyper_dict[id]

    env_params = {id : hyper_dict[id] for id in hyper_dict['env_params']}
    env_params_eval = env_params.copy()
    env_params_eval['evaluation'] = True    

    train_env = VecMonitor(
                    SubprocVecEnv(
                        [lambda: get_env(env_params) for _ in range(n_envs)]))

    eval_env =  VecMonitor(
                    SubprocVecEnv(
                        [lambda:  get_env(env_params) for _ in range(n_envs)]))

    if hyper_dict['vec_normalize']:
        if hyper_dict['vec_normalize_load_path'] is None:
            train_env = VecNormalize(train_env, gamma=hyper_dict['gamma'], **hyper_dict['vec_normalize_kwargs'])
            eval_env = VecNormalize(eval_env, gamma=hyper_dict['gamma'], **hyper_dict['vec_normalize_kwargs'])
        else:
            train_env = VecNormalize.load(hyper_dict['vec_normalize_load_path'], venv=train_env)
            eval_env = VecNormalize.load(hyper_dict['vec_normalize_load_path'], venv=eval_env)
            train_env.training = hyper_dict['vec_normalize_kwargs']['training']
            eval_env.training = hyper_dict['vec_normalize_kwargs']['training']
        

    policy_kwargs = get_policy_kwargs(hyper_dict)
    ppo_kwargs = get_ppo_kwargs(hyper_dict)

    ppo_kwargs['device'] = hyper_dict['model_device']

    model = PPO(MlpPolicy, train_env, policy_kwargs=policy_kwargs, **ppo_kwargs)
    
    return train_env, eval_env, model


def train_model(run_config):
    artifact_dir = run_config['artifact_dir']

    print('Starting new run with this configuration:')
    pprint(dict(run_config))


    if run_config['algorithm'] == 'PPO':
        env, eval_env, model = get_training_setup_ppo(run_config)
    else:
        print('Algorithm not implemented yet.')
        return

    model.set_logger(Logger(artifact_dir, [CSVOutputFormat(artifact_dir + 'log.csv'), WandBLogger(log_max_params=['eval/mean_reward'])]))

    callback_wandb = WandbCallback()

    vec_normalize_value_path = f'{artifact_dir}/vec_normalize/'
    os.makedirs(vec_normalize_value_path, exist_ok=True)
    eval_callback = SaveNormalizeOnBestEvalRewardCallback(
            eval_env=eval_env, 
            best_model_save_path=f'{artifact_dir}models',
            log_path=artifact_dir,
            eval_freq=int(max([run_config['eval_freq'] // run_config['num_envs'],1])),
            deterministic=False,
            render=False,
            seed=run_config['eval_seed'] * 1000,
            verbose=1,
            n_eval_episodes=run_config['eval_episodes'],
            env=env,
            normalization_save_path = vec_normalize_value_path
        )

    eval_callback.model = model

    checkpoint_callback = CheckpointCallback(
        save_freq=int(max([run_config['checkpoint_freq'] // run_config['num_envs'],1])),
        save_path=f'{artifact_dir}checkpoints/',
        name_prefix='checkpoint_model',
        save_vec_normalize=run_config['vec_normalize'],
        env=env,
        verbose=0,
    )

    # save all config parameters to yaml file
    with open(f'{artifact_dir}config.yaml', 'w') as f:
        yaml.dump(dict(run_config), f)

    print('Starting training')
    
    model.learn(total_timesteps=run_config['timesteps'], callback=CallbackList([callback_wandb, eval_callback, checkpoint_callback]))
    print('Finished training')
    model.save(f'{artifact_dir}models/latest_model')
    env.save(f'{vec_normalize_value_path}/latest_model_norm.pkl')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-logging', dest='no_logging', action='store_true', default=True)
    parser.add_argument('--sweep', dest='sweep', action='store_true')
    parser.add_argument('--algorithm', type=str, choices=['PPO','DQN'])
    parser.add_argument('--timesteps', type=int)
    parser.add_argument('--eval-seed', dest='eval_seed', type=int)
    parser.add_argument('--log-sub-dir', dest='log_sub_dir', type=str)

    cmd_config = parser.parse_args()

    hyper_dict = config

    set_cmd_config = {key:value for key, value in vars(parser.parse_args()).items() if value is not None}
  
    hyper_dict.update(set_cmd_config)

    wandb.init(
            project='INFRA.RELEARN  - SB3',
            entity='tum-dh',
            sync_tensorboard=False,
            config=hyper_dict,
            mode='disabled' if cmd_config.no_logging else 'online'
    )

    run_name = wandb.run.name
    
    print('wandb name: ', run_name)

    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    if cmd_config.log_sub_dir is not None:
        artifact_dir = f'artifacts/runs/{cmd_config.log_sub_dir}/{run_name}_{time_stamp}/'
    else:
        artifact_dir = f'artifacts/runs/{run_name}_{time_stamp}/'
    os.makedirs(artifact_dir, exist_ok=True)

    hyper_dict['artifact_dir'] = artifact_dir
    hyper_dict['run_name'] = run_name

    device = 'auto'

    hyper_dict['model_device'] = device

    wandb.config.update(hyper_dict, allow_val_change=True)

    train_model(wandb.config)

if __name__ == '__main__':
    main()
