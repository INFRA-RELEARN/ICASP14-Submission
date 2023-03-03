import time, os
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

class EnvSeedEvalCallback(EvalCallback):
    def __init__(self, seed=0, *args, **kwargs):
        self.seed = seed
        return super().__init__(*args, **kwargs)
        
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # print(f'Seeding envs: {self.seed}')
            self.eval_env.seed(self.seed)
        
        return_value = super()._on_step()
        
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_env.seed(int(time.time()*1e10))
        
        return return_value

class SaveNormalizeOnBestEvalRewardCallback(EnvSeedEvalCallback):
    def __init__(self, env, normalization_save_path, *args, **kwargs):
        self.last_best_mean_reward = None
        self.env = env
        self.save_path = normalization_save_path
        return super().__init__(*args, **kwargs)
        
    def _on_step(self) -> bool:
        self.last_best_mean_reward = self.best_mean_reward
        return_value = super()._on_step()
        if self.best_mean_reward != self.last_best_mean_reward:
            print('Saving normalization!')
            self.env.save(f'{self.save_path}best_model_norm.pkl')

class CheckpointCallback(BaseCallback):
    """
    Callback for saving a model every ``save_freq`` calls
    to ``env.step()``.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``

    :param save_freq:
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: Common prefix to the saved models
    :param save_vec_normalize: Bool indicating whether to save the VecNormalize statistics
    :param verbose:
    """

    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "rl_model", verbose: int = 0, save_vec_normalize: bool = False, env=None):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.save_vec_normalize = save_vec_normalize
        self.env = env

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls == 1 or self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
            self.model.save(path)
            if self.verbose > 1:
                print(f"Saving model checkpoint to {path}")
            if self.save_vec_normalize and self.env is not None:
                self.env.save(f'{path}_norm.pkl')
                if self.verbose > 1:
                    print(f"Saving normalization for checkpoint")
                
        return True