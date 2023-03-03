import wandb
from wandb import AlertLevel
from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple, Union
from stable_baselines3.common.logger import KVWriter, filter_excluded_keys, FormatUnsupportedError, Video, Figure, Image

class WandBLogger(KVWriter):
    def __init__(self, log_max_params=None, alert_callbacks=[]) -> None:
        self.total_steps = 0
        self.last_steps = 0
        self.log_max_params = log_max_params
        self.max_params = {}
        self.alert_callbacks = alert_callbacks
        pass

    def write(self, key_values: Dict[str, Any], key_excluded: Dict[str, Union[str, Tuple[str, ...]]], step: int = 0) -> None:
        """
        Write a dictionary to file

        :param key_values:
        :param key_excluded:
        :param step:
        """
        key_values = filter_excluded_keys(key_values, key_excluded, "wandb")

        step_delta = step - self.last_steps
        if step_delta < 0:
            self.total_steps += step
        else:
            self.total_steps += step_delta
        self.last_steps = step

        log_values = key_values.copy()
        
        if self.log_max_params is not None:
            key_values = {k: v for k, v in key_values.items() if k in self.log_max_params}
            for k, v in key_values.items():
                if k not in self.max_params:
                    self.max_params[k] = v
                else:
                    self.max_params[k] = max(self.max_params[k], v)
            max_param_dict = {f'{k}_max': v for k, v in self.max_params.items()}
            log_values.update(max_param_dict)

        if self.alert_callbacks is not None:
            for callback in self.alert_callbacks:
                trigger, message, title = callback(key_values, step=self.total_steps)
                if trigger:
                    wandb.alert(title=title,text=message, level=AlertLevel.INFO)

        wandb.log(log_values, step=self.total_steps)

    def close(self) -> None:
        """
        Close owned resources
        """
        pass
    