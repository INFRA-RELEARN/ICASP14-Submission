from gym import spaces

import numpy as np
from numpy import array

from ..components.gamma_process import GammaProcessComponent, HieracicalGammaProcessComponent
from .systems import System
from ..registry import register_system

class HieracicalGammaProcessSystem(System):
    def __init__(self, lam, b, delta_t, theta, mu, inspection_cost=10, repair_cost=100, 
            repair_effect=0.9, replacement_cost=500, failure_cost=1000, 
            failure_threshold=100, initial_state=0.0, time_horizon=200, 
            steps_per_observation=1, evaluation=False):
    
        self.component = HieracicalGammaProcessComponent(
            lam=lam, b=b, delta_t=delta_t, theta=theta, mu=mu, initial_state=initial_state)
        
        self.delta_t = delta_t
        self.inspection_cost = inspection_cost
        self.repair_cost = repair_cost
        self.repair_effect = repair_effect
        self.replacement_cost = replacement_cost
        self.failure_cost = failure_cost
        self.failure_threshold = failure_threshold
        self.time_horizon = time_horizon
        self.steps_per_observation = steps_per_observation
        self.evaluation = evaluation

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low = array([0, initial_state, 0], dtype=np.float32), 
            high = array([time_horizon, failure_threshold, 1], dtype=np.float32), 
            dtype=np.float32)

        self.reset()

    def _step(self, cost, info):
        self.t += self.delta_t
        
        self.component.step()

        if self.component.get_state() >= self.failure_threshold:
            info['failed'] = True
            self.component.state = 0
            self.component.t = 0 # system is rejuvenated
            cost += self.failure_cost + self.replacement_cost
        
        return cost, info


    def step(self, action):
        """
        Step the system forward by one time step while performing an action.

        Args:
            action: action to take. 0 for no action, 1 for inspection, 
                2 for repair, 3 for inspection and repair

        Returns:
            observation: 
                state of the system after the action in the format: 
                    [timestep, component_state, inspection_performed]
                if inspection was performed the component_state is up to date and the 
                    inspection_performed is 1
                else the component_state will be 0 and the inspection_performed is 0
            reward:
                negative cost as a result of the action and component state
            done:
                True if the the max number of timesteps has been reached.
                False otherwise.
            info:
                contains "failed": True if the component failed, False otherwise
        """
        cost = 0
        state = np.zeros(3, dtype=np.float32)
        info = {'failed': False}

        state[0] = self.timestep
        self.timestep += 1

        for _ in range(self.steps_per_observation):
            cost, info = self._step(cost, info)

        
        if action == 0: 
            pass
        elif action == 1: # inspect
            state[1] = self.component.get_state()
            state[2] = 1
            cost += self.inspection_cost
        elif action == 2: # repair
            self.component.state *= self.repair_effect
            self.component.state = max(self.component.state, self.component.initial_state)
            cost += self.repair_cost
        elif action == 3: # inspect and repair
            self.component.state *= self.repair_effect
            self.component.state = max(self.component.state, self.component.initial_state)
            state[1] = self.component.get_state()
            state[2] = 1
            cost += self.inspection_cost + self.repair_cost
        else:
            raise ValueError("Action must be within [0, 3]")

        done = self.t >= self.time_horizon

        return state, -cost, done, info

    def reset(self):
        """
        Reset the system to its initial state.

        Returns:
            observation
        """
        self.component.reset()
        self.t = 0
        self.timestep = 0
        return array([0, self.component.get_state(), 1], dtype=np.float32)

    def seed(self, seed=None):
        return self.component.seed(seed)


register_system('HieracicalGammaProcessSystem', HieracicalGammaProcessSystem)

class HieracicalGammaProcessSystem2(HieracicalGammaProcessSystem):
    """
    A system with a hieracical gamma process component. This version has a different
    observation space. The observation space is [timestep, last_known_state, time_since_last_inspection]
    """
    def get_observation(self):
        return array([self.timestep, self.last_component_state, self.time_since_last_inspection], dtype=np.float32)
    def step(self, action):
        """
        Step the system forward by one time step while performing an action.

        Args:
            action: action to take. 0 for no action, 1 for inspection, 
                2 for repair, 3 for inspection and repair

        Returns:
            observation: 
                state of the system after the action in the format: 
                    [timestep, last_known_state, time_since_last_inspection]
                if inspection was performed the component_state is up to date and the 
                    time_since_last_inspection is 0
                else the component_state will be the last known state and the
                    time_since_last_inspection will be the time steps since the last inspection
            reward:
                negative cost as a result of the action and component state
            done:
                True if the the max number of timesteps has been reached.
                False otherwise.
            info:
                contains "failed": True if the component failed, False otherwise
        """
        cost = 0
        state = np.zeros(3, dtype=np.float32)
        info = {'failed': False}

        state[0] = self.timestep
        self.timestep += 1
        self.time_since_last_inspection += 1

        for _ in range(self.steps_per_observation):
            cost, info = self._step(cost, info)
        
        if action == 0:
            pass
        elif action == 1: # inspect
            self.last_component_state = self.component.get_state()
            self.time_since_last_inspection = 0
            cost += self.inspection_cost
        elif action == 2: # repair
            self.component.state *= self.repair_effect
            self.component.state = max(self.component.state, self.component.initial_state)
            cost += self.repair_cost
        elif action == 3: # inspect and repair
            self.component.state *= self.repair_effect
            self.component.state = max(self.component.state, self.component.initial_state)
            self.last_component_state = self.component.get_state()
            self.time_since_last_inspection = 0
            cost += self.inspection_cost + self.repair_cost
        else:
            raise ValueError("Action must be within [0, 3]")

        done = self.t >= self.time_horizon

        if False and not self.evaluation:
            failure_margin = self.failure_threshold - self.component.state
            failure_probability = self.component.step_size_probability(failure_margin)

            cost += failure_probability * self.failure_cost

        state[1] = self.last_component_state
        state[2] = self.time_since_last_inspection

        return state, -cost, done, info

    def reset(self):
        """
        Reset the system to its initial state.

        Returns:
            observation
        """
        self.component.reset()
        self.time_since_last_inspection = 0
        self.last_component_state = self.component.get_state()
        self.t = 0
        self.timestep = 0
        return self.get_observation()

register_system('HieracicalGammaProcessSystem2', HieracicalGammaProcessSystem2)

class HieracicalGammaProcessSystem3(HieracicalGammaProcessSystem):
    """
    A system with a hieracical gamma process component. This version has the replace option. The observation space is [timestep, last_known_state, time_since_last_inspection]
    """
    def __init__(self, *args, **kw_args):
        super().__init__(*args, **kw_args)
        self.action_space = spaces.Discrete(5)


    def get_observation(self):
        return array([self.timestep, self.last_component_state, self.time_since_last_inspection], dtype=np.float32)
    def step(self, action):
        """
        Step the system forward by one time step while performing an action.

        Args:
            action: action to take. 0 for no action, 1 for inspection, 
                2 for repair, 3 for inspection and repair, 4 for replace

        Returns:
            observation: 
                state of the system after the action in the format: 
                    [timestep, last_known_state, time_since_last_inspection]
                if inspection was performed the component_state is up to date and the 
                    time_since_last_inspection is 0
                else the component_state will be the last known state and the
                    time_since_last_inspection will be the time steps since the last inspection
            reward:
                negative cost as a result of the action and component state
            done:
                True if the the max number of timesteps has been reached.
                False otherwise.
            info:
                contains "failed": True if the component failed, False otherwise
        """
        cost = 0
        state = np.zeros(3, dtype=np.float32)
        info = {'failed': False}

        state[0] = self.timestep
        self.timestep += 1
        self.time_since_last_inspection += 1

        for _ in range(self.steps_per_observation):
            cost, info = self._step(cost, info)
        
        if action == 0:
            pass
        elif action == 1: # inspect
            self.last_component_state = self.component.get_state()
            self.time_since_last_inspection = 0
            cost += self.inspection_cost
        elif action == 2: # repair
            self.component.state *= self.repair_effect
            self.component.state = max(self.component.state, self.component.initial_state)
            cost += self.repair_cost
        elif action == 3: # inspect and repair
            self.component.state *= self.repair_effect
            self.component.state = max(self.component.state, self.component.initial_state)
            self.last_component_state = self.component.get_state()
            self.time_since_last_inspection = 0
            cost += self.inspection_cost + self.repair_cost
        elif action == 4: # replace
            self.component.state = 0
            self.component.t = 0 # system is rejuvenated
            cost += self.replacement_cost
        else:
            raise ValueError("Action must be within [0, 4]")

        done = self.t >= self.time_horizon

        if False and not self.evaluation:
            failure_margin = self.failure_threshold - self.component.state
            failure_probability = self.component.step_size_probability(failure_margin)

            cost += failure_probability * self.failure_cost

        state[1] = self.last_component_state
        state[2] = self.time_since_last_inspection

        return state, -cost, done, info

    def reset(self):
        """
        Reset the system to its initial state.

        Returns:
            observation
        """
        self.component.reset()
        self.time_since_last_inspection = 0
        self.last_component_state = self.component.get_state()
        self.t = 0
        self.timestep = 0
        return self.get_observation()
    
register_system('HieracicalGammaProcessSystem3', HieracicalGammaProcessSystem3)

class HieracicalGammaProcessSystemGeneral(HieracicalGammaProcessSystem):
    """
    A system with a hieracical gamma process component. This version has a different
    observation space. The observation space is [timestep, last_known_state, time_since_last_inspection]
    """
    def get_observation(self):
        return array([self.timestep, self.last_component_state, self.time_since_last_inspection], dtype=np.float32)
    def step(self, action):
        """
        Step the system forward by one time step while performing an action.

        Args:
            action: action to take. 0 for no action, 1 for inspection, 
                2 for repair, 3 for inspection and repair

        Returns:
            observation: 
                state of the system after the action in the format: 
                    [timestep, last_known_state, time_since_last_inspection]
                if inspection was performed the component_state is up to date and the 
                    time_since_last_inspection is 0
                else the component_state will be the last known state and the
                    time_since_last_inspection will be the time steps since the last inspection
            reward:
                negative cost as a result of the action and component state
            done:
                True if the the max number of timesteps has been reached.
                False otherwise.
            info:
                contains "failed": True if the component failed, False otherwise
        """
        cost = 0
        state = np.zeros(3, dtype=np.float32)
        info = {'failed': False}

        self.timestep += 1
        self.time_since_last_inspection += 1

        for _ in range(self.steps_per_observation):
            cost, info = self._step(cost, info)
        
        if action == 0:
            pass
        elif action == 1: # inspect
            self.last_component_state = self.component.get_state()
            self.time_since_last_inspection = 0
            cost += self.inspection_cost
        elif action == 2: # repair
            self.component.state *= self.repair_effect
            self.component.state = max(self.component.state, self.component.initial_state)
            cost += self.repair_cost
        elif action == 3: # inspect and repair
            self.component.state *= self.repair_effect
            self.component.state = max(self.component.state, self.component.initial_state)
            self.last_component_state = self.component.get_state()
            self.time_since_last_inspection = 0
            cost += self.inspection_cost + self.repair_cost
        else:
            raise ValueError("Action must be within [0, 3]")

        done = self.t >= self.time_horizon

        return self.get_observation(), -cost, done, info

    def reset(self):
        """
        Reset the system to its initial state.

        Returns:
            observation
        """
        self.component.reset()
        self.time_since_last_inspection = 0
        self.last_component_state = self.component.get_state()
        self.t = 0
        self.timestep = 0
        return self.get_observation()

register_system('HieracicalGammaProcessSystemGeneral', HieracicalGammaProcessSystemGeneral)

class HieracicalGammaProcessSystemState(HieracicalGammaProcessSystemGeneral):
    def __init__(self, *args, **kw_args):
        super().__init__(*args, **kw_args)
        low = array([-1], dtype=np.float32)
        high = array([self.failure_threshold], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, shape=low.shape, dtype=np.float32)
    
    def get_observation(self):
        state = np.zeros(1, dtype=np.float32)
        if self.time_since_last_inspection == 0:
            state[0] = self.last_component_state
        else:
            state[0] = -1
        return state

register_system('HieracicalGammaProcessSystem_State', HieracicalGammaProcessSystemState)

class HieracicalGammaProcessSystemTimestepState(HieracicalGammaProcessSystemGeneral):
    def __init__(self, *args, **kw_args):
        super().__init__(*args, **kw_args)
        low = array([0, -1], dtype=np.float32)
        high = array([self.time_horizon / self.delta_t, self.failure_threshold], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, shape=low.shape, dtype=np.float32)

    def get_observation(self):
        state = np.zeros(2, dtype=np.float32)
        state[0] = self.timestep
        if self.time_since_last_inspection == 0:
            state[1] = self.last_component_state
        else:
            state[1] = -1
        return state
            
register_system('HieracicalGammaProcessSystem_TimestepState', HieracicalGammaProcessSystemTimestepState)

class HieracicalGammaProcessSystemLastStateStepsSice(HieracicalGammaProcessSystemGeneral):
    def __init__(self, *args, **kw_args):
        super().__init__(*args, **kw_args)
        low = array([0, 0], dtype=np.float32)
        high = array([self.failure_threshold, self.time_horizon / self.delta_t], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, shape=low.shape, dtype=np.float32)

    def get_observation(self):
        state = np.zeros(2, dtype=np.float32)
        state[0] = self.last_component_state
        state[1] = self.time_since_last_inspection
        return state

register_system('HieracicalGammaProcessSystem_LastStateStepsSice', HieracicalGammaProcessSystemLastStateStepsSice)

class HieracicalGammaProcessSystemTimestepLastStateStepsSice(HieracicalGammaProcessSystemGeneral):
    def __init__(self, *args, **kw_args):
        super().__init__(*args, **kw_args)
        low = array([0, 0, 0], dtype=np.float32)
        high = array([self.time_horizon / self.delta_t, self.failure_threshold, self.time_horizon / self.delta_t], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, shape=low.shape, dtype=np.float32)

    def get_observation(self):
        state = np.zeros(3, dtype=np.float32)
        state[0] = self.timestep
        state[1] = self.last_component_state
        state[2] = self.time_since_last_inspection
        return state

register_system('HieracicalGammaProcessSystem_TimestepLastStateStepsSice', HieracicalGammaProcessSystemTimestepLastStateStepsSice)

class HieracicalGammaProcessSystemHistory(HieracicalGammaProcessSystem):
    """
    A system with a hieracical gamma process component. This version has a different
    observation space. Observations are the last n known inspection states and times, last n repair times
    """
    n_history = 2

    def __init__(self, *args, **kw_args):
        super().__init__(*args, **kw_args)
        low = array(np.zeros(3*self.n_history ), dtype=np.float32)
        high = array(
            [ self.failure_threshold ] * self.n_history +
            [ self.time_horizon / self.delta_t] * 2 * self.n_history
            , dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, shape=low.shape, dtype=np.float32)

   
    def get_observation(self):
        # last n known inspection states and times, last n repair times
        n = self.n_history
        state = np.zeros(3 * n , dtype=np.float32)

        # inspection states
        for i, last_inspection_state in enumerate(self.last_inspection_states[-n:][::-1]):
            state[i] = last_inspection_state

        # inspection times
        for i, last_inspection_time in enumerate(self.last_inspection_times[-n:][::-1]):
            state[n + i] = self.timestep - last_inspection_time

        # repair times
        for i, last_repair_time in enumerate(self.last_repair_times[-n:][::-1]):
            state[2*n + i] = self.timestep - last_repair_time

        
        return state

    def _step(self, cost, info):
        self.t += self.delta_t
        
        self.component.step()

        if self.component.get_state() >= self.failure_threshold:
            info['failed'] = True
            self.component.state = 0
            self.component.t = 0 # system is rejuvenated
            cost += self.failure_cost + self.replacement_cost
            self.last_inspection_times.append(self.timestep)
            self.last_inspection_times = self.last_inspection_times[-self.n_history:]
            self.last_inspection_states.append(self.component.get_state())
            self.last_inspection_states = self.last_inspection_states[-self.n_history:]
        
        return cost, info

    def _inspect(self, cost, info):
        self.last_inspection_states.append(self.component.get_state())
        self.last_inspection_states = self.last_inspection_states[-self.n_history:]
        self.last_inspection_times.append(self.timestep)
        self.last_inspection_times = self.last_inspection_times[-self.n_history:]
        cost += self.inspection_cost
        return cost, info

    def _repair(self, cost, info):
        self.component.state *= self.repair_effect
        self.component.state = max(self.component.state, self.component.initial_state)
        self.last_repair_times.append(self.timestep)
        self.last_repair_times = self.last_repair_times[-self.n_history:]
        cost += self.repair_cost
        return cost, info

    def step(self, action):
        """
        Step the system forward by one time step while performing an action.

        Args:
            action: action to take. 0 for no action, 1 for inspection, 
                2 for repair, 3 for inspection and repair

        Returns:
            observation: 
                state of the system after the action in the format: 
                    [timestep, last_known_state, time_since_last_inspection]
                if inspection was performed the component_state is up to date and the 
                    time_since_last_inspection is 0
                else the component_state will be the last known state and the
                    time_since_last_inspection will be the time steps since the last inspection
            reward:
                negative cost as a result of the action and component state
            done:
                True if the the max number of timesteps has been reached.
                False otherwise.
            info:
                contains "failed": True if the component failed, False otherwise
        """
        cost = 0
        info = {'failed': False}

        self.timestep += 1

        for _ in range(self.steps_per_observation):
            cost, info = self._step(cost, info) 
        
        if not info['failed']:
            if action == 0: # do nothing
                pass
            elif action == 1: # inspect
                cost, info = self._inspect(cost, info)

            elif action == 2: # repair
                cost, info = self._repair(cost, info)
                
            elif action == 3: # inspect and repair
                cost, info = self._repair(cost, info)
                cost, info = self._inspect(cost, info)
            else:
                raise ValueError("Action must be within [0, 3]")

        done = self.t >= self.time_horizon

        return self.get_observation(), -cost, done, info

    def reset(self):
        """
        Reset the system to its initial state.

        Returns:
            observation
        """
        self.component.reset()
        self.last_inspection_times = [0]
        self.last_repair_times = []
        self.last_inspection_states = [0]
        self.t = 0
        self.timestep = 0
        return self.get_observation()

class HieracicalGammaProcessSystemHistory1(HieracicalGammaProcessSystemHistory):
    n_history = 1

register_system('HieracicalGammaProcessSystemHistory1', HieracicalGammaProcessSystemHistory1)

class HieracicalGammaProcessSystemHistory2(HieracicalGammaProcessSystemHistory):
    n_history = 2

register_system('HieracicalGammaProcessSystemHistory2', HieracicalGammaProcessSystemHistory2)

class HieracicalGammaProcessSystemHistory3(HieracicalGammaProcessSystemHistory):
    n_history = 3

register_system('HieracicalGammaProcessSystemHistory3', HieracicalGammaProcessSystemHistory3)
    

