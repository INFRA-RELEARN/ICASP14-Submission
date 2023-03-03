from .components import Component
from ..registry import register_component

import numpy as np
from scipy.stats import gamma


class GammaProcessComponent(Component):
    """
    Component for a  gamma process.
    """
    def __init__(self, k, lam, initial_state=0.0):
        """
        Args:
            k: shape parameter
            lam: scale parameter
            initial_state: initial state of the component upon reset
        """
        super().__init__()
        self.k = k
        self.lam = lam
        self.initial_state = initial_state
        self.reset()

    def reset(self):
        self.state = self.initial_state
        return self.state

    def step(self):
        self.state += self.np_random.gamma(self.k, 1/self.lam) 
        return self.state

    def get_state(self):
        return self.state
    
    def failed(self) -> bool:
        return False

    def step_size_probability(self, step_size):
        return 1.0 - gamma(a=self.k, scale=1/self.lam).cdf(step_size)

register_component('gamma_process', GammaProcessComponent)

class HieracicalGammaProcessComponent(GammaProcessComponent):
    def __init__(self, lam, b, delta_t, theta, mu, initial_state=0):
        self.b = b
        self.delta_t = delta_t
        self.theta = theta
        self.mu = mu
        super().__init__(k=0, lam=lam, initial_state=initial_state)
        self.reset()

    def _update_parameters(self):
        m_t = (self.t+self.delta_t)**self.b
        m_t_1 = self.t ** self.b

        self.k = (m_t - m_t_1) * self.a * self.lam

        return self.k, self.lam

    def step(self):
        self.t += self.delta_t
        self._update_parameters()

        delta_d = self.np_random.gamma(self.k, 1.0 / self.lam)
        self.state += delta_d
        
        return self.state

    def reset(self):
        self.t = 0
        self.a = self.np_random.lognormal(self.mu,self.theta)
        return super().reset()
    
register_component('gamma_process_hieracical', HieracicalGammaProcessComponent, )
