import abc

from gym.utils import seeding


class Component:
    '''Abstract base class for all components.'''
    def __init__(self) -> None:
        self.np_random, _ = seeding.np_random()

    @abc.abstractproperty
    def step(self):
        pass

    @abc.abstractproperty
    def reset(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @abc.abstractproperty
    def get_state(self):
        pass
    
    @abc.abstractproperty
    def failed(self) -> bool:
        pass