import abc

from gym import Env

from ..components.components import Component

class System(Component, Env):
    '''Abstract base class for all systems.'''
    def __init__(self) -> None:
        super().__init__()

    @abc.abstractproperty
    def step(self, action):
        pass

    @abc.abstractproperty
    def reset(self):
        pass

    @abc.abstractproperty
    def get_state(self):
        pass

    @abc.abstractproperty
    def failed(self) -> bool:
        pass

    def set_attribute(self, attribute, value):
        self.__dict__[attribute] = value
