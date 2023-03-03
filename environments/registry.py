class RegistryEntry():
    """A registry entry"""
    def __init__(self, id, cls, name=None, description=None, kwparams={}):
        self.id = id
        self.cls = cls
        self.name = name or cls.__name__
        self.description = description or cls.__doc__
        self.kwparams = kwparams
        self.type = None


class ComponentRegistryEntry(RegistryEntry):
    """Class to hold information about a component class"""
    def __init__(self, id, cls, name=None, description=None, kwparams={}):
        super().__init__(id, cls, name, description, kwparams)
        self.type = 'component'

class SystemRegistryEntry(RegistryEntry):
    """Class to hold information about a system class"""
    def __init__(self, id, cls, name=None, description=None, kwparams={}):
        super().__init__(id, cls, name, description, kwparams)
        self.type = 'system'

class Registry():
    '''Class for registering components and systems.'''
    def __init__(self):
        self.registry = {}
        self.disable_reregister_warnings_flag = False

    def __getitem__(self, id):
        return self.get(id)

    def get(self, id):
        if id in self.registry:
            return self.registry[id]
        else:
            raise KeyError(f'Component with id {id} not registered.')

    def ids(self):
        return self.registry.keys()

    def entries(self):
        return self.registry.items()

    def register(self, id, entry, overwrite=False):
        if self.registry.get(id) is None or overwrite:
            self.registry[id] = entry
        else:
            if not self.disable_reregister_warnings_flag:
                raise KeyError(f'Component with id {id} already registered.')

    def disable_reregister_warnings(self):
        self.disable_reregister_warnings_flag = True

registry = Registry()

def register_component(id, cls, name=None, description=None, kwparams={}):
    registry.register(id, ComponentRegistryEntry(id, cls, name, description, kwparams))

def register_system(id, cls, name=None, description=None, kwparams={}):
    registry.register(id, SystemRegistryEntry(id, cls, name, description, kwparams))

def get_registry_entry(id):
    return registry[id]

def disable_reregister_warnings():
    """
    Disable warnings when registering components that are already registered.
    Useful in case of module reloads in Jupyter notebooks.
    """
    registry.disable_reregister_warnings()
