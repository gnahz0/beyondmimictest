"""Environment registry system for vuer-sim-example."""

from typing import Dict, Type, Callable, Any, Optional
import importlib


class EnvSpec:
    """Specification for an environment."""
    
    def __init__(
        self,
        id: str,
        entry_point: str,
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.id = id
        self.entry_point = entry_point
        self.kwargs = kwargs or {}
    
    def make(self, **kwargs) -> Any:
        """Create an instance of the environment."""
        # Merge spec kwargs with provided kwargs (provided kwargs override)
        env_kwargs = {**self.kwargs, **kwargs}
        
        # Parse entry point
        module_name, class_name = self.entry_point.rsplit(":", 1)
        
        # Import module and get class
        module = importlib.import_module(module_name)
        env_class = getattr(module, class_name)
        
        # Create and return environment instance
        return env_class(**env_kwargs)


class Registry:
    """Registry for environments."""
    
    def __init__(self):
        self.env_specs: Dict[str, EnvSpec] = {}
    
    def register(
        self,
        id: str,
        entry_point: str,
        **kwargs
    ):
        """Register a new environment."""
        if id in self.env_specs:
            raise ValueError(f"Environment '{id}' is already registered")
        
        self.env_specs[id] = EnvSpec(
            id=id,
            entry_point=entry_point,
            kwargs=kwargs
        )
    
    def make(self, id: str, **kwargs) -> Any:
        """Create an environment instance."""
        if id not in self.env_specs:
            raise ValueError(
                f"Environment '{id}' not found. Available environments: {list(self.env_specs.keys())}"
            )
        
        return self.env_specs[id].make(**kwargs)
    
    def list(self) -> list:
        """List all registered environments."""
        return list(self.env_specs.keys())


# Global registry instance
registry = Registry()


def register(id: str, entry_point: str, **kwargs):
    """Register an environment in the global registry."""
    registry.register(id, entry_point, **kwargs)


def make(id: str, **kwargs):
    """Create an environment from the global registry."""
    return registry.make(id, **kwargs)


def list_envs():
    """List all registered environments."""
    return registry.list()