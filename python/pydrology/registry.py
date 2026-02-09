"""Model registry for pydrology hydrological models.

Provides a global registry for hydrological models, enabling dynamic model
discovery and runtime model selection. Each registered model must follow
the standard model contract with required exports.

Required model exports:
    - PARAM_NAMES: tuple[str, ...] - Parameter names in order
    - DEFAULT_BOUNDS: tuple[tuple[float, float], ...] - (min, max) bounds per parameter
    - STATE_SIZE: int - Number of elements in state array representation
    - Parameters: class - Dataclass with from_array() classmethod and __array__() method
    - State: class - Dataclass with from_array() classmethod and __array__() method
    - run: function - Execute model over timeseries
    - step: function - Execute single timestep

"""

import logging
from types import ModuleType

from pydrology.types import Resolution

logger = logging.getLogger(__name__)

# Required module-level exports
_REQUIRED_EXPORTS: tuple[str, ...] = (
    "PARAM_NAMES",
    "DEFAULT_BOUNDS",
    "STATE_SIZE",
    "Parameters",
    "State",
    "run",
    "step",
    "SUPPORTED_RESOLUTIONS",
)

# Required methods for Parameters class
_REQUIRED_PARAMETERS_METHODS: tuple[str, ...] = ("from_array", "__array__")

# Required methods for State class
_REQUIRED_STATE_METHODS: tuple[str, ...] = ("from_array", "__array__")

# Global registry: {name: module}
_models: dict[str, ModuleType] = {}


def _validate_module(name: str, module: ModuleType) -> None:
    """Validate that a module has all required exports.

    Args:
        name: The model name being registered.
        module: The module to validate.

    Raises:
        ValueError: If the module is missing required exports or methods.
    """
    # Check required exports
    missing_exports: list[str] = []
    for export in _REQUIRED_EXPORTS:
        if not hasattr(module, export):
            missing_exports.append(export)

    if missing_exports:
        missing_str = ", ".join(missing_exports)
        msg = f"Model '{name}' is missing required exports: {missing_str}"
        raise ValueError(msg)

    # Check Parameters class has required methods
    parameters_cls = module.Parameters
    missing_param_methods: list[str] = []
    for method in _REQUIRED_PARAMETERS_METHODS:
        if not hasattr(parameters_cls, method):
            missing_param_methods.append(method)

    if missing_param_methods:
        missing_str = ", ".join(missing_param_methods)
        msg = f"Model '{name}' Parameters class is missing required methods: {missing_str}"
        raise ValueError(msg)

    # Check State class has required methods
    state_cls = module.State
    missing_state_methods: list[str] = []
    for method in _REQUIRED_STATE_METHODS:
        if not hasattr(state_cls, method):
            missing_state_methods.append(method)

    if missing_state_methods:
        missing_str = ", ".join(missing_state_methods)
        msg = f"Model '{name}' State class is missing required methods: {missing_str}"
        raise ValueError(msg)

    # Check SUPPORTED_RESOLUTIONS is valid
    resolutions = module.SUPPORTED_RESOLUTIONS
    if not isinstance(resolutions, tuple):
        msg = f"Model '{name}' SUPPORTED_RESOLUTIONS must be a tuple"
        raise ValueError(msg)
    if not all(isinstance(r, Resolution) for r in resolutions):
        msg = f"Model '{name}' SUPPORTED_RESOLUTIONS must contain only Resolution enum values"
        raise ValueError(msg)


def register(name: str, module: ModuleType) -> None:
    """Register a model module under the given name.

    The module must have the following exports:
        - PARAM_NAMES: tuple[str, ...] - Parameter names in order
        - DEFAULT_BOUNDS: tuple[tuple[float, float], ...] - (min, max) bounds per parameter
        - STATE_SIZE: int - Number of elements in state array representation
        - Parameters: class with from_array() classmethod and __array__() method
        - State: class with from_array() classmethod and __array__() method
        - run: function - Execute model over timeseries
        - step: function - Execute single timestep

    Args:
        name: The name to register the model under (e.g., "gr6j", "cemaneige").
        module: The model module containing required exports.

    Raises:
        ValueError: If the module is missing required exports or methods.

    Example:
        >>> from pydrology import registry
        >>> from pydrology.models import gr6j
        >>> registry.register("gr6j", gr6j)
    """
    _validate_module(name, module)
    _models[name] = module
    logger.debug("Registered model '%s'", name)


def get_model(name: str) -> ModuleType:
    """Get a registered model module by name.

    Args:
        name: The registered model name.

    Returns:
        The model module.

    Raises:
        KeyError: If the model name is not registered.

    Example:
        >>> model = registry.get_model("gr6j")
        >>> result = model.run(params, forcing)
    """
    if name not in _models:
        available = ", ".join(sorted(_models.keys())) if _models else "(none)"
        msg = f"Unknown model '{name}'. Available models: {available}"
        raise KeyError(msg)
    return _models[name]


def list_models() -> list[str]:
    """Return sorted list of registered model names.

    Returns:
        Sorted list of model names that have been registered.

    Example:
        >>> registry.list_models()
        ['cemaneige', 'gr6j']
    """
    return sorted(_models.keys())


def get_model_info(name: str) -> dict[str, object]:
    """Get metadata about a registered model.

    Args:
        name: The registered model name.

    Returns:
        Dictionary containing:
            - param_names: tuple[str, ...] - Parameter names in order
            - default_bounds: tuple[tuple[float, float], ...] - Parameter bounds
            - state_size: int - Number of state elements

    Raises:
        KeyError: If the model name is not registered.

    Example:
        >>> info = registry.get_model_info("gr6j")
        >>> info["param_names"]
        ('x1', 'x2', 'x3', 'x4', 'x5', 'x6')
    """
    module = get_model(name)

    return {
        "param_names": module.PARAM_NAMES,
        "default_bounds": module.DEFAULT_BOUNDS,
        "state_size": module.STATE_SIZE,
        "supported_resolutions": module.SUPPORTED_RESOLUTIONS,
    }
