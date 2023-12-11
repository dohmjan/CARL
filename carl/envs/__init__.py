# flake8: noqa: F401
# Modular imports
import importlib.util as iutil
import warnings

# Classic control is in gym and thus necessary for the base version to run
from carl import envs
from carl.envs.gymnasium import *

__all__ = envs.gymnasium.__all__


def check_spec(spec_name: str) -> bool:
    """Check if the spec is installed

    Parameters
    ----------
    spec_name : str
        Name of package that is necessary for the environment suite.

    Returns
    -------
    bool
        Whether the spec was found.
    """
    spec = iutil.find_spec(spec_name)
    found = spec is not None
    if not found:
        with warnings.catch_warnings():
            warnings.simplefilter("once")
            warnings.warn(
                f"Module {spec_name} not found. If you want to use these environments, please follow the installation guide."
            )
    return found


# Environment loading
found = check_spec("Box2D")
if found:
    from carl.envs.gymnasium.box2d import *

    __all__ += envs.gymnasium.box2d.__all__

found = check_spec("brax")
if found:
    from carl.envs.brax import *

    __all__ += envs.brax.__all__

found = check_spec("py4j")
if found:
    from carl.envs.mario import *

    __all__ += envs.mario.__all__

found = check_spec("dm_control")
if found:
    from carl.envs.dmc import *

    __all__ += envs.dmc.__all__

found = check_spec("distance")
if found:
    from carl.envs.rna import *

    __all__ += envs.rna.__all__
