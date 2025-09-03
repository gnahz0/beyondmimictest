from .base import BasicSdk2Bridge, ElasticBand
from .unitree import UnitreeSdk2Bridge
from .booster import BoosterSdk2Bridge


def create_sdk2py_bridge(mj_model, mj_data, robot_config):
    """
    Factory function to create the appropriate SDK2Py bridge based on configuration.
    
    Args:
        mj_model: MuJoCo model
        mj_data: MuJoCo data
        robot_config: Robot configuration dictionary
    
    Returns:
        An instance of the appropriate bridge class
    """
    sdk_type = robot_config.get("SDK_TYPE", "unitree")
    
    if sdk_type == "unitree":
        return UnitreeSdk2Bridge(mj_model, mj_data, robot_config)
    elif sdk_type == "booster":
        return BoosterSdk2Bridge(mj_model, mj_data, robot_config)
    else:
        raise ValueError(f"Unsupported SDK type: {sdk_type}")


__all__ = [
    "BasicSdk2Bridge",
    "UnitreeSdk2Bridge", 
    "BoosterSdk2Bridge",
    "ElasticBand",
    "create_sdk2py_bridge",
] 