from dataclasses import dataclass, fields

import numpy as np


@dataclass
class Frame:
    qpos: np.ndarray
    qvel: np.ndarray
    mocap_pos: np.ndarray
    mocap_quat: np.ndarray
    site_xpos: np.ndarray
    site_xmat: np.ndarray
    act: np.ndarray = None
    ctrl: np.ndarray = None

    sensordata: np.ndarray = None
    front_camera_rgb: np.ndarray = None

    # camera_pos: np.ndarray
    # camera_quat: np.ndarray
    # image: np.ndarray
    # depth: np.ndarray
    # state: np.ndarray
    # action: np.ndarray


FRAME_KEYS = ", ".join([field.name for field in fields(Frame)])
