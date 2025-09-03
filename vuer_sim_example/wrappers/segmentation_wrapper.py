from typing import List

from matplotlib import pyplot as plt

from vuer_sim_example.wrappers.camera_wrapper import CameraWrapper
from vuer_sim_example.wrappers.utils.depth_util import invisibility
import numpy as np


class SegmentationWrapper(CameraWrapper):
    """Need to consider getting a floating point depth render wrapper, too.
    """
    def __init__(self, env, *, invisible_prefix: List = None, **kwargs):
        super().__init__(env, **kwargs)

        model = self.unwrapped.env.physics.model
        all_geom_names = [model.geom(i).name for i in range(model.ngeom)]

        self.invisible_objects = []
        if invisible_prefix is not None:
            for prefix in invisible_prefix:
                self.invisible_objects.extend([geom_name for geom_name in all_geom_names if geom_name.startswith(prefix)])

        self.colors = np.random.randint(0, 255, (50, 3), dtype=np.uint8) #TODO Hardcoded

    def _compute_additional_obs(self, obs=None):
        physics = self.unwrapped.env.physics
        with invisibility(physics, self.invisible_objects):
            frame = self.render(
                segmentation=True,
                width=self.width,
                height=self.height,
                camera_id=self.camera_id,
        )
        segmentation = frame[..., 0].astype(np.int8)
        segmentation[segmentation == 255] = -1
        segmentation_image_uint8 = self.colors[segmentation]
        return {self.image_key: segmentation_image_uint8}

