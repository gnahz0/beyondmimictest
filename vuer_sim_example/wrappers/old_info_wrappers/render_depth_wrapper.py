import numpy as np
from vuer_sim_example.wrappers.movable_camera_wrapper import MovableCameraWrapper


class RenderDepthWrapper(MovableCameraWrapper):
    """
    Renders normalized linear depth. This wrapper does NOT apply inversion as in RenderDepthWrapper.
    Output is normalized wrt the specified near and far.
    """

    def __init__(
        self,
        env,
        *,
        width=1280,
        height=768,
        camera_id="ego_rgb-render",
        near=0.0,
        far=5.0,
        **_,
    ):
        super().__init__(env)

        self.env = env
        self.width = width
        self.height = height
        self.camera_id = camera_id
        self.near = near
        self.far = far

        self.camera = None


    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if "depth_frame" in info:
            frame = info["depth_frame"]
        else:
            frame = self.render(
                "depth",
                width=self.width,
                height=self.height,
                camera_id=self.camera_id,
            )
            info["depth_frame"] = frame

        for c_id in frame:
            # clip
            frame[c_id] = np.clip(frame[c_id], self.near, self.far)

            # normalize
            frame[c_id] = (frame[c_id] - self.near) / (self.far - self.near)

            frame[c_id] = (frame[c_id] * 255).astype(np.uint8)

            if len(frame[c_id].shape) == 2:
                frame[c_id] = frame[:, :, None]

        info["render_depth"] = frame
        return obs, rew, done, info
