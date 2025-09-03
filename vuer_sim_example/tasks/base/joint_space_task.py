from copy import deepcopy

import numpy as np

from vuer_sim_example.schemas.se3.rot_gs6 import mat2gs6
from vuer_sim_example.tasks.base.lucidxr_task import PlaybackTask, SimplePhysics, _mj_physics_to_dict


class JointSpaceTask(PlaybackTask):
    """
    Base class for Mocap tasks.

    Change the Action space to End-Effector trajectories by commanding mocap points
    """

    def __init__(self, random=None):
        super().__init__(random=random)

    # fall-back to default action spec, that just uses the control definition.
    # def action_spec(self, physics: SimplePhysics):
    #     """
    #     Returns a `BoundedArraySpec` matching the End-Effector pose
    #     See dm_control/mujoco/engine.py
    #     """
    #     # hard-code for now, all these need to change somehow.
    #     raise NotImplementedError("control is well-defined, actuation is not.")
    #
    #     frame = _mj_physics_to_dict(physics)
    #     act = self._get_prev_action(**frame)
    #
    #     return specs.BoundedArray(shape=act.shape, dtype=float, minimum=, maximum=)

    @staticmethod
    def _get_prev_action(ctrl, **_):
        """Converts a physics.data frame into actions"""
        # todo: need to support multiple mocap points
        return deepcopy(ctrl)
        # action = np.concatenate([act, ctrl], axis=-1)
        # return action

    def before_step(self, action, physics: SimplePhysics):
        """
        Commands the actuators to the right pose.

        note: the "action" is 1:1 mapped to the control input.

        :param action: consists of [ actuators, gripper_ctrl ]
        :param physics:
        :return: nothing
        """
        physics.set_mujoco_data(ctrl=action)

    @staticmethod
    def _get_obs(physics, *, site_xpos, site_xmat, act, ctrl, **_):
        """This is slightly different from the default _get_obs
        in that we specifically return the gripper site pose, encoded
        via the SO3 gs6 representation.
        """
        # Ge: this is hard coded for now.
        gripper_site_xpos = site_xpos[-2]
        gripper_site_xmat = site_xmat[-2]

        gripper_site_gs6 = mat2gs6(gripper_site_xmat)

        obs = np.concatenate(
            [
                gripper_site_xpos,
                gripper_site_gs6,
                act,
                ctrl,
                # todo: add contact forces here. Use first as the success plate
                #   use the last two as gripper forces.
            ]
        )
        return obs
