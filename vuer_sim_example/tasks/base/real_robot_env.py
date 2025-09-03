import numpy as np
# from autolab_core import RigidTransform
# from vuer_envs_real.robot_envs import DualUR5
# from vuer_envs_real.robot_envs import SingleUR5Env
from scipy.spatial.transform import Rotation as R
from vuer_sim_example.schemas.se3.rot_gs6 import gs62quat, mat2gs6, quat2gs6

class RealRobotEnv:
    R_zplus90 = np.array([[0.0, -1.0, 0.0],
                          [1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0]])
    mujoco_home = np.array([0, 0, 0.77, 0, 0, 1, 0])
    robot_home = np.array([0, 0, 0, 0, 0, 1, 0])
    rotated_robot_home = R_zplus90 @ R_zplus90 @ robot_home[:3]
    offset_xyz = mujoco_home[:3] - rotated_robot_home

    def __init__(self, robot, max_steps: int = 1000, mujoco: bool = False):
        self.robot = robot
        self.max_steps = max_steps
        self.mujoco = mujoco
        self.step_counter = 0


        # offset_xyz = mujoco_home[:3] - robot_home[:3]

        # if isinstance(robot, DualUR5):
        #     self.n_arms = 2
        #     self._move  = self._step_dual
        #     self._obs   = self._obs_dual
        #     self.prev_action = np.zeros(16, dtype=np.float32)
        # elif isinstance(robot, SingleUR5Env):
        #     self.n_arms = 1
        #     self._move  = self._step_single
        #     self._obs   = self._obs_single
        #     self.prev_action = np.zeros(8, dtype=np.float32)
        # else:
        #     raise TypeError(
        #         f"RealRobotEnv expects DualUR5 or SingleUR5Env, got {type(robot)}"
        #     )

    # ──────────────────────────────────────────────
    # Public gym-style API
    # ──────────────────────────────────────────────
    def reset(self):
        """Send robot to its home pose and return the initial observation."""
        self.robot.set_home(asyn=False)
        self.step_counter = 0
        return self._obs()

    def step(self, action: np.ndarray):
        """Take one environment step."""
        self._move(action)                        # motion + gripper commands
        obs = self._obs()
        rew = self.get_reward(obs)
        done = self.is_done(obs)
        info = {}

        self.prev_action = action.copy()
        self.step_counter += 1
        return obs, rew, done, info

    # ──────────────────────────────────────────────
    # Convenience wrappers
    # ──────────────────────────────────────────────
    def get_obs(self):
        """Always returns the current observation (single- or dual-arm)."""
        return self._obs()

    def get_ordi(self):
        """For code that expects (obs, reward, done, info) without acting."""
        obs = self._obs()
        return obs, self.get_reward(obs), self.is_done(obs), {}

    def get_prev_action(self):
        return self.prev_action

    # ──────────────────────────────────────────────
    # Dual-arm internals
    # ──────────────────────────────────────────────
    def _step_dual(self, a):
        assert a.shape == (16,), "16-D action required for DualUR5."
        l_pos = a[0:3]
        l_quat = a[3:7] / np.linalg.norm(a[3:7])
        r_pos = a[7:10]
        r_quat = a[10:14] / np.linalg.norm(a[10:14])
        l_pose = RigidTransform(translation=l_pos, rotation=l_quat)
        r_pose = RigidTransform(translation=r_pos, rotation=r_quat)
        l_g, r_g = a[14], a[15]

        self.robot.move_pose_dual(l_pose, r_pose, asyn=False)
        self.robot.move_gripper_left(l_g)
        self.robot.move_gripper_right(r_g)

    def _obs_dual(self):
        l_pose = self.robot.get_pose_left()
        r_pose = self.robot.get_pose_right()
        l_g, r_g = self.robot.get_gripper_state()

        return {
            "left_arm_pose":  np.concatenate([l_pose.translation,
                                              l_pose.quaternion]),
            "right_arm_pose": np.concatenate([r_pose.translation,
                                              r_pose.quaternion]),
            "left_grip":  np.array([l_g], dtype=np.float32),
            "right_grip": np.array([r_g], dtype=np.float32),
            "state": np.concatenate([
                        l_pose.translation, l_pose.quaternion,
                        r_pose.translation, r_pose.quaternion,
                        [l_g, r_g]
                     ], dtype=np.float32),
        }

    # ──────────────────────────────────────────────
    # Single-arm internals
    # ──────────────────────────────────────────────
    def mujoco_to_robot_pose(self, mj_pose):
        """
        Inverts robot_to_mujoco_pose.
        mj_pose: [x_mj, y_mj, z_mj, w, x, y, z]  (MuJoCo frame, wxyz order)
        returns: [x_rb, y_rb, z_rb, w, x, y, z]  (Robot frame, wxyz order)
        """
        R_zminus90 = np.array([
            [ 0.,  1.,  0.],
            [-1.,  0.,  0.],
            [ 0.,  0.,  1.]
        ])
        Q_zminus90_scipy = R.from_euler('z', -90, degrees=True)

        # 1) Position
        pos_mj = mj_pose[:3]

        # Reverse offset
        pos_rotated = pos_mj - self.offset_xyz

        # Undo the two +90° Z rotations => apply two -90° Z rotations
        pos_rb = R_zminus90 @ R_zminus90 @ pos_rotated

        # 2) Orientation
        q_mj_wxyz = mj_pose[3:]
        q_mj_wxyz /= np.linalg.norm(q_mj_wxyz)

        # Convert to [x, y, z, w] for scipy
        q_mj_xyzw = np.array([q_mj_wxyz[1], q_mj_wxyz[2], q_mj_wxyz[3], q_mj_wxyz[0]])
        R_mj_obj = R.from_quat(q_mj_xyzw)

        # Inverse of: R_mj = Q_zplus90 * Q_zplus90 * R_rb * R_z90
        # So:         R_rb = Q_zminus90 * Q_zminus90 * R_mj_obj * R_zminus90
        R_rb = Q_zminus90_scipy * Q_zminus90_scipy * R_mj_obj * R.from_euler("z", -90, degrees=True)

        # Convert back to [w, x, y, z]
        q_rb_xyzw = R_rb.as_quat()
        q_rb_wxyz = np.array([q_rb_xyzw[3], q_rb_xyzw[0], q_rb_xyzw[1], q_rb_xyzw[2]])

        return np.concatenate([pos_rb, q_rb_wxyz])

    @classmethod
    def robot_to_mujoco_pose(cls, rb_pose):
        """
        rb_pose: [x, y, z, w, rx, ry, rz]
                (translation + quaternion in wxyz)
        returns same shape [x_mj, y_mj, z_mj, w_mj, rx_mj, ry_mj, rz_mj]
        """
        R_zplus90 = np.array([
            [ 0., -1.,  0.],
            [ 1.,  0.,  0.],
            [ 0.,  0.,  1.]
        ])
        Q_zplus90_scipy = R.from_euler('z', 90, degrees=True)  # a 90° about z

        # 1) Position
        pos_rb = rb_pose[:3]
        pos_mj = R_zplus90 @ R_zplus90 @ pos_rb
        pos_mj = pos_mj + cls.offset_xyz

        # 2) Orientation
        q_rb = rb_pose[3:]
        q_rb /= np.linalg.norm(q_rb)  # safeguard

        q_rb_xyzw = np.array([q_rb[1], q_rb[2], q_rb[3], q_rb[0]])  # => x,y,z,w
        R_mj_obj = R.from_quat(q_rb_xyzw)

        R_mj_obj = Q_zplus90_scipy * Q_zplus90_scipy * R_mj_obj * R.from_euler("z", 90, degrees=True)

        # Convert back to [w, x, y, z]
        q_mj_xyzw = R_mj_obj.as_quat()  # => [x, y, z, w]
        q_mj_wxyz = np.array([q_mj_xyzw[3], q_mj_xyzw[0], q_mj_xyzw[1], q_mj_xyzw[2]])

        return np.concatenate([pos_mj, q_mj_wxyz])

    def _step_single(self, a):
        if self.mujoco:
            assert a.shape == (10, ), "10-D action required for SingleUR5."
            a = np.concatenate([a[:3], gs62quat(a[3: 9]), [a[9]]])
        else:
            assert a.shape == (8,), "8-D action required for SingleUR5."
        if self.mujoco:
            a[0:7] = self.mujoco_to_robot_pose(a[0:7])

        pos = a[0:3]
        quat = a[3:7]/ np.linalg.norm(a[3:7])
        pose = RigidTransform(translation=pos, rotation=quat)
        g = a[7]

        self.robot.servo_pose(pose)
        self.robot.move_gripper(g)

    def _obs_single(self):
        pose = self.robot.get_pose()
        g = self.robot.get_gripper_state()

        if self.mujoco:
            rb_pose = self.robot_to_mujoco_pose(np.concatenate([pose.translation, pose.quaternion]))
            gs6 = quat2gs6(rb_pose[3:7])
            return {
                "arm_pose": np.concatenate([rb_pose[0:3], gs6]),
                "grip":  np.array([g], dtype=np.float32),
                "state": np.concatenate([rb_pose[0:3], gs6, [g]], dtype=np.float32),
            }
        else:
            return {
                "arm_pose": np.concatenate([pose.translation, pose.quaternion]),
                "grip":  np.array([g], dtype=np.float32),
                "state": np.concatenate([pose.translation, pose.quaternion, [g]], dtype=np.float32),
            }

    # ──────────────────────────────────────────────────────────
    # Task-specific stubs (same for 1- or 2-arm)
    # ──────────────────────────────────────────────────────────
    def get_reward(self, obs):
        return 0.0

    def is_done(self, obs):
        return self.step_counter >= self.max_steps


class RealSingleRobotEnv:
    """
    Same API as RealRobotEnv but for a single UR5 + gripper.

    Action (len = 8):
        [arm_translation(3), arm_quaternion(4), gripper(1)]

    Observation dict keys:
        'arm_pose'  – np.float32[7]
        'grip'      – np.float32[1]
        'state'     – np.float32[8]   (arm_pose ‖ grip)
    """
    def __init__(self, robot):          # robot: SingleUR5Env or UR5Robot-like
        self.robot        = robot
        self.step_counter = 0
        self.max_steps    = 1000
        self.prev_action  = np.zeros(8, dtype=np.float32)

    def reset(self):
        """Move to home and return first obs."""
        self.robot.set_home(asyn=False)
        self.step_counter = 0
        return self.get_obs()

    def step(self, action: np.ndarray):
        """
        Args
        ----
        action[0:3]  : xyz translation (m, base frame)
        action[3:7]  : wxyz quaternion
        action[7]    : gripper (0=open … 1=closed)
        """
        assert action.shape == (8,), "Action must be length-8 – see docstring."

        pose = RigidTransform(translation=action[:3], rotation=action[3:7])
        grip = float(action[7])

        # motion
        self.robot.move_pose(pose, asyn=False)
        self.robot.move_gripper(grip)

        obs   = self.get_obs()
        rew   = self.get_reward(obs)
        done  = self.is_done(obs)
        info  = {}

        self.prev_action = action.copy()
        self.step_counter += 1
        return obs, rew, done, info

    # ────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────
    def get_obs(self):
        pose = self.robot.get_pose()
        grip = self.robot.get_gripper_state()

        obs_vec = np.concatenate([pose.translation,
                                  pose.quaternion,
                                  [grip]])

        return {
            "arm_pose": obs_vec[:7],
            "grip":     np.array([grip], dtype=np.float32),
            "state":    obs_vec.astype(np.float32),
        }

    def get_prev_action(self):
        return self.prev_action

    # ────────────────────────────────────────────
    # Task-specific stubs
    # ────────────────────────────────────────────
    def get_reward(self, obs):          # plug in your own logic
        return 0.0

    def is_done(self, obs):
        return self.step_counter >= self.max_steps


class OfflineRealRobotEnv:
    """
    Light wrapper that *replays* a recorded Dual-UR5 trajectory.
    It ignores all robot control, uses the dataframe row supplied
    by render_worker (mocap_pos, qpos, ctrl, images already on disk).
    """

    def __init__(self, camera_keys=None, **kwargs):
        self._prev_action = np.zeros(16)
        self._camera_keys = camera_keys or []

    def get_obs(self, **frame):
        qpos = np.asarray(frame.get("qpos"), dtype=np.float32)
        ctrl = np.asarray(frame.get("ctrl"), dtype=np.float32)
        combined = np.concatenate([qpos, ctrl])
        obs  = {"state": combined}

        mocap_pos = np.asarray(frame.get("mocap_pos"), dtype=np.float32)
        mocap_quat = np.asarray(frame.get("mocap_quat"), dtype=np.float32)
        self._prev_action = np.concatenate([mocap_pos, mocap_quat, ctrl])

        for k in self._camera_keys:
            if k in frame:
                # dataframe row stored (C,H,W)  uint8 or float; convert -> (H,W,C)
                img = np.asarray(frame[k])
                if img.ndim == 3 and img.shape[0] in (3, 4):      # (C,H,W)
                    img = img.transpose(1, 2, 0)
                obs[k] = img

        return obs


    def get_prev_action(self, **frame):
        # ctrl is  gripper command(s) recorded in the dataframe
        return self._prev_action

    def get_ordi(self, **frame):
        obs  = self.get_obs(**frame)
        act  = self.get_prev_action(**frame)
        self._prev_action = act
        reward, done, info = 0.0, False, {}
        return obs, reward, done, info