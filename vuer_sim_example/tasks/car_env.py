import os
import numpy as np
import gymnasium as gym
from typing import Optional, Dict, Any, Tuple
from ..environment import VuerSimEnvironment


class CarEnv(VuerSimEnvironment):
    """
    Car environment with simple action and state space.
    
    Action space: 2D continuous
    - forward/backward motor control [-1, 1]
    - turning motor control [-1, 1]
    
    Observation space:
    - Car position (x, y, z)
    - Car orientation (quaternion)
    - Car velocity (linear and angular)
    - Wheel joint positions and velocities
    """
    
    def __init__(
        self,
        frame_skip: int = 5,
        vuer_port: int = 8012,
        enable_vuer: bool = True,
        render_mode: Optional[str] = None,
        reward_type: str = "forward",  # "forward", "target", or "exploration"
        target_position: Optional[np.ndarray] = None,
    ):
        # Get the path to the car MJCF file
        model_path = os.path.join(
            os.path.dirname(__file__), 
            "car.mjcf.xml"
        )
        
        self.reward_type = reward_type
        self.target_position = target_position or np.array([2.0, 2.0, 0.0])
        self.initial_position = np.array([0.0, 0.0, 0.03])
        
        # Track exploration for reward calculation
        self.visited_positions = set()
        self.max_distance_from_origin = 0.0
        
        # Define action and observation spaces
        action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        # Observation: car pose (7) + car velocity (6) + wheel joints (4)
        observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32
        )
        
        super().__init__(
            model_path=model_path,
            frame_skip=frame_skip,
            observation_space=observation_space,
            action_space=action_space,
            vuer_port=vuer_port,
            enable_vuer=enable_vuer,
            render_mode=render_mode,
        )
        
        # Get indices for car body and joints
        self.car_body_id = self.model.body("car").id
        self.left_wheel_joint_id = self.model.joint("left").id
        self.right_wheel_joint_id = self.model.joint("right").id
    
    def _get_obs(self) -> np.ndarray:
        """Get observation of the car state."""
        # Get car body position and orientation (quaternion)
        car_pos = self.data.body("car").xpos.copy()
        car_quat = self.data.body("car").xquat.copy()
        
        # Get car body velocity (linear and angular)
        car_vel = self.data.body("car").cvel.copy()
        
        # Get wheel joint positions and velocities
        left_wheel_pos = self.data.joint("left").qpos[0]
        right_wheel_pos = self.data.joint("right").qpos[0]
        left_wheel_vel = self.data.joint("left").qvel[0]
        right_wheel_vel = self.data.joint("right").qvel[0]
        
        obs = np.concatenate([
            car_pos,           # 3D position
            car_quat,          # 4D quaternion orientation
            car_vel.flatten(), # 6D velocity (3 linear + 3 angular)
            [left_wheel_pos, right_wheel_pos, left_wheel_vel, right_wheel_vel]  # 4D wheel state
        ])
        
        return obs.astype(np.float32)
    
    def _compute_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        """Compute reward based on reward type."""
        car_pos = obs[:3]
        car_vel = obs[7:10]  # Linear velocity
        
        if self.reward_type == "forward":
            # Reward forward motion
            forward_vel = car_vel[0]  # Assuming x is forward
            reward = forward_vel
            
        elif self.reward_type == "target":
            # Reward for reaching target position
            dist_to_target = np.linalg.norm(car_pos[:2] - self.target_position[:2])
            reward = -dist_to_target
            
            # Bonus for reaching target
            if dist_to_target < 0.5:
                reward += 10.0
                
        elif self.reward_type == "exploration":
            # Reward for exploring new areas
            grid_pos = tuple(np.round(car_pos[:2] * 2).astype(int))  # Discretize position
            
            if grid_pos not in self.visited_positions:
                self.visited_positions.add(grid_pos)
                reward = 1.0
            else:
                reward = 0.0
            
            # Track max distance
            dist_from_origin = np.linalg.norm(car_pos[:2])
            if dist_from_origin > self.max_distance_from_origin:
                self.max_distance_from_origin = dist_from_origin
                reward += 0.5
        else:
            reward = 0.0
        
        # Small penalty for large actions (energy cost)
        reward -= 0.01 * np.sum(action**2)
        
        return float(reward)
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate."""
        car_pos = self.data.body("car").xpos
        car_quat = self.data.body("car").xquat
        
        # Terminate if car flips over (check if upright)
        up_vector = np.array([0, 0, 1, 0])
        rotated_up = self._rotate_vector_by_quat(up_vector[:3], car_quat)
        if rotated_up[2] < 0.5:  # Car is flipped if z-component of up vector is too low
            return True
        
        # Terminate if car goes out of bounds
        if np.abs(car_pos[0]) > 3.0 or np.abs(car_pos[1]) > 3.0:
            return True
        
        # Terminate if reached target (for target reward type)
        if self.reward_type == "target":
            dist_to_target = np.linalg.norm(car_pos[:2] - self.target_position[:2])
            if dist_to_target < 0.3:
                return True
        
        return False
    
    def _rotate_vector_by_quat(self, vec: np.ndarray, quat: np.ndarray) -> np.ndarray:
        """Rotate a 3D vector by a quaternion."""
        # Convert quaternion rotation to rotation matrix and apply
        import mujoco
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, quat)
        mat = mat.reshape(3, 3)
        return mat @ vec
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info about the environment state."""
        car_pos = self.data.body("car").xpos
        info = {
            "car_position": car_pos.copy(),
            "distance_from_origin": np.linalg.norm(car_pos[:2]),
        }
        
        if self.reward_type == "target":
            info["distance_to_target"] = np.linalg.norm(car_pos[:2] - self.target_position[:2])
        elif self.reward_type == "exploration":
            info["explored_cells"] = len(self.visited_positions)
            info["max_distance"] = self.max_distance_from_origin
        
        return info
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the car to initial position."""
        # Reset tracking variables
        self.visited_positions.clear()
        self.max_distance_from_origin = 0.0
        
        # Set initial position with some randomization
        init_options = options or {}
        
        if "init_qpos" not in init_options:
            # Random initial position near origin
            if seed is not None:
                np.random.seed(seed)
            
            init_qpos = np.zeros(self.model.nq)
            # Add small random perturbations to x, y position
            init_qpos[0] = np.random.uniform(-0.2, 0.2)  # x
            init_qpos[1] = np.random.uniform(-0.2, 0.2)  # y
            init_qpos[2] = 0.03  # z (keep at default height)
            # Random initial orientation
            init_qpos[3:7] = np.array([1, 0, 0, 0])  # Identity quaternion
            
            init_options["init_qpos"] = init_qpos
        
        return super().reset(seed=seed, options=init_options)