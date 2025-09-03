import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer
from typing import Optional, Dict, Any, Tuple
from vuer import Vuer
from vuer.schemas import Mjcf, Scene
import asyncio
import threading


class VuerSimEnvironment(gym.Env):
    """Base environment class that wraps MuJoCo with Vuer server integration."""
    
    def __init__(
        self,
        model_path: str,
        frame_skip: int = 5,
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
        vuer_port: int = 8012,
        enable_vuer: bool = True,
        render_mode: Optional[str] = None,
    ):
        self.model_path = model_path
        self.frame_skip = frame_skip
        self.render_mode = render_mode
        self.enable_vuer = enable_vuer
        self.vuer_port = vuer_port
        
        # Initialize MuJoCo
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Set up spaces
        if observation_space is None:
            obs_size = self._get_obs().shape[0]
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
            )
        else:
            self.observation_space = observation_space
            
        if action_space is None:
            self.action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
            )
        else:
            self.action_space = action_space
        
        # Initialize Vuer server if enabled
        self.vuer_app = None
        self.vuer_thread = None
        if self.enable_vuer:
            self._init_vuer_server()
        
        # Initialize viewer for native rendering
        self.viewer = None
        if render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
    
    def _init_vuer_server(self):
        """Initialize the Vuer server for web-based visualization."""
        self.vuer_app = Vuer(port=self.vuer_port)
        
        async def setup_scene():
            self.vuer_app.set @ Scene()
            
            # Load the MJCF model into Vuer
            with open(self.model_path, 'r') as f:
                mjcf_content = f.read()
            
            self.vuer_app.set @ Mjcf(
                src=mjcf_content,
                key="mujoco_model"
            )
        
        def run_vuer():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(setup_scene())
            loop.run_until_complete(self.vuer_app.run())
        
        self.vuer_thread = threading.Thread(target=run_vuer, daemon=True)
        self.vuer_thread.start()
    
    def _get_obs(self) -> np.ndarray:
        """Get the current observation from the environment."""
        # Default implementation: concatenate qpos and qvel
        return np.concatenate([self.data.qpos, self.data.qvel])
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the current state."""
        return {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment."""
        # Apply action to actuators
        self.data.ctrl[:] = action
        
        # Step the simulation
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        
        # Update visualization if needed
        if self.viewer is not None:
            self.viewer.sync()
        
        if self.enable_vuer and self.vuer_app:
            self._update_vuer_visualization()
        
        # Get observation, reward, and done
        obs = self._get_obs()
        reward = self._compute_reward(obs, action)
        terminated = self._is_terminated()
        truncated = False
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _compute_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        """Compute the reward for the current step."""
        # To be implemented by subclasses
        return 0.0
    
    def _is_terminated(self) -> bool:
        """Check if the episode is terminated."""
        # To be implemented by subclasses
        return False
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset MuJoCo simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Apply any initial configuration
        if options and "init_qpos" in options:
            self.data.qpos[:] = options["init_qpos"]
        if options and "init_qvel" in options:
            self.data.qvel[:] = options["init_qvel"]
        
        # Forward dynamics to update derived quantities
        mujoco.mj_forward(self.model, self.data)
        
        # Update visualization
        if self.viewer is not None:
            self.viewer.sync()
        
        if self.enable_vuer and self.vuer_app:
            self._update_vuer_visualization()
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info
    
    def _update_vuer_visualization(self):
        """Update the Vuer visualization with current state."""
        # This will be implemented to send state updates to Vuer
        pass
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human" and self.viewer:
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            # Return RGB array for recording
            return self._get_rgb_array()
    
    def _get_rgb_array(self) -> np.ndarray:
        """Get RGB array of current view."""
        # Implementation for getting RGB array from MuJoCo
        renderer = mujoco.Renderer(self.model)
        renderer.update_scene(self.data)
        return renderer.render()
    
    def close(self):
        """Clean up resources."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        
        if self.vuer_app is not None:
            # Vuer cleanup if needed
            pass