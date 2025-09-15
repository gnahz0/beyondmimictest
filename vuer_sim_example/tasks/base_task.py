from typing import Dict, Any, Optional
import numpy as np

class Task:
    def reset(self, physics) -> None:
        pass

    def compute_obs(self, physics) -> np.ndarray:
        state = physics.get_state()
        qpos = state.get("qpos", [])
        qvel = state.get("qvel", [])
        ctrl = state.get("ctrl", [])

        if not qpos and not qvel and not ctrl:
            return np.array([])

        return np.concatenate([qpos, qvel, ctrl])

    def compute_rew(self, physics) -> float:
        return 0.0

    def is_done(self, physics) -> bool:
        return False

    def get_info(self, physics) -> Dict[str, Any]:
        return {}