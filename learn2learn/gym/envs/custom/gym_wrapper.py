import gym
import numpy as np
from typing import Tuple, Dict, Any
import gymnasium as new_gym

class GymCompatibility(gym.Wrapper, new_gym.Env):
    """
    A wrapper to convert a gymnasium-based environment into a gym-compatible one.

    This wrapper handles the main API differences between Gymnasium (v0.26+) and
    the older Gym API (v0.21 to v0.25).

    Specifically, it modifies:
    - `step()`: Combines the `terminated` and `truncated` flags into a single `done` flag.
    - `reset()`: Returns only the observation, discarding the info dictionary.
    """
    def __init__(self, env):
        super().__init__(self, env)
        self.observation_space = self._convert_space(env.observation_space)
        self.action_space = self._convert_space(env.action_space)

    def _convert_space(self, space):
        if isinstance(space, new_gym.spaces.Box):
            return gym.spaces.Box(
                low=space.low, 
                high=space.high, 
                shape=space.shape, 
                dtype=space.dtype
            )
        return space

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Steps the environment and converts the output to the old gym format.

        Returns:
            (observation, reward, done, info)
        """
        # Call the gymnasium environment's step function
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Combine terminated and truncated into a single 'done' flag
        done = terminated or truncated

        return obs, reward, done, info

    def reset(self, **kwargs) -> np.ndarray:
        """
        Resets the environment and returns only the observation.
        """
        # Call the gymnasium environment's reset function
        obs, info = self.env.reset(**kwargs)

        # Return only the observation, as per the old gym API
        return obs