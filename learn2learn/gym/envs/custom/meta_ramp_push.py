import numpy as np
import learn2learn as l2l
from ml_collections import config_dict
import gymnasium

from ramp_push_old import RampPushEnv, get_physics_ranges, default_config

class MetaRampPushEnv(l2l.gym.MetaEnv, gymnasium.Env, RampPushEnv):
    """
    Wrapper for RampPushEnv that allows for sampling and setting tasks
    with varying physical properties.
    """
    def __init__(self, render_mode=None):
        RampPushEnv.__init__(self)
        # The underlying environment is re-initialized for each task.
        self._env = None
        self._render_mode = render_mode
        self.set_task(self.sample_tasks(1)[0])

    def sample_tasks(self, num_tasks):
        """
        Samples a list of task configurations.
        Each task is a dictionary of physical properties.
        """
        tasks = []
        ranges = get_physics_ranges()
        for _ in range(num_tasks):
            task_config = {
                "ramp_size_x": np.random.uniform(ranges.ramp_size_x["low"], ranges.ramp_size_x["high"]),
                "ramp_pos_x": np.random.uniform(ranges.ramp_pos_x["low"], ranges.ramp_pos_x["high"]),
                "slope": np.random.uniform(ranges.slope["low"], ranges.slope["high"]),
                "object_mass": np.random.uniform(ranges.object_mass["low"], ranges.object_mass["high"]),
                "tool_mass": np.random.uniform(ranges.tool_mass["low"], ranges.tool_mass["high"]),
                "friction": np.random.uniform(ranges.friction["low"], ranges.friction["high"], size=3),
                "gravity": np.random.uniform(ranges.gravity["low"], ranges.gravity["high"]),
            }
            tasks.append(task_config)
        return tasks

    def set_task(self, task):
        """
        Sets the environment to a new task configuration.
        """
        if self._env is not None:
            self._env.close()

        # Create a new config based on the sampled task
        new_config = default_config()
        new_config.ramp_size = np.array([task["ramp_size_x"], 1.0, 0.05])
        new_config.ramp_pos = np.array([task["ramp_pos_x"], 0.0, 0.0])
        new_config.slope = task["slope"]
        new_config.object_mass = task["object_mass"]
        new_config.tool_mass = task["tool_mass"]
        new_config.friction = np.array(task["friction"])
        new_config.gravity = np.array([0.0, 0.0, task["gravity"]])
        
        # Re-initialize the environment with the new physics
        self._env = RampPushEnv(config=new_config)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        return True

    def reset(self, *args, **kwargs):
        return self._env.reset(*args, **kwargs)

    def step(self, action):
        return self._env.step(action)

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def close(self):
        if self._env is not None:
            self._env.close()

if __name__ == "__main__":

    env = MetaRampPushEnv()
    print(env._ramp_size, env._goal_z)
    tasks = env.sample_tasks(10)
    print(tasks)
    for t in tasks:
        env.set_task(t)
        env.reset()
        print(env)
        action = np.random.uniform(-5, 5, (3,))
        obs, reward, done, info = env.step(action)
        print(reward)

    env.close()