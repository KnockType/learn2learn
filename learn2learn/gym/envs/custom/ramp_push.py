# learn2learn/gym/envs/ramp_push.py

import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from enum import Enum

import gym
from gym import utils
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box
from ml_collections import config_dict

import learn2learn as l2l

FRAME_SKIP = 5
TIMESTEP = 0.002
RENDER_FPS = int(np.round(1.0 / (TIMESTEP * FRAME_SKIP)))

class TaskType(Enum):
    """Enum to define the type of task randomization."""
    PHYSICS = "physics"                # Only physics parameters are randomized
    REWARD_WEIGHTS = "reward_weights"  # Only reward weights are randomized
    BOTH = "both"                      # Both physics and reward are randomized


def get_physics_ranges() -> config_dict.ConfigDict:
    # (This function remains the same)
    return config_dict.create(
        ramp_size_x={"low": 1.0, "high": 3.0},
        ramp_pos_x={"low": 0.5, "high": 1.5},
        slope={"low": -50.0, "high": -20.0},
        object_mass={"low": 1.0, "high": 3.0},
        tool_mass={"low": 1.0, "high": 3.0},
        friction={"low": 0.3, "high": 0.8},
        gravity={"low": 1.5 * -9.81, "high": 0.8 * -9.81},
        reward_weight = {"low": 0.5, "high": 3},
    )

def get_reward_weight_ranges() -> config_dict.ConfigDict:
    """Defines the sampling range for reward weights."""
    return config_dict.create(
        reward_weight={"low": 0.5, "high": 2.0},
        ctrl_cost_weight={"low": 0.1, "high": 0.3},
    )

def default_config() -> config_dict.ConfigDict:
    # (This function remains the same)
    return config_dict.create(
        xml="ramp_push.xml",
        reward_weight=1.0,
        ctrl_cost_weight=0.1,
        ramp_size=np.array([2.0, 1.0, 0.05]),
        ramp_pos=np.array([0.5, 0.0, 0.0]),
        gravity=np.array([0.0, 0.0, -9.81]),
        friction=np.array([0.3, 0.3, 0.3]),
        object_mass=1.0,
        tool_mass=1.0,
        slope=-30.0,
        env_name="ramp_push"
    )

class RampPushEnv(l2l.gym.MetaEnv, MujocoEnv, utils.EzPickle):
    """
    RampPushEnv fully compatible with the learn2learn MetaEnv interface.
    """
    def __init__(self, config: config_dict.ConfigDict = default_config()):
        utils.EzPickle.__init__(self)
        self._config = config
        self._steps = 0
        self._task_type = TaskType.BOTH

        # This is a new helper to contain the initialization logic
        self._re_init(self._config)

        # Call parent constructor AFTER everything is set up.
        xml_path = os.path.join(os.path.dirname(__file__), "assets", config.xml)
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML {xml_path} not found!")
        MujocoEnv.__init__(self, xml_path, frame_skip=FRAME_SKIP)

        # Get IDs and set physics properties now that the model is loaded.
        self._geom_ids = {name: self.model.geom_name2id(name) for name in ["ramp", "ground", "object", "tool"]}
        self._body_ids = {name: self.model.body_name2id(name) for name in ["object", "tool"]}
        self._set_physics()

    def _re_init(self, config: config_dict.ConfigDict):
        """Helper function to re-initialize the environment with a new config."""
        # 1. Set all physical properties from config
        self._config = config
        self._reward_weight = config.reward_weight
        self._ctrl_cost_weight = config.ctrl_cost_weight
        self._ramp_size = config.ramp_size
        self._ramp_pos = config.ramp_pos
        self._gravity = config.gravity
        self._friction = config.friction
        self._object_mass = config.object_mass
        self._tool_mass = config.tool_mass
        self._slope = config.slope
        self._verify_physics()

        # 2. Set goals
        self._set_goals()

        # 3. Define observation space
        obs_dim = 16
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )


    def _sample_physics_params(self) -> dict:
        ranges = get_physics_ranges()
        friction_val = np.random.uniform(ranges.friction.low, ranges.friction.high)
        gravity_z = np.random.uniform(ranges.gravity.low, ranges.gravity.high)
        return {
            "slope": np.random.uniform(ranges.slope.low, ranges.slope.high),
            "object_mass": np.random.uniform(ranges.object_mass.low, ranges.object_mass.high),
            "tool_mass": np.random.uniform(ranges.tool_mass.low, ranges.tool_mass.high),
            "friction": np.array([friction_val] * 3),
            "gravity": np.array([0.0, 0.0, gravity_z]),
        }


    def _sample_reward_weight_params(self) -> dict:
        """Samples a dictionary of random reward weight parameters."""
        ranges = get_reward_weight_ranges()
        return {
            "reward_weight": np.random.uniform(ranges.reward_weight.low, ranges.reward_weight.high),
            "ctrl_cost_weight": np.random.uniform(ranges.ctrl_cost_weight.low, ranges.ctrl_cost_weight.high),
        }

    def sample_tasks(self, num_tasks: int) -> list:
        tasks = []
        for _ in range(num_tasks):
            task_config = default_config()
            if self._task_type == TaskType.PHYSICS:
                task_config.update(self._sample_physics_params())
            elif self._task_type == TaskType.REWARD_WEIGHTS:
                task_config.update(self._sample_reward_weight_params())
            elif self._task_type == TaskType.BOTH:
                task_config.update(self._sample_physics_params())
                task_config.update(self._sample_reward_weight_params())
            tasks.append(task_config)
        return tasks

    def set_task_type(self, new_task_type: TaskType):
        """
        Dynamically changes the task sampling behavior of the environment.

        Args:
            new_task_type (TaskType): The new task type to use for subsequent
                                      calls to sample_tasks().
        """
        self._task_type = new_task_type

    def set_task(self, task: config_dict.ConfigDict):
        """Sets the environment to a specific task configuration."""
        self._re_init(task)
        self._set_physics()

    def _set_physics(self):
        self.model.geom_pos[self._geom_ids["ramp"]] = self._ramp_pos
        self.model.geom_size[self._geom_ids["ramp"]] = self._ramp_size
        r = R.from_euler('xyz', [0, self._slope, 0], degrees=True)
        self.model.geom_quat[self._geom_ids["ramp"]] = np.array([r.as_quat()[3], r.as_quat()[0], r.as_quat()[1], r.as_quat()[2]])
        self.model.opt.gravity[:] = self._gravity
        for geom_id in self._geom_ids.values():
            self.model.geom_friction[geom_id] = self._friction
        self.model.body_mass[self._body_ids["object"]] = self._object_mass
        self.model.body_mass[self._body_ids["tool"]] = self._tool_mass

    def _verify_physics(self):
        # Check that physics parameters are within bounds
        r = get_physics_ranges()

        # Verify the ramp size
        assert self._ramp_size.shape == (3,)
        assert r.ramp_size_x["low"] <= self._ramp_size[0] <= r.ramp_size_x["high"]
        assert self._ramp_size[1] == 1.0 and self._ramp_size[2] == 0.05 # Fixed values

        # Verify the ramp pos
        assert self._ramp_pos.shape == (3,)
        assert r.ramp_pos_x["low"] <= self._ramp_pos[0] <= r.ramp_pos_x["high"]
        assert self._ramp_pos[1] == 0.0 and self._ramp_pos[2] == 0.0 # Fixed values

        # Verify ramp slope
        assert r.slope["low"] <= self._slope <= r.slope["high"]

        # Verify object and tool mass
        assert r.object_mass["low"] <= self._object_mass <= r.object_mass["high"]
        assert r.tool_mass["low"] <= self._tool_mass <= r.tool_mass["high"]

        # Verify friction
        assert self._friction.shape == (3,)
        for f in self._friction:
            assert r.friction["low"] <= f <= r.friction["high"]

        # Verify gravity
        assert self._gravity.shape == (3,)
        assert r.gravity["low"] <= self._gravity[2] <= r.gravity["high"]
        assert self._gravity[0] == 0.0 and self._gravity[1] == 0.0

    def _set_goals(self):
        self._goal_z = np.abs((self._ramp_size[0] / 2.0) * np.sin(np.deg2rad(self._slope)))
        self._goal_x = self._ramp_pos[0] + np.abs((self._ramp_size[0] / 2.0) * np.cos(np.deg2rad(self._slope)))

    def _get_obs(self):
        # (This logic remains the same)
        pos_obj = self.sim.data.get_body_xpos("object")
        quat_obj = self.sim.data.get_body_xquat("object")
        pos_tool = self.sim.data.get_body_xpos("tool")
        vel_obj = self.sim.data.get_body_xvelp("object")
        vel_tool = self.sim.data.get_body_xvelp("tool")

        self._d_tool2obj = np.linalg.norm(pos_tool - pos_obj)
        self._z_obj2goal = np.abs(self._goal_z - pos_obj[2])
        self._x_obj2goal = np.abs(self._goal_x - pos_obj[0])

        return np.concatenate([
            pos_obj, quat_obj, pos_tool, vel_obj, vel_tool
        ]).astype(np.float32)

    def step(self, action: np.ndarray):
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        reward_goal = -1.0 * self._reward_weight * (self._d_tool2obj + self._z_obj2goal + self._x_obj2goal)
        reward_ctrl = -1.0 * self._ctrl_cost_weight * np.sum(np.square(action))
        reward = reward_ctrl + reward_goal
        done = self._steps >= self.spec.max_episode_steps if self.spec else self._steps >= 200
        info = {'d_tool2obj': self._d_tool2obj, 'z_obj2goal': self._z_obj2goal}
        self._steps += 1
        return obs, reward, done, info

    def reset_model(self):
        # (This logic remains the same)
        self.set_state(self.init_qpos, self.init_qvel)
        self._steps = 0
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = 6.0

    def render(self, mode='human'):
        width, height = 500, 500
        if mode == 'rgb_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width,
                                                      height,
                                                      depth=False)
            return data
        elif mode == 'human':
            self._get_viewer(mode).render(width, height)


if __name__ == "__main__":

    env = RampPushEnv()
    print(env._ramp_size, env._goal_z)
    tasks = env.sample_tasks(5)
    for t in tasks:
        env.set_task(t)
        env.reset()
        action = np.random.uniform(-5, 5, (3,))
        obs, reward, done, info = env.step(action)
        print(reward)

    env.close()

# ... (rest of the RampPushEnv class code) ...

if __name__ == "__main__":
    print("--- Starting Verification Script for RampPushEnv ---")

    # 1. Initialize the environment with its default configuration
    env = RampPushEnv()
    print("\n[Step 1] Initializing environment with default parameters:")
    print(f"  - Default Slope: {env._slope:.2f} degrees")
    print(f"  - Default Object Mass: {env._object_mass:.2f} kg")
    print(f"  - Default Gravity: {env._gravity[2]:.2f} m/s^2")
    print("-" * 50)

    # 2. Sample a small batch of new tasks
    NUM_TASKS_TO_VERIFY = 3
    print(f"\n[Step 2] Sampling {NUM_TASKS_TO_VERIFY} new tasks...")
    tasks = env.sample_tasks(NUM_TASKS_TO_VERIFY)
    assert len(tasks) == NUM_TASKS_TO_VERIFY, "sample_tasks() did not return the correct number of tasks."
    print(f"Successfully sampled {len(tasks)} tasks.")
    print("-" * 50)

    # 3. Iterate through each sampled task and verify set_task()
    for i, task_config in enumerate(tasks):
        print(f"\n[Step 3.{i+1}] Verifying Task {i+1}/{NUM_TASKS_TO_VERIFY}...")

        # These are the target values we expect the environment to adopt
        target_slope = task_config.slope
        target_obj_mass = task_config.object_mass
        target_gravity_z = task_config.gravity[2]

        print("  - Target Parameters from task_config:")
        print(f"    - Slope: {target_slope:.2f}")
        print(f"    - Object Mass: {target_obj_mass:.2f}")
        print(f"    - Gravity Z: {target_gravity_z:.2f}")

        # Apply the task to the environment
        env.set_task(task_config)

        # Check if the environment's high-level attributes were updated
        print("  - Checking environment's internal attributes after set_task():")
        assert np.isclose(env._slope, target_slope), "Slope did not update correctly!"
        assert np.isclose(env._object_mass, target_obj_mass), "Object mass did not update correctly!"
        assert np.isclose(env._gravity[2], target_gravity_z), "Gravity did not update correctly!"
        print("    - Internal attributes match target values. [SUCCESS]")

        # (Optional but recommended) Check the underlying MuJoCo model
        print("  - Checking underlying MuJoCo model parameters:")
        actual_model_mass = env.model.body_mass[env._body_ids["object"]]
        actual_model_gravity_z = env.model.opt.gravity[2]
        assert np.isclose(actual_model_mass, target_obj_mass), "MuJoCo model mass did not update!"
        assert np.isclose(actual_model_gravity_z, target_gravity_z), "MuJoCo model gravity did not update!"
        print("    - MuJoCo model parameters match target values. [SUCCESS]")
        
        # Run a single step to ensure the simulation is stable with the new parameters
        env.reset()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"  - Successfully ran one step. Reward: {reward:.3f}")
        print("-" * 50)

    print("\nVerification Complete: sample_tasks() and set_task() are working correctly!")
    env.close()
