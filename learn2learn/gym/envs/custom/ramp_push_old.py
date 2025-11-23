import os
import numpy as np
from scipy.spatial.transform import Rotation as R

import gym
from gym import utils
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box
from ml_collections import config_dict

FRAME_SKIP = 5
TIMESTEP = 0.002
RENDER_FPS = int(np.round(1.0 / (TIMESTEP*FRAME_SKIP)))

def get_physics_ranges() -> config_dict.ConfigDict:
    return config_dict.create(
        ramp_size_x = {"low": 1.0, "high": 3.0},
        ramp_pos_x = {"low": 0.5, "high": 1.5},
        slope = {"low": -50, "high": -20},
        object_mass = {"low": 1.0, "high": 3.0},
        tool_mass = {"low": 1.0, "high": 3.0},
        friction = {"low": 0.3, "high": 0.8},
        gravity = {"low": 1.5*-9.81, "high": 0.8*-9.81},
    )

def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        xml="ramp_push.xml",
        vision=False,
        sparse=False,
        reward_weight=1.0,
        ctrl_cost_weight=0.1,
        default_camera_config={"distance": 6.0,},
        # configurable physical properties of env
        ramp_size=np.array([2.0, 1.0, 0.05]),   # length, width and thickness of ramp
        ramp_pos=np.array([0.5, 0.0, 0.0]),     # position of ramp center
        gravity=np.array([0.0, 0.0, -9.81]),    # gravity
        friction=np.array([0.3, 0.3, 0.3]),     # coeff of friction
        object_mass=1.0,                        # mass of object pushed up the ramp
        tool_mass=1.0,                          # mass of tool pushed up the ramp
        slope=-30,                              # slope of the ramp in degrees
    )

class RampPushEnv(MujocoEnv, utils.EzPickle):
    """
    RampPushEnv fully compatible with the old gym API (v0.21-v0.25).
    """
    metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": RENDER_FPS
            }
    
    def __init__(self, config: config_dict.ConfigDict = default_config(), **kwargs):
        utils.EzPickle.__init__(**locals())
        self._config = config
        self.vision = config.vision
        if self.vision:
            raise NotImplementedError("Vision mode is not implemented for RampPushEnv.")
        self._default_camera_config = config.default_camera_config

        self._sparse = config.sparse
        # 1. Set all physical properties from config first
        self._reward_weight = config.reward_weight
        self._ctrl_cost_weight = config.ctrl_cost_weight
        self._ramp_size = config.ramp_size
        self._ramp_pos = config.ramp_pos
        self._gravity = config.gravity
        self._friction = config.friction
        self._object_mass = config.object_mass
        self._tool_mass = config.tool_mass
        self._slope = config.slope
        self._steps = 0
        

        # 2. Set goals BEFORE calling the parent constructor

        self._set_goals()

        self.xml_path = os.path.join(os.path.dirname(__file__), "assets", config.xml)
        if not os.path.exists(self.xml_path):
            raise FileNotFoundError(f"XML {self.xml_path} not found!")

        # 4. Call parent constructor. It will internally call step(), which now works.
        MujocoEnv.__init__(
            self,
            model_path=self.xml_path,
            frame_skip=FRAME_SKIP)
        # 3. Define observation space
        obs_dim = 16
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        # 5. NOW that the model is loaded, get IDs and set the physics properties.
        self._geom_ids = {name: self.model.geom_name2id(name) for name in ["ramp", "ground", "object", "tool"]}
        self._body_ids = {name: self.model.body_name2id(name) for name in ["object", "tool"]}
        self._verify_physics()
        self._set_physics()

    def _set_physics(self):

        # Set ramp position
        self.model.geom_pos[self._geom_ids["ramp"]] = self._ramp_pos

        # Set ramp size
        self.model.geom_size[self._geom_ids["ramp"]] = self._ramp_size

        # Set slope of ramp
        euler_deg = [0, self._slope, 0]
        r = R.from_euler('xyz', euler_deg, degrees=True)
        quat_xyzw = r.as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        self.model.geom_quat[self._geom_ids["ramp"]] = quat_wxyz

        # Set gravity
        self.model.opt.gravity[:] = self._gravity

        # Set friction
        for geom_id in self._geom_ids.values():
            self.model.geom_friction[geom_id] = self._friction

        # Set mass of object
        self.model.body_mass[self._body_ids["object"]] = self._object_mass

        # Set mass of tool
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
        # This method calculates and sets self._goal_z and self._goal_x
        self._goal_z = np.abs((self._ramp_size[0] / 2.0) * np.sin(np.deg2rad(self._slope)))
        self._goal_x = self._ramp_pos[0] + np.abs((self._ramp_size[0] / 2.0) * np.cos(np.deg2rad(self._slope)))

    def _get_obs(self):
        
        self.pos_obj = self.sim.data.get_body_xpos("object")
        self.quat_obj = self.sim.data.get_body_xquat("object")
        self.pos_tool = self.sim.data.get_body_xpos("tool")
        self.vel_obj = self.sim.data.get_body_xvelp("object")
        self.vel_tool = self.sim.data.get_body_xvelp("tool")

        self._d_tool2obj = np.linalg.norm(self.pos_tool - self.pos_obj)
        self._z_obj2goal = np.abs(self._goal_z- self.pos_obj[2])
        self._x_obj2goal = np.abs(self._goal_x- self.pos_obj[0])

        self.obs_dict = dict(
            pos_obj=self.pos_obj,
            quat_obj=self.quat_obj,
            pos_tool=self.pos_tool,
            lin_vel_obj=self.vel_obj,
            lin_vel_tool=self.vel_tool,
        )

        # Observation
        return np.concatenate([
            self.pos_obj,
            self.quat_obj,
            self.pos_tool,
            self.vel_obj,
            self.vel_tool,
            ])

    def step(self, action: np.ndarray):
        
        # Step the simulation with provided control
        self.do_simulation(action, self.frame_skip)

        # Get the current observation.
        obs = self._get_obs()

        # Goal cost
        # Negative distance between "tool" and "object"
        # Negative z-distance between "object" and "goal"
        # Negative x-distance between "object" and "goal"
        reward_goal = -1.0 * self._reward_weight * (self._d_tool2obj + self._z_obj2goal + self._x_obj2goal)

        # Control cost
        reward_ctrl = -1.0 * self._ctrl_cost_weight * np.sum(action**2)

        # Compute reward
        reward = reward_ctrl + reward_goal

        # done is always False, as we want to stop at the goal
        # and rely on a time limit wrapper to end the episode.
        done = False

        info = dict(
            obs_dict=self.obs_dict,
            d_tool2obj=self._d_tool2obj,
            z_obj2goal=self._z_obj2goal,
            x_obj2goal=self._x_obj2goal,
            goal_x=self._goal_x,
            goal_z=self._goal_z,
            reward_goal=reward_goal,
            reward_ctrl=reward_ctrl,
        )

        self._steps += 1

        return obs, reward, done, info

    def reset_model(self):
        self.set_state(self.init_qpos.copy(), self.init_qvel.copy())
        self._steps = 0
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = 6.0

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

    def sample_tasks(self, num_tasks: int) -> list:
        """Generates a list of 'num_tasks' different physics configurations."""
        tasks = []
        ranges = get_physics_ranges()
        for _ in range(num_tasks):
            task_config = default_config()
            task_config.object_mass = np.random.uniform(ranges.object_mass.low, ranges.object_mass.high)
            task_config.tool_mass = np.random.uniform(ranges.tool_mass.low, ranges.tool_mass.high)
            friction_val = np.random.uniform(ranges.friction.low, ranges.friction.high)
            task_config.friction = np.array([friction_val] * 3)
            gravity_z = np.random.uniform(ranges.gravity.low, ranges.gravity.high)
            task_config.gravity = np.array([0.0, 0.0, gravity_z])
            tasks.append(task_config)
        return tasks

    def set_task(self, task: config_dict.ConfigDict):
        """Sets the environment to a specific task configuration."""
        self._re_init(task)
        self._set_physics()


if __name__ == "__main__":

    env = RampPushEnv()
    print(env._ramp_size, env._goal_z)
    for i in range(10):
        env.reset()
        action = np.random.uniform(-5, 5, (3,))
        obs, reward, done, info = env.step(action)
        print(reward)

    env.close()


