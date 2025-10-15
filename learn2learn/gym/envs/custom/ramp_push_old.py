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
    # (This function remains the same)
    return config_dict.create(
        ramp_size_x={"low": 1.0, "high": 3.0},
        ramp_pos_x={"low": 0.5, "high": 1.5},
        slope={"low": -50.0, "high": -20.0},
        object_mass={"low": 1.0, "high": 3.0},
        tool_mass={"low": 1.0, "high": 3.0},
        friction={"low": 0.3, "high": 0.8},
        gravity={"low": 1.5 * -9.81, "high": 0.8 * -9.81},
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
    )

class RampPushEnv(MujocoEnv, utils.EzPickle):
    """
    RampPushEnv fully compatible with the old gym API (v0.21-v0.25).
    """
    def __init__(self, config: config_dict.ConfigDict = default_config()):
        utils.EzPickle.__init__(self)
        self._config = config

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
        
        self._verify_physics()

        # 2. Set goals BEFORE calling the parent constructor
        self._set_goals()

        # 3. Define observation space
        obs_dim = 16
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        xml_path = os.path.join(os.path.dirname(__file__), "assets", config.xml)
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML {xml_path} not found!")

        # 4. Call parent constructor. It will internally call step(), which now works.
        MujocoEnv.__init__(self, xml_path, frame_skip=FRAME_SKIP)

        # 5. NOW that the model is loaded, get IDs and set the physics properties.
        self._geom_ids = {name: self.model.geom_name2id(name) for name in ["ramp", "ground", "object", "tool"]}
        self._body_ids = {name: self.model.body_name2id(name) for name in ["object", "tool"]}
        self._set_physics()

    def _set_physics(self):
        # (This logic remains the same)
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
        # (This logic remains the same)
        r = get_physics_ranges()
        assert r.slope["low"] <= self._slope <= r.slope["high"]
        # ...

    def _set_goals(self):
        # This method calculates and sets self._goal_z and self._goal_x
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
        # (This logic remains the same)
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        reward_goal = -1.0 * self._reward_weight * (self._d_tool2obj + self._z_obj2goal + self._x_obj2goal)
        reward_ctrl = -1.0 * self._ctrl_cost_weight * np.sum(np.square(action))
        reward = reward_ctrl + reward_goal
        done = False
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


if __name__ == "__main__":

    env = RampPushEnv()
    print(env._ramp_size, env._goal_z)
    for i in range(10):
        env.reset()
        action = np.random.uniform(-5, 5, (3,))
        obs, reward, done, info = env.step(action)
        print(reward)

    env.close()