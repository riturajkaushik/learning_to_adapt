import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
# from simu.hexapod_simu import Hexapod_env 
from .bullet_simu.hexapod_simu import Hexapod_env, HexaController 

class HexapodEnv(gym.Env): 
    def __init__(self, goal=np.array([-4, 2.3]), task=None, reset_every_episode=False):
        self.action_space = spaces.Box(low= np.zeros(36), high=np.ones(36))
        self.observation_space = spaces.Box(low= np.array([-20,-20, -1, -1]), high=np.array([20,20, 1, 1])) #(x, y, sin_theta, cos_theta)
        self.ctlr = HexaController()
        self.hexa = Hexapod_env(gui=True, visualizationSpeed=8.0)
        self.hexa.setController(self.ctlr)
        self.sim_time = 3.0
        self.state = np.array([0, 0, np.sin(0), np.cos(0)])
        self.goal = goal
        self.reset_every_episode = reset_every_episode
        self.task = task

    def __get_state(self):
        cm =  self.hexa.getState()[0:2]
        ang = self.hexa.getEulerAngles()[2]
        ang = np.deg2rad(ang)
        return np.array([cm[0], cm[1], np.sin(ang), np.cos(ang)])
    
    def step(self, action):
        # print ("\nApplied: ", action.tolist())
        self.ctlr.setParams(action)
        self.hexa.run(self.sim_time)
        self.state = self.__get_state()
        diff = (self.state[0]-self.goal[0])**2 +  (self.state[1]-self.goal[1])**2 
        rew = -diff#np.exp(-0.05*diff)
        return self.__get_state(), rew, False, {}

    def reward(self, obs, action, next_obs):
        return 0
        # assert obs.ndim == 2
        # assert obs.shape == next_obs.shape
        # assert obs.shape[0] == action.shape[0]
        # ctrl_cost = 1e-1 * 0.5 * np.sum(np.square(action), axis=1)
        # forward_reward = (next_obs[:, -3] - obs[:, -3])/self.dt
        # reward = forward_reward - ctrl_cost
        # return reward

    def reset(self):
        self.hexa.reset()
        self.state = self.__get_state()
        return self.state
 
    def render(self, mode='human', close=False):
        pass

    def reset_mujoco(self, init_state=None):
        print("reset_mujoco called")
        self.reset()

    def reset_task(self, value=None):
        print("reset_task called")