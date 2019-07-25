import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
import copy
from  learning_to_adapt.bullet_envs.bullet_simu.hexapod_simu import Hexapod_env, HexaController 
from learning_to_adapt.logger import logger

class HexapodEnv(gym.Env): 
    def __init__(self, goal=np.array([-4, 2.3]), task=None, reset_every_episode=False, gui=False,  visualizationSpeed= 3.0):
        # assert set(task) <= set(['blocked_leg', 'friction']) or task is None, "Task must me a list or None"
        self.action_space = spaces.Box(low= np.zeros(36), high=np.ones(36))
        self.observation_space = spaces.Box(low= np.array([-20,-20, -1, -1]), high=np.array([20,20, 1, 1])) #(x, y, sin_theta, cos_theta)
        self.ctlr = HexaController()
        self.gui = gui
        self.vspeed = visualizationSpeed
        self.simu = Hexapod_env(gui=self.gui, visualizationSpeed=self.vspeed)
        self.simu.setController(self.ctlr)
        self.sim_time = 3.0
        self.state = np.array([0, 0, np.sin(0), np.cos(0)])
        self.goal = goal
        self.reset_every_episode = reset_every_episode
        self.task = task

        self.disable_legs = []  
        self.friction = 5.0 

    def disable_leg_param(self, param):
        blocked_legs = self.disable_legs
        pp = param.copy()
        for leg in blocked_legs:
            assert leg < 6 and leg >= 0
            # disable first joint
            pp[6*leg] = 0   
            pp[6*leg+1] = 0   
            pp[6*leg+2] = 0  
            
            # disable 2nd joint
            pp[6*leg+3] = 0   
            pp[6*leg+4] = 0   
            pp[6*leg+5] = 0   
        return pp
      
    def __get_state(self):
        cm =  self.simu.getState()[0:2]
        ang = self.simu.getEulerAngles()[2]
        ang = np.deg2rad(ang)
        return np.array([cm[0], cm[1], np.sin(ang), np.cos(ang)])
    
    def step(self, action):
        # self.ctlr.setParams(self.disable_leg_param(action))
        self.ctlr.setParams(action)
        self.simu.run(self.sim_time)
        self.state = self.__get_state()
        diff = (self.state[0]-self.goal[0])**2 +  (self.state[1]-self.goal[1])**2 
        rew = -diff#np.exp(-0.05*diff)
        return self.__get_state(), rew, False, {"friction":self.friction, "blocked_leg":self.disable_leg}

    def reward(self, obs, action, next_obs):
        self.state = self.__get_state()
        diff = (self.state[0]-self.goal[0])**2 +  (self.state[1]-self.goal[1])**2 
        rew = -diff
        return rew

    def reset(self):
        self.simu.reset()
        self.reset_task()
        self.state = self.__get_state()
        return self.state
 
    def render(self, mode='human', close=False):
        pass

    def reset_mujoco(self, init_state=None):
        print("reset_mujoco called")
        self.reset()

    def reset_task(self, value=None):
        if 'blocked_leg' in self.task:
            self.disable_leg = [np.random.randint(0,6)]
            print("Disabled legs: ", self.disable_leg)
        
        if 'friction' in self.task:
            self.friction = np.random.choice([0.6, 1.0, 5.0])
            self.simu.setFriction(self.friction)
            print("Friction: ", self.friction)
    
    def log_diagnostics(self, paths, prefix):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
            ]
        logger.logkv(prefix + 'AverageForwardProgress', np.mean(progs))
        logger.logkv(prefix + 'MaxForwardProgress', np.max(progs))
        logger.logkv(prefix + 'MinForwardProgress', np.min(progs))
        logger.logkv(prefix + 'StdForwardProgress', np.std(progs))
    
    def clone(self):
        if not self.simu.gui:
            env = copy.deepcopy(self)
            env.simu.init()
        else:
            raise Exception('Environment cannot be cloned if GUI enabled')
        return env
