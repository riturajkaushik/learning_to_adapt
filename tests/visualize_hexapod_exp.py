import matplotlib.pyplot as plt
import time
import numpy as np
from learning_to_adapt.bullet_envs.bullet_simu.hexapod_simu import Hexapod_env, HexaController 

def block(pp,blocks):
    for i in blocks:
        pp[i*6] = 0 
        pp[i*6 + 3] = 0 
    return pp

env = Hexapod_env(gui=True, visualizationSpeed=5.0)
ctlr = HexaController()
sim_time = 3.0
env.setController(ctlr)

try: 
    data = np.load("../learning_to_adapt/data/path_infomation.npy")
except:
    print("Cannot open path_infomation.npy. Check the path")

task = 0

for i in range(len(data)):     
    episode = i   
    friction = data[episode][task]['env_infos']['friction'][0]
    block_leg = data[episode][task]['env_infos']['blocked_leg'][0]
    pp = np.array([])
    for step in range(20):
        print("Action: ", pp.tolist())
        print("\nObs: ", env.getState()[0:2], "    Expected: ", data[episode][task]['observations'][step][0:2], "\n\n")
        pp = (data[episode][task]['actions'][step]+1.0)/2.0    
        pp = block(pp, block_leg)
        env.setFriction(friction)        
        ctlr.setParams(pp)
        env.run(sim_time)
    env.reset()