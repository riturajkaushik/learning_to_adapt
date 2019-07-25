import matplotlib.pyplot as plt
import time
import numpy as np
from learning_to_adapt.bullet_envs.bullet_simu.hexapod_simu import Hexapod_env, HexaController 

def block(pp,blocks):
    for i in blocks:
        pp[i*6] = 0 
        pp[i*6 + 3] = 0 
    return pp

try: 
    data = np.load("../learning_to_adapt/data/path_infomation.npy")
except:
    print("Cannot open path_infomation.npy. Check the path")


def visualize_simulation(task):
    env = Hexapod_env(gui=True, visualizationSpeed=5.0, simStep=0.004, controlStep=0.01, jointControlMode="velocity")
    ctlr = HexaController()
    sim_time = 3.0
    env.setController(ctlr)

    for i in range(len(data)):     
        episode = -1   
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

def visualize_paths(task):
    for i in range(len(data)):     
        episode = i
        obs = data[episode][task]['observations'][:,0:2]
        plt.plot([d[0] for d in obs], [d[1] for d in obs], '.-r', alpha=i/float(len(data)))
    plt.plot([0], [0], 'ob')
    plt.plot([-4.0], [2.3], '*b')

    plt.show()


visualize_paths(2)