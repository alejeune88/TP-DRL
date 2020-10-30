import gym
import numpy as np

class PendulumWrapper(gym.Wrapper):
    """
    Specific wrapper to scale the reward of the pendulum environment
    """
    def __init__(self, env):
        super(PendulumWrapper, self).__init__(env)

    def step(self, action):
        next_state, reward, done, y = self.env.step(action)
        norm = False
        if norm:
            mini = (np.pi**2 + 0.1*8**2 + 0.001*2**2)
            reward = ((reward/mini)+1)*2 - 0.8
            
        
        return next_state, reward, done, y
