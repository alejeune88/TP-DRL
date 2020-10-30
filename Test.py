# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:51:39 2020

@author: Arthur
"""
import os
import numpy as np
from itertools import count
from policies import GenericNet, PolicyWrapper
from environment import make_env

#from policies import policy_wrapper 

def evaluate_pol(env, policy, deterministic):
    """
    Function to evaluate a policy over 900 episodes
    :param env: the evaluation environment
    :param policy: the evaluated policy
    :param deterministic: whether the evaluation uses a deterministic policy
    :return: the obtained vector of 900 scores
    """
    scores = []
    for i in range(900):
        state = env.reset()
        env.render(mode='rgb_array')
        # print("new episode")
        total_reward = 0

        for _ in count():
            action = policy.select_action(state, deterministic)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state

            if done:
                scores.append(total_reward)
                break
    scores = np.array(scores)
    # print("team: ", policy.team_name, "mean: ", scores.mean(), "std:", scores.std())
    return scores

pw = PolicyWrapper(GenericNet(), "", "", "", 100)
folder = 'data/policies/Pendulum-v0#Lejeune_-611.998015954661#bernoulli#None#-611.998015954661.pt'
policy = pw.load(folder)

env = make_env("Pendulum-v0", pw.policy_type, pw.max_steps)

evaluate_pol(env, pw, True)


if __name__ == '__main__':
    directory = os.getcwd() + 'data/policies/Pendulum-v0#Lejeune_-611.998015954661#bernoulli#None#-611.998015954661.pt'
    ev = Evaluator()
    ev.load_policies(directory)
    ev.display_hall_of_fame()

    