"""
Main loop
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import argparse

from maxent_irl_fl import *


deter = False
def load_demo(folder, n_trajs):
    """
    Load expert demonstrations
        trajs: list of trajectories
        each trajectory contains [[action, state_list], ..., []]
    """
    trajs = []
    T = 10  # Fix episode length

    for i in range(n_trajs):
        path = folder + '/trajectory_%d.pkl' % i
        if os.path.exists(path):
            trajectory = pickle.load(open(path, 'rb'))
            trajs.append(trajectory)
    assert len(trajs) >= 2, "At least provide two demonstrations!"
    return trajs


if __name__ == "__main__":
    random.seed(100)
    np.random.seed(100)
    trajs = load_demo('./demos', 15)  # Load Demos
    feat_map = np.load('feat_map_fl.npy')  # Load Feature Map (DO NOT CHANGE FEATURE MAP)
    print(feat_map.shape)
    P_a = np.load('trans_matrix_fl.npy')  # Load Transition Matrix  (DO NOT CHANGE TRANSITION MATRIX)

    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=int, required=True, help='0 for vanilla MaxEnt, 1 for Deep MaxEnt')
    args = parser.parse_args()
    maxent_version = args.version

    # learn rewards from demonstrations using Deep MaxEnt
    if maxent_version == 1:
        print("Using Deep MaxEnt...")
        MAX_ENT_VERSION = "deep"
        # Set hyperparameters for Deep MaxEnt
        gamma = 0.99
        lr = 1e-2
        n_iters = 50
        rewards = deep_maxent_irl(feat_map, P_a, gamma, trajs, lr, n_iters,deterministic = deter)
        rewards = rewards.T.squeeze()  # Must be of shape (N,) where N is the number of states

    # learn rewards from demonstrations using Vanilla MaxEnt (i.e., Linear Mapping of Features)
    else:
        print("Using Vanilla MaxEnt...")
        # Set hyperparameters for Tabular MaxEnt
        MAX_ENT_VERSION = "std"
        gamma = 0.99
        lr = 1e-1
        n_iters = 400
        rewards = maxent_irl(feat_map, P_a, gamma, trajs, lr, n_iters)

    # compute optimal policy w.r.t. the learned rewards
    values, trained_policy = value_iteration(P_a, rewards, gamma, error=0.1, deterministic=deter)
    # save policy
    np.save(f'trained_policy_{MAX_ENT_VERSION}', trained_policy)

    # save reward & plot
    np.save(f'learned_reward_{MAX_ENT_VERSION}', rewards)