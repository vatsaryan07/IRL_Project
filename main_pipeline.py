"""
Main loop
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import argparse
import pathlib
import textwrap
import PIL.Image
import google.generativeai as genai
import provide_demos
import eval


from maxent_irl import *


def load_demo(folder, n_trajs,corrective = 'N'):
    """
    Load expert demonstrations
        trajs: list of trajectories
        each trajectory contains [[action, state_list], ..., []]
    """
    if corrective == 'Y':
        trajs = []
        T = 15
        for i in range(n_trajs):
            path = folder + '/trajectory_%d.pkl' % i
            if os.path.exists(path):
                trajectory = pickle.load(open(path, 'rb'))
                trajs.append(trajectory)

        corrective_set = trajs[-15:]
        
        # sampled_set = np.random.choice(np.arange(len(trajs) - 5),size = 10,replace=False)
        # for i in sampled_set:
        #     corrective_set += [trajs[i]]
        
        return corrective_set
        
        
    else:
        trajs = []
        T = 15    

        for i in range(n_trajs):
            path = folder + '/trajectory_%d.pkl' % i
            if os.path.exists(path):
                trajectory = pickle.load(open(path, 'rb'))
                trajs.append(trajectory)
        assert len(trajs) >= 2, "At least provide two demonstrations!"
        return trajs


if __name__ == "__main__":
    corrective_feedback = 'N'
    itraj = 15
    rounds = 0
    is_deterministic = False
    random.seed(100)
    np.random.seed(100)
    while True:
        trajs = load_demo('./demos', itraj,corrective=corrective_feedback)  # Load Demos
        feat_map = np.load('feat_map.npy')  # Load Feature Map (DO NOT CHANGE FEATURE MAP)
        P_a = np.load('trans_matrix.npy')  # Load Transition Matrix  (DO NOT CHANGE TRANSITION MATRIX)

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
            n_iters = 10
            rewards = deep_maxent_irl(feat_map, P_a, gamma, trajs, lr, n_iters,deterministic = is_deterministic,rounds = rounds)
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
        values, trained_policy = value_iteration(P_a, rewards, gamma, error=0.1, deterministic=is_deterministic)

        # Access your API key as an environment variable.
        genai.configure(api_key=os.environ['API_KEY'])
        # Choose a model that's appropriate for your use case.
        model = genai.GenerativeModel('gemini-1.5-flash')

        pict = PIL.Image.open('results/shapley_2_'+str(rounds)+'_.png')
        prompt = """We are training an inverse reinforcement learning model using Deep Max Entropy algorithm on the Wumpus World environment. The model uses small feedforward neural networks to learn the reward function from the given demonstrations. The training manual of the game given below has information about the environment.

        * TRAINING MANUAL * 

        The game starts with the player at location [0,0] of the frozen lake grid world with the goal (treasure) located at far extent of the world e.g. [3,3] for the 4x4 environment.The player makes moves until they reach the goal or fall in a hole or gets eaten by the monster. There is always a path to the goal.

        Holes in the ice are distributed in set locations of [(1,1), (1,3), (2,3), (3,0)]. The monster starts at location (2,2). It chooses uniformly randomly whether it wants to proceed in an upward direction [(2,2), 1,2), (0,2)] or towards the left [(2,2), (2,1), (2,0]. 

        ## Action Space
        The action shape is (1,) in the range {0, 3} indicating which direction to move the player.
            - 0: Move left
            - 1: Move down
            - 2: Move right
            - 3: Move up

        The agent is in a stochastic environment where with probability 0.1 to move in a random direction and probability 

        ## Observation Space
        The observation is a value representing the player's current position as current_row * ncols + current_col (where both the row and col start at 0). For example, the goal position in the 4x4 map can be calculated as follows: 3 * 4 + 3 = 15. The number of possible observations is dependent on the size of the map.The observation is returned as an int().

        ## Starting State
        The episode starts with the player in state [0] (location [0, 0]).

        ## Episode End
        The episode ends if the following happens:
                1. The player moves into a hole.
                2. The player reaches the goal at max(nrow) * max(ncol) - 1 (location [max(nrow)-1, max(ncol)-1]).
                3. The player gets eaten by the monster.

        * END OF TRAINING MANUAL *

        In order to understand the model better we use SHAP. The SHAPley values for the above model are given in the plot below. These are the features - 
        1. Distance left to reach the goal. 
        2. Distance to the nearest hole. 
        3. Location of the monster 
        4. Distance to the monster 
        5. Location of the agent 
        
        Additional features cannot be provided and you should not recommend providing more features. 

        Can you explain these values?
        What features the robot is using to learn the reward function? Your response will help the demonstrator in understanding the robot learning better. How would you suggest these values should be modified to achieve the best reward function?
        Be critical in your evaluation of how the demonstrator can improve the agent. Remember positive distance value encourages the agent to increase the distance and negative distance value encourages it to decrease the distance."""

        feature_shaps = pickle.load(open('/Users/rynaa/IRL_Project/new_results/shapley_values.pkl', 'rb'))

        shap_values_prompt = ''
        feature_names=['Agent Loc','Monster Loc','Dist to Hole','Dist to Monster','Dist to Goal']
        for i in range(len(feature_names)):
            shap_values_prompt += f'Feature - {feature_names[i]} has mean SHAP value of {feature_shaps[:,i].mean()} and standard deviation of {feature_shaps[:,i].std()}\n'
        
        print("------------------------------------------------")
        print(shap_values_prompt)
        print("------------------------------------------------")
        print("Results")
        np.save(f'trained_policy_{MAX_ENT_VERSION}', trained_policy)
        np.save(f'learned_reward_{MAX_ENT_VERSION}', rewards)
        eval.main(deterministic=is_deterministic,return_wins=False)
        response = model.generate_content(prompt+shap_values_prompt)
        response.resolve()
        eval.main(deterministic=is_deterministic,render=1)
        
        print("------------------------------------------------")
        print("Do you wish to see the feedback from the agent for the following training run?(Y/N)")
        see_feedback = input()
        if see_feedback == 'Y':
            print("------------------------------------------------")
            print(response.text)
            print("------------------------------------------------")
        print("Do you wish to provide corrective demonstrations?(Y/N)")
        
        corrective_feedback = input()
        if corrective_feedback == 'Y':
            # run provide demos to get demonstrations
            provide_demos.main(itraj,5,deterministic=is_deterministic)
            itraj += 5
        else:
            break
        rounds += 1
        
    
    
    # save policy
    np.save(f'trained_policy_{MAX_ENT_VERSION}', trained_policy)

    # save reward & plot
    np.save(f'learned_reward_{MAX_ENT_VERSION}', rewards)