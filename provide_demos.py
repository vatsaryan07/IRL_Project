import numpy as np
import gymnasium as gym
import pygame
from pygame.locals import *
from utils import saveObject
import os

np.random.seed(1)
TRAJ_LEN = 15  # max time steps for the game (i.e. episode horizon)
DEMODIR = 'demos'

import env as det
import env_copy as stoch
# env = IRLEnv(
#     render_mode="human",
#     seed=0,
#     version=5
# )

det_env = det.IRLEnv(
    render_mode="human",
    seed=42,
    version=3
)

stoch_env = stoch.IRLEnv(
    render_mode="human",
    seed=42,
    version=3
)

KEY_DICT = {
    K_LEFT: (0, 'LEFT'),
    K_DOWN: (1, 'DOWN'),
    K_RIGHT: (2, 'RIGHT'),
    K_UP: (3, 'UP'),
}

def main(itraj = 0,num_demos = 15,deterministic = True):
    init_itraj = itraj
    os.makedirs(DEMODIR, exist_ok=True)
    if deterministic:
        env = det_env
    else:
        env = stoch_env
    print(env.version)
    print('=============================================================================================================')
    print("Instructions:")
    print(f"Provided demos must be of fixed length = {TRAJ_LEN}.")
    print(f"In case you reach the goal for t <{TRAJ_LEN}, keep pressing RIGHT to remain in goal until t={TRAJ_LEN}.")
    print("The monster starts in a fixed state and moves either UP/LEFT. Provide demos accordingly to avoid the monster.")
    print('=============================================================================================================')
    while itraj < init_itraj + num_demos:
        print("Episode {}".format(itraj))

        obs, info = env.reset()
        print("Press RETURN to start or q to quit")
        while True:
            event = pygame.event.wait()
            if event.type == KEYDOWN and event.key == K_RETURN:
                break
            if event.type == KEYDOWN and event.key == K_q:
                quit()
        print("Started")

        tau = []
        total_reward = 0
        T_terminal = TRAJ_LEN

        for t in range(TRAJ_LEN):
            print(f"Action for robot at t={t}: ", end="", flush=True)
            while True:
                event = pygame.event.wait()
                if event.type == KEYDOWN:
                    if event.key in KEY_DICT.keys():
                        act_idx = KEY_DICT[event.key][0]
                        act_str = KEY_DICT[event.key][1]
                        print(act_str)
                        break

            act = act_idx
            obs_next, rew, term, trunc, info = env.step(act)
            tau.append((act, obs))
            print("Reward : ",rew)
            total_reward += rew
            if term or trunc:
                break
            else:
                obs = obs_next

        T_terminal = t + 1

        print("Press s to save or d to discard")
        while True:
            event = pygame.event.wait()
            if event.type == KEYDOWN and event.key == K_s:
                fname = f'{DEMODIR}/trajectory_{itraj}.pkl'
                itraj = itraj + 1
                saveObject(tau, fname)
                print('Saved', fname)
                break
            if event.type == KEYDOWN and event.key == K_d:
                print('Discarded')
                break
            
if __name__ == "__main__":
    main(itraj = 0,num_demos = 15,deterministic=False)