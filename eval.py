import os
import numpy as np
import env as det
import env_copy as stoch
from utils import load_VI_policy, loadObject
import argparse

agent_feats = loadObject("agent_feats.pkl")
monster_feats = loadObject("monster_feats.pkl")
trans_matrix = np.load("trans_matrix.npy")


def get_episode_returns(env_version, policy, n_episodes, deterministic=True, max_ep_len=10, render_human=False,
                        seed=789):
    np.random.seed(seed)
    if deterministic == True:
        env = det.IRLEnv(
            render_mode="human" if render_human else None,
            seed=seed + 765,
            version=env_version,
        )
    else:
        env = stoch.IRLEnv(
            render_mode="human" if render_human else None,
            seed=seed + 765,
            version=env_version,
        )
    
    ep_returns = []

    # Main loop
    
    wins = 0
    for i in range(n_episodes):

        # collect data
        obs, info = env.reset()
        agent_loc, monster_loc = obs
        total_reward = 0
        for t in range(max_ep_len):
            ai = agent_feats.index(agent_loc)
            mi = monster_feats.index(monster_loc)

            si = ai * len(monster_feats) + mi
            act = int(np.argmax(policy[si]))
            # print(act)

            obs_next, rew, term, trunc, info = env.step(act)
            agent_loc, monster_loc = obs_next
            total_reward += rew

            # Determine if the new state is a terminal state. If so, then quit
            # the game. If not, step forward into the next state.
            if term or trunc or rew==50:
                # The next state is a terminal state (goal / hole). Therefore, we should
                # record the outcome of the game in winLossRecord for game i.
                if rew == 50:
                    wins += 1
                break
            else:
                # Simply step to the next state
                obs = obs_next

        ep_returns.append(total_reward)

    return ep_returns,wins

def main(deterministic = True,fname = 'trained_policy_deep.npy',render = 0,return_wins = False):
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--fname', type=str, default="trained_policy_deep.npy", help='Path to trained policy file')
    # parser.add_argument('--render', type=int, default=0, help='set to 1 to visualize trained policy')

    render_human = bool(render)
    assert os.path.exists(fname), f"Policy file '{fname}' does not exist"
    env_version = 3  # DO NOT CHANGE
    if render_human == True:
        n_episodes = 5
    else:
        n_episodes = 1000

    policy = load_VI_policy(fname)
    ep_returns,wins = get_episode_returns(env_version, policy, deterministic=deterministic, n_episodes=n_episodes, render_human=render_human,max_ep_len=15)
    if return_wins:
        return wins/n_episodes
    print('No Episodes:', n_episodes)
    print('Mean Return:', np.mean(ep_returns))
    print('Wins:', wins)
    
if __name__ == '__main__':
    main(render = 1,deterministic=False)