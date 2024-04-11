
import pickle
import numpy as np


def saveObject(tau, fname):
    with open(fname, 'wb') as f:
        pickle.dump(tau, f)


def loadObject(fname):
    with open(fname, 'rb') as f:
        tau = pickle.load(f)
    return tau


def load_VI_policy(fname):
    # Load the tabular value iteration policy (stored as numpy array)
    policy = np.load(fname)
    return policy


def add_dim_last(obs):
    return np.expand_dims(obs, 1)


def dist_to_goal(agent_loc, n_row=4, n_col=4):
    goal_loc = 15
    g_row, g_col = goal_loc // n_row, goal_loc % n_row
    a_row, a_col = agent_loc // n_row, agent_loc % n_col
    dist = np.abs(a_row - g_row) + np.abs(a_col - g_col)  # L1 dist
    return dist


def dist_to_monster(agent_loc, monster_loc, n_row=4, n_col=4):
    a_row, a_col = agent_loc // n_row, agent_loc % n_col
    m_row, m_col = monster_loc // n_row, monster_loc % n_col
    dist = np.abs(a_row - m_row) + np.abs(a_col - m_col)  # L1 dist
    # dist = dist/(n_row-1 + n_col -1)
    return np.array([dist])


def dist_to_hole(agent_loc, n_row=4, n_col=4):
    hole_locations = [5, 7, 11, 12]
    min_dist = 0.0
    for hole_loc in hole_locations:
        a_row, a_col = agent_loc // n_row, agent_loc % n_col
        m_row, m_col = hole_loc // n_row, hole_loc % n_col
        dist = np.abs(a_row - m_row) + np.abs(a_col - m_col)  # L1 dist
        dist = dist/(n_row-1 + n_col -1)
        min_dist = min(min_dist, dist)
    return np.array([min_dist])


def check_current_location(agent_loc):
    hole_locations = [5, 7, 11, 12]
    if agent_loc in hole_locations:
        return np.array([1.0])
    else:
        return np.array([0.0])


if __name__=="__main__":
    # Env dims
    n_row = 4
    n_col = 4
    # Create feature matrix
    all_agent_locs = np.arange(n_row*n_col).tolist()
    all_monster_locs = [2, 6, 8, 9, 10]
    hole_locations = [5, 7, 11, 12]

    # Save locations of agent and monster --> to be used when computing features in MaxEnt
    saveObject(all_agent_locs, "agent_feats.pkl")
    saveObject(all_monster_locs, "monster_feats.pkl")

    # Feature Engineering...
    n_additional_feats = 3  # Include binary variable to indicate if the agent is in a hole, and the dist to monster
    n_total_feats = len(all_agent_locs) + len(all_monster_locs) + n_additional_feats
    feat_map = np.zeros((len(all_agent_locs)*len(all_monster_locs), n_total_feats))

    print()
    for a_idx, a in enumerate(all_agent_locs):
        for m_idx, m in enumerate(all_monster_locs):
            s_idx = a_idx * len(all_monster_locs) + m_idx
            print("Agent Locations")
            print(np.eye(len(all_agent_locs))[a])
            print("Monster Locations")
            print(np.eye(len(all_monster_locs))[m_idx])
            feat_map[s_idx] = np.concatenate((np.eye(len(all_agent_locs))[a],
                                              np.eye(len(all_monster_locs))[m_idx],
                                              dist_to_hole(a),
                                              dist_to_monster(a, m),
                                              [dist_to_goal(a)]))

    with open("feat_map.npy", "wb") as f:
        np.save(f, feat_map)

    # Create Transition Probabilities Matrix
    monster_next_states = [[2], [2], [8], [8], [6, 9]]  # Monster can either move up or left

    # For each action, compute the next states of the agent
    agent_next_states = []
    for a in all_agent_locs:
        a_row, a_col = a // n_row, a % n_row
        left_row, left_col = (a_row, a_col-1 if 0<=a_col-1<n_col else a_col)
        up_row, up_col = (a_row - 1 if 0 <= a_row - 1 < n_row else a_row, a_col)
        right_row, right_col = (a_row, a_col+1 if 0<=a_col+1<n_col else a_col)
        down_row, down_col = (a_row + 1 if 0 <= a_row + 1 < n_row else a_row, a_col)

        agent_next_states.append([left_row*n_col+left_col,
                                  down_row * n_col + down_col,
                                  right_row*n_col+right_col,
                                  up_row*n_col+up_col])

    agent_next_states.pop(-1)
    agent_next_states.append([15, 15, 15, 15])  # After reaching goal, remain in goal

    actions = np.arange(4)  # 0:LEFT, 1:DOWN, 2:RIGHT, 3:UP
    trans_matrix = np.zeros((len(all_agent_locs)*len(all_monster_locs),
                             len(all_agent_locs)*len(all_monster_locs),
                             actions.size))

    for a_idx, a in enumerate(all_agent_locs):
        for action in actions:
            for m_idx, m in enumerate(all_monster_locs):
                curr_state_idx = a_idx * len(all_monster_locs) + m_idx
                next_monster_state = monster_next_states[m_idx]
                next_agent_state = agent_next_states[a_idx][action]
                next_state_idx = np.array([next_agent_state * len(all_monster_locs) + all_monster_locs.index(m_i)
                                           for m_i in next_monster_state])
                trans_matrix[curr_state_idx, next_state_idx, action] = 1.0 / next_state_idx.size

    with open("trans_matrix.npy", "wb") as f:
        np.save(f, trans_matrix)