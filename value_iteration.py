"""
Tabular Value iteration function for solving the MDP to get optimal policy
"""

import math
import numpy as np


def value_iteration(P_a, rewards, gamma, error=0.1, deterministic=True):
    """
        Static value iteration function.

        inputs:
            P_a         NxNxN_ACTIONS transition probabilities matrix -
                                  P_a[s0, s1, a] is the transition prob of
                                  landing at state s1 when taking action
                                  a at state s0
            rewards     Nx1 matrix - rewards for all the states
            gamma       float - RL discount
            error       float - threshold for a stop
            deterministic   bool - to return deterministic policy or stochastic policy

        returns:
            values    Nx1 matrix - estimated values
            policy    Nx1 (NxN_ACTIONS if non-det) matrix - policy
    """
    N_STATES, _, N_ACTIONS = np.shape(P_a)
    trans = np.zeros([N_STATES, N_ACTIONS], dtype=int)
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            s_prime = np.argmax(P_a[s, :, a])
            trans[s, a] = s_prime

    values = np.zeros([N_STATES])

    # estimate values
    while True:
        values_tmp = values.copy()
        for s in range(N_STATES):
            values[s] = max([rewards[s] + gamma * values_tmp[trans[s, a]] for a in range(N_ACTIONS)])

        max_diff = max([abs(values[s] - values_tmp[s]) for s in range(N_STATES)])
        # print("max_diff in VI:", max_diff)
        if max_diff < error:
            break

    if deterministic:
        # generate deterministic policy
        policy = np.zeros([N_STATES])
        for s in range(N_STATES):
            policy[s] = np.argmax([sum([P_a[s, s1, a] * (rewards[s] + gamma * values[s1])
                                        for s1 in range(N_STATES)])
                                        for a in range(N_ACTIONS)])
            

        return values, policy
    else:
        # generate stochastic policy
        policy = np.zeros([N_STATES, N_ACTIONS])
        for s in range(N_STATES):
            policy[s,:] = np.array([sum([P_a[s, s1, a] * (rewards[s] + gamma * values[s1]) 
                                 for s1 in range(N_STATES)]) for a in range(N_ACTIONS)]).reshape(-1,)
            # print(v_s)
            
        
        policy -= policy.max(axis=1).reshape((N_STATES, 1))  # For numerical stability.
        policy = np.exp(policy)/np.exp(policy).sum(axis=1).reshape((N_STATES, 1))
        return values, policy

# def logsumexp(_arr, axis=-1):
#     arrc = _arr.copy()
#     arrc = np.exp(arrc)
#     arrc = np.sum(arrc, axis=axis)
#     arrc = np.log(arrc)
#     return arrc

# copied from scipy source
def logsumexp(a, axis=None, keepdims=False, return_sign=False):
    a_max = np.amax(a, axis=axis, keepdims=True)

    if a_max.ndim > 0:
        a_max[~np.isfinite(a_max)] = 0
    elif not np.isfinite(a_max):
        a_max = 0

    tmp = np.exp(a - a_max)

    # suppress warnings about log of zero
    with np.errstate(divide='ignore'):
        s = np.sum(tmp, axis=axis, keepdims=keepdims)
        if return_sign:
            sgn = np.sign(s)
            s *= sgn  # /= makes more sense but we need zero -> zero
        out = np.log(s)

    if not keepdims:
        a_max = np.squeeze(a_max, axis=axis)
    out += a_max

    if return_sign:
        return out, sgn
    else:
        return out

def normalize(vals):
    min_val = np.min(vals)
    max_val = np.max(vals)
    return (vals - min_val) / (max_val - min_val)

def soft_value_iteration(P_a, rewards, gamma, error=0.1, deterministic=True):
    """
        Static value iteration function.

        inputs:
            P_a         NxNxN_ACTIONS transition probabilities matrix -
                                  P_a[s0, s1, a] is the transition prob of
                                  landing at state s1 when taking action
                                  a at state s0
            rewards     Nx1 matrix - rewards for all the states
            gamma       float - RL discount
            error       float - threshold for a stop
            deterministic   bool - to return deterministic policy or stochastic policy

        returns:
            values    Nx1 matrix - estimated values
            policy    Nx1 (NxN_ACTIONS if non-det) matrix - policy
    """
    np.set_printoptions(suppress=True)
    N_STATES, _, N_ACTIONS = np.shape(P_a)

    # def term_zero_set(arr):
    #     hgposs = [5, 7, 11, 12, 15]
    #     arr = arr.reshape(16, 5)
    #     # holes + goal
    #     arr[hgposs] = 0
    #     # monster, agent in same loc
    #     mposs = [2, 6, 8, 9, 10]
    #     for mpos, apos in enumerate(mposs):
    #         arr[apos, mpos] = 0
    #     arr = arr.reshape(-1)

    # add a terminating state extra
    N_STATES += 1
    rewards = np.concatenate([rewards, [0.0]])
    # P_a[s, ns, a]
    P_a_ex = np.zeros((N_STATES, N_STATES, N_ACTIONS))
    P_a_ex[:N_STATES-1, :N_STATES-1, :] = P_a.copy()
    P_a_ex[N_STATES-1, N_STATES-1, :] = 1.0

    # holes
    hgposs = np.array([5, 7, 11, 12, 15]).reshape(5, -1)*5 + np.arange(5).reshape(1, 5)
    hgposs = hgposs.reshape(-1)
    P_a_ex[hgposs, :, :] = 0.0
    P_a_ex[hgposs, N_STATES-1, :] = 1.0
    # monster, agent in same loc
    mposs = np.array([2, 6, 8, 9, 10])*5 + np.arange(5)
    P_a_ex[mposs, :, :] = 0.0
    P_a_ex[mposs, N_STATES-1, :] = 1.0
    P_a = P_a_ex

    def term_zero_set(arr):
        arr[-1] = 0

    values = np.ones([N_STATES])*-1e4
    qs = np.zeros([N_STATES, N_ACTIONS])

    # estimate values
    while True:
        values_tmp = values.copy()

        # set terminal states value to 0
        term_zero_set(values)

        qs = rewards.reshape(-1, 1) + gamma * P_a.transpose((0, 2, 1))@values
        values = logsumexp(qs, axis=1)

        # print(values.max())

        max_diff = np.abs(values - values_tmp).max()
        # print(max_diff)
        if max_diff < error:
            break
    # print(max_diff)
    # print(values.max())

    qs = qs[:-1, :]
    values = values[:-1]
    policy = np.exp(qs - values.reshape(-1, 1))

    if deterministic:
        return values, policy.argmax(-1)
    else:
        return values, policy
