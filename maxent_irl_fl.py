from value_iteration import value_iteration
from neuralNet import *
from utils import *
from model import *
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import shap
import lime
from matplotlib import pyplot as plt
import statistics

def normalize(vals):
    """
    Normalize to [0, 1]

    input:
      vals: 1d array
    """
    min_val = np.min(vals)
    max_val = np.max(vals)
    return (vals - min_val) / (max_val - min_val)

def compute_state_visit_freq(P_a, gamma, trajs, policy, deterministic=True):
    """
        Compute the expected states visitation frequency p(s| theta, T)
            using algorithm 1 from Ziebart et al. 2008.

        inputs:
            P_a     NxNxN_ACTIONS matrix - transition dynamics
            gamma   float - discount factor
            trajs   list of Steps - collected from expert
            policy  Nx1 vector (or NxN_ACTIONS if deterministic=False) - policy

        returns:
            p       Nx1 vector - state visitation frequencies
        """
    N_STATES, _, N_ACTIONS = np.shape(P_a)
    T = len(trajs[0])  # Length of trajectories, fixed episode length
    # mu[s, t] is the prob of visiting state s at time t
    mu = np.zeros([N_STATES, T])

    with open("agent_feats.pkl", 'rb') as f:
        agent_feats = pickle.load(f)

    # Make sure to save trajectories with the corresponding monster and agent features for each state
    for tau in trajs:
        # Get current state index
        agent_feature = tau[0][1][0]  # Get agent location

        ai = agent_feats.index(agent_feature)
        si = ai   # Calculate state index

        mu[si, 0] = mu[si, 0] + 1  # Compute initial state visitation counts (i.e., t=0)

    mu[:, 0] = mu[:, 0] / len(trajs)
    for t in range(T-1):
        for s in range(N_STATES):
            if deterministic:
                ##### INSERT CODE HERE
                # Step 5 of algorithm 1 under a deterministic policy
                # if t == 0:
                #     continue
                
                # print(np.sum(mu[s,t]*P_a[s,:,int(policy[s])]) == stsum)
                # print(stsum)
                
                stsum=0
                for si in range(N_STATES):
                    stsum+=mu[si,t]*P_a[si,s,int(policy[si])]
                mu[s,t+1]+=stsum
                
            else:
                sum = 0
                for si in range(N_STATES):
                    for ai in range(N_ACTIONS):
                        sum += mu[si, t] * P_a[si, s, ai] * int(policy[si,ai])
                mu[s, t+1] += sum
                

    # Summing Frequencies across time
    p = np.sum(mu, 1)
    return p


def convert_feat(feat_map):
        # feat_map - (N, 24)
        
        N = feat_map.shape[0]
        
        feat_1 = np.argmax(feat_map[:, :-3], axis=1).reshape((N, 1))
        
        
        feat_2 = feat_map[:, -2].reshape((N, 1))
        feat_3 = feat_map[:, -1].reshape((N, 1))
        
        return np.concatenate([feat_1, feat_2, feat_3], axis=1)

def get_shapley_values(reward_net_new, feat_map):
    
    feat_map_torch = torch.Tensor(feat_map)
    explainer = shap.DeepExplainer(reward_net_new, feat_map_torch)
    torch.save(reward_net_new,'results/reward_network')

    shap_vals = explainer.shap_values(torch.Tensor(feat_map), check_additivity=False)
    shap_vals = shap_vals.squeeze()
    print(len(shap_vals))
    feat_1 = shap_vals[:,:-3]
    feat_1 = np.mean(feat_1,axis = 1,keepdims= True)
    
    
    feat_2 = shap_vals[:,-2].reshape(-1,1)
    feat_3 = shap_vals[:,-1].reshape(-1,1)

    final_shaps = np.concatenate([feat_1,feat_2,feat_3],axis = 1)
    
    shap.plots.violin(final_shaps,feature_names=['Agent Loc','Check Hole','Dist to Goal'],plot_type='violin',color_bar = True,show = False)
    plt.savefig('results/shapley_FL.png',bbox_inches='tight')


def deep_maxent_irl(feat_map, P_a, gamma, trajs, lr, n_iters,deterministic):
    """
        Maximum Entropy Deep Inverse Reinforcement Learning (Deep Maxent IRL)
        Please refer: https://arxiv.org/pdf/1507.04888.pdf

        inputs:
            feat_map    NxD matrix - the features for each state
            P_a         NxNxN_ACTIONS matrix - P_a[s0, s1, a] is the transition prob of
                                             landing at state s1 when taking action
                                             a at state s0
            gamma       float - RL discount factor
            trajs       a list of demonstrations
            lr          float - learning rate
            n_iters     int - number of optimization steps
            reward_net  neural Net object - the network used to predict rewards

        returns
            rewards     Nx1 vector - recovered state rewards
    """
    # n_iters = 100
    N_STATES, _, N_ACTIONS = np.shape(P_a)
    T = len(trajs[0]) # Length of trajectories, fixed episode length

    # init parameters
    feat_dim = feat_map.shape[1]
    
    with open("agent_feats.pkl", 'rb') as f:
        agent_feats = pickle.load(f)

    # Compute state visitation counts of the expert
    mu_exp = np.zeros([N_STATES])
    
    # for tau in trajs:
    #     print(tau[0])
    
    for t in range(T):
        for tau in trajs:
            # Get current state index
            # print(len(tau))
            
            agent_feature = tau[t][1][0]  # Get agent location

            ai = agent_feats.index(agent_feature)
            si = ai# Calculate state index

            mu_exp[si] = mu_exp[si] + 1

    mu_exp = mu_exp / len(trajs)

    # Create neural networks 
    reward_net = create_neural_net([10],feat_dim,1)
    reward_net_new = RewardNet(feat_dim)
    
    for i in range(n_iters):
        print('iteration: {}/{}'.format(i, n_iters))
        
        # 1) Compute the reward function with the neural network
        rew= forward_pass(reward_net,feat_map.T).reshape(N_STATES,1)
        
        # 2) Compute policy w.r.t. the reward function (HINT: use value_iteration func)
        _, pol = value_iteration(P_a,rew,gamma,error = 0.1,deterministic=deterministic)
        # print(pol.shape)
        # 3) Compute state visitation frequencies
        stvisit = compute_state_visit_freq(P_a,gamma,trajs,pol,deterministic=deterministic)

        # 4) Compute Final layer Gradient (HINT: Eq. 11 of Wulfmeier et al.)
        final_grad = (stvisit - mu_exp)

        # 5) Use Backprop to compute gradients of all network parameters
        grads = backprop(reward_net,feat_map.T,final_grad)
        
        # 6) Update network parameters with the computed gradient
        for i in range(len(reward_net)):
            reward_net[i][0] -= lr*grads[i][0]
            reward_net[i][1] -= lr*grads[i][1]

    # Update weights and biases in the new network
    rewards = forward_pass(reward_net, feat_map.T)
    with torch.no_grad():
        for i,layer in enumerate(reward_net_new.children()):
            if isinstance(layer, nn.Linear):
                layer.weight = nn.Parameter(torch.Tensor(reward_net[i][0]))
                layer.bias = nn.Parameter(torch.Tensor(reward_net[i][1]).reshape(-1))

    # Get shapley values using the pytorch network
    get_shapley_values(reward_net_new, feat_map)
    
    
    with open('results/feat_map_fl.pkl', 'wb') as f:
        pickle.dump(feat_map, f)

    return normalize(rewards)
