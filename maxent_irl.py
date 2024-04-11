
from value_iteration import value_iteration
from neuralNet import *
from utils import *
from model import *
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import shap
import lime
from matplotlib import pyplot as plt

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

    with open("monster_feats.pkl", 'rb') as f:
        monster_feats = pickle.load(f)

    # Make sure to save trajectories with the corresponding monster and agent features for each state
    for tau in trajs:
        # Get current state index
        agent_feature = tau[0][1][0]  # Get agent location
        monster_feature = tau[0][1][1]  # Get monster location

        ai = agent_feats.index(agent_feature)
        mi = monster_feats.index(monster_feature)
        si = ai * len(monster_feats) + mi  # Calculate state index

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
                
                ##### INSERT CODE HERE (Optional)
                # Step 5 of algorithm 1 under a stochastic policy
                pass

    # Summing Frequencies across time
    p = np.sum(mu, 1)
    return p


def maxent_irl(feat_map, P_a, gamma, trajs, lr, n_iters):
    """
        Maximum Entropy Inverse Reinforcement Learning (Maxent IRL)
        Please refer: https://www.cs.cmu.edu/~bziebart/publications/maxentirl-bziebart.pdf

        inputs:
            feat_map    NxD matrix - the features for each state
            P_a         NxNxN_ACTIONS matrix - P_a[s0, s1, a] is the transition prob of
                                             landing at state s1 when taking action
                                             a at state s0
            gamma       float - RL discount factor
            trajs       a list of demonstrations
            lr          float - learning rate
            n_iters     int - number of optimization steps

        returns
            rewards     Nx1 vector - recovered state rewards
    """
    N_STATES, _, N_ACTIONS = np.shape(P_a)

    # init parameters
    feat_dim = feat_map.shape[1]
    theta = np.random.uniform(low=-1, high=1, size=(feat_dim,))  # Initialize theta

    with open("agent_feats.pkl", 'rb') as f:
        agent_feats = pickle.load(f)

    with open("monster_feats.pkl", 'rb') as f:
        monster_feats = pickle.load(f)

    # Calculate feature expectations from expert demonstrations
    feat_exp = np.zeros(feat_dim)
    for episode in trajs:
        for step in episode:
            feat_temp = np.concatenate((np.eye(len(agent_feats))[step[1][0]],
                                        np.eye(len(monster_feats))[monster_feats.index(step[1][1])],
                                        check_current_location(step[1][0]),
                                        dist_to_monster(step[1][0], step[1][1])))
            feat_exp += feat_temp

    feat_exp = feat_exp / len(trajs)

    # training
    for i in range(n_iters):
        print('iteration: {}/{}'.format(i, n_iters))
        ##### INSERT CODE HERE
        # Training goes here
        # 1) Compute the reward function with theta
        reward = np.dot(feat_map,theta)
        # 2) Compute policy w.r.t. the reward function (HINT: use value_iteration func)=
        val,pol = value_iteration(P_a,reward.reshape(N_STATES,1),gamma,error = 0.1, deterministic=True)
        # 3) Compute state visitation frequencies
        stvisit = compute_state_visit_freq(P_a,gamma,trajs,pol)
        # 4) Compute Gradients (HINT: Eq. 6 of Zierbart et al.)
        grad = feat_exp - np.dot(stvisit,feat_map)
        # 5) Update theta
        theta = theta + lr*grad

    rewards = np.dot(feat_map, theta)

    return normalize(rewards)


def convert_feat(feat_map):
        # feat_map - (N, 24)
        
        monster_locs = [2, 6, 8, 9, 10]
        N = feat_map.shape[0]
        
        feat_1 = np.argmax(feat_map[:, :16], axis=1).reshape((N, 1))
        
        feat_2 = np.argmax(feat_map[:, 16:21], axis=1).reshape((N, 1))
        feat_2 = np.array([monster_locs[idx] for idx in feat_2.flatten()]).reshape((N, 1))
        
        feat_3 = feat_map[:, 21].reshape((N, 1))
        feat_4 = feat_map[:, 22].reshape((N, 1))
        feat_5 = feat_map[:, 23].reshape((N, 1))
        
        return np.concatenate([feat_1, feat_2, feat_3, feat_4, feat_5], axis=1)

def get_shapley_values(reward_net_new, feat_map):
    feat_map_torch = torch.Tensor(feat_map)
    explainer = shap.DeepExplainer(reward_net_new, feat_map_torch)
    torch.save(reward_net_new,'results/reward_network')

    shap_vals = explainer.shap_values(torch.Tensor(feat_map), check_additivity=False)
    shap_vals = shap_vals.squeeze()
    feat_1 = shap_vals[:,:16]
    feat_1 = np.mean(feat_1,axis = 1,keepdims= True)
    
    feat_2 = shap_vals[:,16:21]
    feat_2 = np.mean(feat_2,axis = 1,keepdims= True)
    
    feat_3 = shap_vals[:,21].reshape(-1,1)
    feat_4 = shap_vals[:,22].reshape(-1,1)
    feat_5 = shap_vals[:,23].reshape(-1,1)

    final_shaps = np.concatenate([feat_1,feat_2,feat_3,feat_4,feat_5],axis = 1)
    
    shap.plots.violin(final_shaps,feature_names=['Agent Loc','Monster Loc','Check Hole','Dist to Monster','Dist to Goal'],plot_type='violin',color_bar = True,show = False)
    plt.savefig('results/shapley.png',bbox_inches='tight')

def get_lime_values(reward_net_new, feat_map):
    def predict(feats):
        monster_locs = [2, 6, 8, 9, 10]
        reward_net_new.eval()
        feat_map = np.zeros((feats.shape[0],24))
        for i,feat in enumerate(feats):
            feat_map[i,int(feat[0])] = 1
            monster = min([2, 6, 8, 9, 10], key=lambda x: abs(x - feat[1]))
            feat_map[i,16 + monster_locs.index(monster)] = 1
            feat_map[i,21:] = feat[2:]
        logits = reward_net_new.forward(torch.Tensor(feat_map))
        return normalize(logits.detach().numpy())

    converted_feat_map = convert_feat(feat_map)
    lexp = lime.lime_tabular.LimeTabularExplainer(converted_feat_map,mode = 'regression',feature_names=  ['Agent Loc','Monster Loc','Check Hole','Dist to Monster','Dist to Goal'])
    explainers = []

    for i in converted_feat_map:
        e = lexp.explain_instance(i,predict,num_features = 5)
        explainers.append(e)
        
    dicts = [dict(i.as_map()[0]) for i in explainers]



    median_values = {}

    # Iterate through each dictionary in the list
    for d in dicts:
        # Iterate through each key-value pair in the dictionary
        for key, value in d.items():
            # If the key is not already in median_values, create a list
            if key not in median_values:
                median_values[key] = []
            # Append the value to the list for that key
            median_values[key].append(value)

    # Calculate the median for each key
    for key, values_list in median_values.items():
        median_values[key] = median(values_list)

    
    # average_dict = {key: value / num_dicts for key, value in sums.items()}
        
    map = {i:j for i,j in zip(lexp.feature_values,lexp.feature_names)}
    
    x = list(median_values.keys())
    y = list(median_values.values())
    x = [map[i] for i in x]
    plt.clf()
    plt.barh(x, y)

    plt.xlabel("Feature")
    plt.ylabel("Local explanation value")
    plt.title("Local explanation of model")

    for i, v in enumerate(y):
        if v < 0:
            plt.barh(x[i], v, color='red')
        else:
            plt.barh(x[i], v, color='green')

    exp = lexp.explain_instance(converted_feat_map[1], predict, num_features = 5)
    with open('results/explainer.pkl','wb') as f:
        pickle.dump(exp, f)
    
    # exp.as_pyplot_figure()
    plt.savefig('results/lime.png',bbox_inches = 'tight')

def deep_maxent_irl(feat_map, P_a, gamma, trajs, lr, n_iters):
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
    N_STATES, _, N_ACTIONS = np.shape(P_a)
    T = len(trajs[0])  # Length of trajectories, fixed episode length

    # init parameters
    feat_dim = feat_map.shape[1]
    
    with open("agent_feats.pkl", 'rb') as f:
        agent_feats = pickle.load(f)

    with open("monster_feats.pkl", 'rb') as f:
        monster_feats = pickle.load(f)

    # Compute state visitation counts of the expert
    mu_exp = np.zeros([N_STATES])

    for t in range(T):
        for tau in trajs:
            # Get current state index
            agent_feature = tau[t][1][0]  # Get agent location
            monster_feature = tau[t][1][1]  # Get monster location

            ai = agent_feats.index(agent_feature)
            mi = monster_feats.index(monster_feature)
            si = ai * len(monster_feats) + mi  # Calculate state index

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
        _, pol = value_iteration(P_a,rew,gamma,error = 0.1,deterministic=True)

        # 3) Compute state visitation frequencies
        stvisit = compute_state_visit_freq(P_a,gamma,trajs,pol,deterministic=True)

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
    
    # Get lime values
    get_lime_values(reward_net_new, feat_map)
    
    with open('results/feat_map.pkl', 'wb') as f:
        pickle.dump(feat_map, f)

    return normalize(rewards)
