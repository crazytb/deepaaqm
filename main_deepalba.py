import randomaccess as ra
from environment import *
import numpy as np
import pandas as pd
import torch
from torch.distributions import Categorical
from torch import nn, optim
import random
import matplotlib
import matplotlib.pyplot as plt

from collections import namedtuple, deque
from itertools import count, chain

# import csv

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")  
print("device: ", device)

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

RAALGO = 'CSMA'

forwardprobability = 0.5
writing = 1
p_sred = 0
p_max = 0.15
totaltime = 0
maxrep = 1
df = pd.DataFrame()

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

env = ShowerEnv()
n_actions = env.action_space.n
state, info = env.reset()
n_observation = len(state)

policy_net = DQN(n_observation, n_actions).to(device)
target_net = DQN(n_observation, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=0.001, amsgrad=True)
BATCH_SIZE = 512
memory = ReplayMemory(BATCH_SIZE)

steps_done = 0
episode_rewards = []

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = 0.1
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # Return the action with the largest expected reward
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    
def plot_rewards(show_result=False):
    plt.figure(1)
    reward_t = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(reward_t.numpy())
    # Take 10 episode averages and plot them too
    if len(reward_t) >= 10:
        means = reward_t.unfold(0, 10, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(9), means))
        plt.plot(means.numpy())
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
        
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * 0.99) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    
num_episodes = 100

for i_episode in range(num_episodes):
    # Initialize the environment and state
    dflog = ra.randomaccess(NUMNODES, BEACONINTERVAL, FRAMETXSLOT, PER, RAALGO)
    dflog = dflog[dflog['result'] == 'succ']
    dflog = dflog.reset_index(drop=True)
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
    for epoch in dflog.index:
        print(f"Episode: {i_episode}, Epoch: {epoch}")
        # Select and perform an action
        action = select_action(state)
        print(f"State: {info}, Action: {action}")

        env.probenqueue(dflog)
        # observation, reward, done, info = env.step(action.item())
        observation, reward, done, info = env.step(0)
        reward = torch.tensor([reward], device=device)
        
        if done:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            # next_state = torch.tensor(env.flatten_dict_values(observation), dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*0.005 + target_net_state_dict[key]*(1-0.005)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_rewards.append(reward)
            plot_rewards()
    
print('Complete')
plot_rewards(show_result=True)
plt.ioff()
plt.show()
    

# for rep in range(maxrep):
#     # env = gym.make('CartPole-v1')
#     env = ShowerEnv()
#     pi = Policy().to(device)
#     score = 0.0
#     print_interval = 10
#     # dflog = ra.randomaccess(NUMNODES, BEACONINTERVAL, FRAMETXSLOT, PER, RAALGO)
#     # dflog = dflog[dflog['result'] == 'succ']
#     # dflog = pd.read_csv(f'dflog_{RAALGO}.csv')
#     # df = pd.DataFrame()

#     print(f"RA algorithm:{RAALGO}, Buffer algorithm:{BUFALGO}, Packet error rate:{PER}, Numnodes:{NUMNODES}")
#     print(f"Peak AoI threshold = {PEAKAOITHRES*100}ms")
#     print(f"Repetition: ({rep+1}/{maxrep})")
#     print("=============================================")

#     for n_epi in range(ITERNUM):
#         s = env.reset()
#         # s = torch.from_numpy(s).float().to(device)
#         done = False
#         blockcount = 0
#         compositeaoi = 0
#         score = 0
#         a_set = np.zeros(0)

#         dflog = ra.randomaccess(NUMNODES, BEACONINTERVAL, FRAMETXSLOT, PER, RAALGO)
#         dflog = dflog[dflog['result'] == 'succ']
#         link_utilization = dflog.shape[0] * FRAMETXSLOT * 9 / BEACONINTERVAL

#         # while not done:  # One beacon interval
#         for countindex in dflog.index:  # One beacon interval
#             prob = pi(torch.from_numpy(s).float().to(device))
#             # prob = pi(torch.from_numpy(s).float())
#             if BUFALGO == 'prop':
#                 m = Categorical(prob)
#                 a = m.sample().to(device)
#             elif BUFALGO == 'random':
#                 unifrv = random.random()
#                 if unifrv < forwardprobability:
#                     a = torch.tensor(1, device=device)
#                 else:
#                     a = torch.tensor(0, device=device)
#             else:
#                 if env.previous_action == 0:
#                     a = torch.tensor(2, device=device)
#                 else:
#                     a = torch.tensor(0, device=device)

#             if BUFALGO == 'sred':
#                 if env.qpointer < BUFFERSIZE/6:
#                     p_sred = 0
#                 elif (env.qpointer >= BUFFERSIZE/6) and (env.qpointer < BUFFERSIZE/3):
#                     p_sred = p_max/4
#                 else:
#                     p_sred = p_max
#                 if p_sred < random.random():
#                     env.probenqueue(dflog)
#             else:
#                 if not env.previous_action == 0:
#                     env.probenqueue(dflog)
#             a_set = np.append(a_set, a.item())
#             if BUFALGO == 'rlaqm':
#                 s_prime, r, done, info = env.step_rlaqm(a.item(), dflog, countindex, link_utilization)
#             else:
#                 s_prime, r, done, info = env.step(a.item(), dflog, countindex)
#             r = torch.tensor([r], device=device)
#             pi.put_data((r, prob[a]))
#             env.previous_action = a.item()
#             s = s_prime
#             score += r
#         # blockcount = env.getblockcount()
#         # aoi = env.getaoi()

#         pi.train_net()

#         count_forward = sum(a_set == 0)
#         count_discard = sum(a_set == 1)
#         count_skip = sum(a_set == 2)

#         df1 = pd.DataFrame(data=[[n_epi, count_forward, count_discard, count_skip, env.txed.sum(), (score / print_interval),
#                                   env.aoi[env.aoi != 0].mean() * BEACONINTERVAL / 1000,
#                                   env.aoi.max() * BEACONINTERVAL / 1000, env.consumedenergy / BEACONINTERVAL]], index=[rep])
#         # df = df.append(df1, sort=True, ignore_index=True)
#         df = pd.concat([df, df1])

#         if n_epi % print_interval == 0 and n_epi != 0:  # print된 값들을 csv로 만들 것.
#             unique, counts = np.unique(a_set, return_counts=True)
#             print(f"# of episode:{n_epi}, Channel:{env.channel}, F,D,S:{counts}, txed:{env.txed.sum()}, avg score:{int(score / print_interval)}, meanAoI:{env.aoi[env.aoi != 0].mean()*BEACONINTERVAL/1000:.2f}ms, maxAoI:{env.aoi.max()*BEACONINTERVAL/1000:.2f}ms, consumedPower:{env.consumedenergy/BEACONINTERVAL:.2f} Watt")
#             print(f"Probabilities: {prob}")
#             score = 0.0

# df.columns = ["N_epi", "Forward", "Discard", "Skip", "Txed", "AvgScore", "MeanAoI(ms)", "MaxAoI(ms)", "AveConsPower(Watts)"]

# if writing == 1:
#     if BUFALGO == "prop":
#         filename = f'result_{RAALGO}_{BUFALGO}_{PER}_{NUMNODES}_{velocity}_{int(PEAKAOITHRES*100)}'
#     else:
#         filename = f'result_{RAALGO}_{BUFALGO}_{PER}_{NUMNODES}'
#     print(filename + ".csv")
#     df.to_csv(filename + ".csv")
#     torch.save(pi, filename + '.pt')

# env.close()
# # 100번 반복해서 돌리고 shade plot 할 수 있도록 csv파일 뽑아볼 것.