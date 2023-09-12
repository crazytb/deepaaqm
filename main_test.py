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

# Make test model for one episode
def test_model(model, env, dflog, simmode):
    df = pd.DataFrame()
    next_state, info = env.reset()
    reward = 0
    for epoch in range(BEACONINTERVAL//TIMEEPOCH):
        env.probenqueue(dflog)
        
        # Codel parameters
        target_delay = 500    # microseconds
        interval_delay = 10000 / BEACONINTERVAL
        timer = 0
        dropping_state = False
        
        # 0: Forward, 1: Discard, 2: Skip
        if simmode == "deepaaqm" or simmode == "rlaqm":
            action = model.forward(torch.tensor(next_state, dtype=torch.float32, device=device)).max(0)[1].view(1, 1)
        
        elif simmode == "sred":
            q_val = 1 - (env.leftbuffers / BUFFERSIZE)
            if 0 <= q_val <= 1/6:
                action = torch.tensor([0], dtype=torch.int64, device=device)
            elif 1/6 < q_val <= 2/6:
                prob = 0.15/4
                if prob > random.random():
                    action = torch.tensor([1], dtype=torch.int64, device=device)
                else:
                    action = torch.tensor([0], dtype=torch.int64, device=device)
            elif 2/6 < q_val <= 1:
                prob = 0.15
                if prob > random.random():
                    action = torch.tensor([1], dtype=torch.int64, device=device)
                else:
                    action = torch.tensor([0], dtype=torch.int64, device=device)
                    
        elif simmode == "codel":
            
            if (dropping_state == True
                and env.inbuffer_info_node[0] != 0
                and env.current_time - env.inbuffer_info_timestamp[0]/BEACONINTERVAL):
            
            
            
            if env.inbuffer_info_node[0] != 0:
                departure_time = env.inbuffer_info_timestamp[0]
            
            
            
            if env.inbuffer_info_timestamp[0] < target_delay:
                action = torch.tensor([0], dtype=torch.int64, device=device)
            else:
                action = torch.tensor([0], dtype=torch.int64, device=device)
                timer += TIMEEPOCH 
                
        
        # print(f"selected_action: {selected_action}")
        next_state, reward_inst, _, _, info = env.step(action.item())
        
        # print(f"next_state: {next_state}")
        reward += reward_inst
        
        # info and reward_inst to dataframe
        df1 = pd.DataFrame(data=[[epoch, action.item(), env.leftbuffers, env.consumed_energy, env.current_aois.max(), env.current_aois.mean()]],
                           columns=['epoch', 'action', 'left_buffer', 'consumed_energy', 'aoi_max', 'aoi_mean'])
        df = pd.concat([df, df1], axis=0)
    return df, reward

# Test loop
test_num = 10
RAALGO = 'slottedaloha'

test_env = ShowerEnv()
policy_net_deepaaqm = torch.load("policy_model_deepaaqm_" + RAALGO + ".pt")
policy_net_deepaaqm.eval()

rewards = np.zeros([2, test_num])
df_total = [pd.DataFrame() for x in range(2)]

for iter in range(test_num):
    print(f"iter: {iter}")
    dflog = ra.randomaccess(NUMNODES, BEACONINTERVAL, FRAMETXSLOT, PER, RAALGO)
    dflog = dflog[dflog['result'] == 'succ']
    dflog = dflog.reset_index(drop=True)
    for i, simmode in enumerate(['deepaaqm', 'sred']):
        df, reward = test_model(model=policy_net_deepaaqm, env=test_env, dflog=dflog, simmode=simmode)
        print(f"algorithm: {simmode}, reward: {reward}")
        df.insert(0, 'iteration', iter)
        df_total[i] = pd.concat([df_total[i], df], axis=0)

for i, simmode in enumerate(['deepaaqm', 'sred']):
    filename = "test_log_" + RAALGO + "_" + simmode + ".csv"
    df_total[i].to_csv(filename)
        
    #     # Plot rewards
    #     plt.figure()
    #     plt.clf()
    #     rewards_t = torch.tensor(reward, dtype=torch.float)
    #     plt.xlabel('Episode #')
    #     plt.ylabel('Return')
    #     plt.plot(rewards_t.numpy())

    #     means = rewards_t.unfold(0, 20, 1).mean(1).view(-1)
    #     means = torch.cat((torch.zeros(19), means))
    #     plt.plot(means.numpy())
    #     # Save plot into files
    #     filename = simmode + "_test_rewards.png"
    #     plt.savefig(filename)

    # plt.show()


# for simmode in ["deepaaqm", "red", "rlaqm"]:
#     dflog = ra.randomaccess(NUMNODES, BEACONINTERVAL, FRAMETXSLOT, PER, 'CSMA')
#     dflog = dflog[dflog['result'] == 'succ']
#     dflog = dflog.reset_index(drop=True)

#     df, rewards = test_model(net=policy_net_deepaaqm, env=test_env, dflog=dflog, iterations=200, simmode=simmode)
#     filename = simmode + "_test_log.csv"
#     df.to_csv(filename)


