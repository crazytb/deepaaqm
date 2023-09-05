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
    for i in range(BEACONINTERVAL//TIMEEPOCH):
        env.probenqueue(dflog)
        # 0: Forward, 1: Discard, 2: Skip
        if simmode == "deepaaqm" or simmode == "rlaqm":
            action = model.forward(torch.tensor(next_state, dtype=torch.float32, device=device)).max(0)[1].view(1, 1)
        # selected_action = 1 only if the channel quality is good
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
        # print(f"selected_action: {selected_action}")
        next_state, reward_inst, terminated, truncated, info = env.step(action.item())
        # print(f"next_state: {next_state}")
        reward += reward_inst
        # info and reward_inst to dataframe
        df_misc = pd.DataFrame(data=[[i, action.item(), reward_inst, reward]], 
                            columns=['epoch', 'action', 'reward_inst', 'reward'])
        df_data = pd.DataFrame(data=[info.values()], columns=info.keys())
        
        df1 = pd.concat([df_misc, df_data], axis=1)
        df = pd.concat([df, df1], axis=0)
    return df, reward

# Define test env and test model    
test_env = ShowerEnv()
policy_net_deepaaqm = torch.load("policy_model_deepaaqm_keep.pt")
policy_net_deepaaqm.eval()
# policy_net_rlaqm = torch.load("rlaqm.pt")

# Test loop
dflog = ra.randomaccess(NUMNODES, BEACONINTERVAL, FRAMETXSLOT, PER, 'CSMA')
dflog = dflog[dflog['result'] == 'succ']
dflog = dflog.reset_index(drop=True)
df, rewards = test_model(model=policy_net_deepaaqm, env=test_env, dflog=dflog, simmode="deepaaqm")
filename = "deepaaqm" + "_test_log.csv"
df.to_csv(filename)

# for simmode in ["deepaaqm", "red", "rlaqm"]:
#     dflog = ra.randomaccess(NUMNODES, BEACONINTERVAL, FRAMETXSLOT, PER, 'CSMA')
#     dflog = dflog[dflog['result'] == 'succ']
#     dflog = dflog.reset_index(drop=True)

#     df, rewards = test_model(net=policy_net_deepaaqm, env=test_env, dflog=dflog, iterations=200, simmode=simmode)
#     filename = simmode + "_test_log.csv"
#     df.to_csv(filename)

#     # Plot rewards
#     plt.figure()
#     plt.clf()
#     rewards_t = torch.tensor(rewards, dtype=torch.float)
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
