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

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")  
print("device: ", device)

# Codel functions
# https://queue.acm.org/appendices/codel.html
def get_first_above_time_and_ok_to_drop(env, target_delay, interval_delay) -> (float, bool):
    ok_to_drop = False
    if env.inbuffer_info_node[0] == 0:
        first_above_time = 0
    else:
        sojourn_time = env.current_time - env.inbuffer_info_timestamp[0]
        if sojourn_time < target_delay:
            first_above_time = 0
        else:
            if first_above_time == 0:
                first_above_time = env.current_time + interval_delay
            elif env.current_time >= first_above_time:
                ok_to_drop = True
    return first_above_time, ok_to_drop

def get_next_drop_time(current_time, interval_delay, drop_count) -> float:
    return current_time + interval_delay / (drop_count**(1/2))

def do_deque(env, interval_delay, first_above_time, next_drop, drop_count, dropping_state, ok_to_drop) -> int:    # (dropping action)
    if env.inbuffer_info_node[0] == 0:
        dropping_state = False
        return 2    # Leave
    elif dropping_state:
        if ok_to_drop == False:
            # Sojourn time is below target_delay
            dropping_state = False
            return 0    # Forward
        elif env.current_time >= next_drop:
            if dropping_state:
                drop_count += 1
                if not ok_to_drop:
                    # Leave dropping state
                    dropping_state = False
                else:
                    # Schedule the next drop time
                    next_drop = get_next_drop_time(env.current_time, interval_delay, drop_count)
            return 1    # Drop
    elif ((ok_to_drop == True) 
          and ((env.current_time - next_drop < interval_delay)
          or (env.current_time - first_above_time >= interval_delay))):
        dropping_state = True
        if (env.current_time - next_drop < interval_delay):
            drop_count = drop_count - 2 if drop_count > 2 else 1
        else:
            drop_count = 1
        next_drop = get_next_drop_time(env.current_time, interval_delay, drop_count)
        return 1    # Drop
    else:
        return 0
                
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
        next_drop = interval_delay
        dropping_state = False
        drop_count = 1
        
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
            first_above_time, ok_to_drop = get_first_above_time_and_ok_to_drop(env, target_delay, interval_delay)
            action = do_deque(env, interval_delay, first_above_time, next_drop, drop_count, dropping_state, ok_to_drop)
            action = torch.tensor([action], dtype=torch.int64, device=device)    
        
        # print(f"selected_action: {selected_action}")
        next_state, reward_inst, _, _, info = env.step(action.item())
        
        # print(f"next_state: {next_state}")
        reward += reward_inst
        
        # info and reward_inst to dataframe
        df1 = pd.DataFrame(data=[[epoch, action.item(), env.leftbuffers, env.consumed_energy]],
                           columns=['epoch', 'action', 'left_buffer', 'consumed_energy'])
        df2 = pd.DataFrame(data=[env.current_aois], columns=[f"aoi{x}" for x in range(NUMNODES)])
        df3 = pd.concat([df1, df2], axis=1)
        df = pd.concat([df, df3], axis=0)
    return df, reward


# Test loop
test_num = 10
RAALGO = 'slottedaloha'
aqm_algorithms = ['deepaaqm', 'sred', 'codel']

test_env = ShowerEnv()
test_env.reset()
policy_net_deepaaqm = torch.load("policy_model_deepaaqm_" + RAALGO + "_" + str(NUMNODES) + ".pt")
policy_net_deepaaqm.eval()

rewards = np.zeros([2, test_num])
df_total = [pd.DataFrame() for x in range(len(aqm_algorithms))]

for iter in range(test_num):
    print(f"iter: {iter}")
    dflog = ra.randomaccess(NUMNODES, BEACONINTERVAL, FRAMETXSLOT, PER, RAALGO)
    dflog = dflog[dflog['result'] == 'succ']
    dflog = dflog.reset_index(drop=True)
    for i, simmode in enumerate(aqm_algorithms):
        df, reward = test_model(model=policy_net_deepaaqm, env=test_env, dflog=dflog, simmode=simmode)
        print(f"algorithm: {simmode}, reward: {reward}")
        df.insert(0, 'iteration', iter)
        df_total[i] = pd.concat([df_total[i], df], axis=0)

for i, simmode in enumerate(aqm_algorithms):
    filename = "test_log_" + RAALGO + "_" + simmode + ".csv"
    df_total[i].to_csv(filename)
        
# Make a dataframe for each algorithm
consumed_energy = pd.DataFrame(columns=aqm_algorithms)
for i, simmode in enumerate(aqm_algorithms):
    consumed_energy[simmode] = df_total[i][df_total[i].epoch==BEACONINTERVAL//TIMEEPOCH-1]['consumed_energy']/BEACONINTERVAL

print(consumed_energy)

# for i in range(3):
#     plt.plot(df_total[i].epoch, df_total[i].aoi0, label=aqm_algorithms[i])
# plt.legend()
# plt.show()

# for i in range(5):
#     st = 'aoi' + str(i)
#     plt.plot(df_total[0].epoch, df_total[0][st], label=st)
# plt.legend()
# plt.show()

# Draw a histogram plot for each algorithm using the 'aoi_max' column of df_total in one figure.
plt.figure(1)
plt.clf()
plt.xlabel('Max AoI')
plt.ylabel('Frequency')
plt.axvline(x=PEAKAOITHRES/BEACONINTERVAL, color='blue', linestyle='solid')
for i, simmode in enumerate(aqm_algorithms):
    counts, bins = np.histogram(df_total[i].iloc[:,5:], bins=100)
    cdf = np.cumsum(counts)/np.sum(counts)
    plt.plot(bins[:-1], cdf, label=simmode)
plt.legend()
plt.title(f"RA: {RAALGO}, Nodes: {NUMNODES}")
plt.savefig("test_log_" + RAALGO + "_" + str(NUMNODES) + ".png")
plt.show()

# # Draw a histogram plot for each algorithm using the 'aoi_mean' column of df_total in one figure.
# plt.figure(2)
# plt.clf()
# plt.xlabel('Mean AoI')
# plt.ylabel('Frequency')
# plt.axvline(x=PEAKAOITHRES/BEACONINTERVAL, color='blue', linestyle='solid')
# for i, simmode in enumerate(aqm_algorithms):
#     counts, bins = np.histogram(df_total[i]['aoi_mean'], bins=100)
#     cdf = np.cumsum(counts)/np.sum(counts)
#     plt.plot(bins[:-1], cdf, label=simmode)
# plt.legend()
# plt.savefig("test_log_" + RAALGO + "_aoi_mean.png")
# # plt.show()

# # Draw a histogram plot for each algorithm using the 'left_buffer' column of df_total in one figure.
# plt.figure(3)
# plt.clf()
# plt.xlabel('Left Buffer')
# plt.ylabel('Frequency')
# for i, simmode in enumerate(aqm_algorithms):
#     counts, bins = np.histogram(df_total[i]['left_buffer'], bins=100)
#     cdf = np.cumsum(counts)/np.sum(counts)
#     plt.plot(bins[:-1], cdf, label=simmode)
# plt.legend()
# plt.savefig("test_log_" + RAALGO + "_left_buffer.png")
# plt.show()

# Save the dataframe df_total into a csv file
for i, simmode in enumerate(aqm_algorithms):
    filename = f"test_log_{simmode}_{RAALGO}_{NUMNODES}.csv"
    df_total[i].to_csv(filename)
