import randomaccess as ra
from environment_rev import *
import numpy as np
import pandas as pd
import torch
from torch.distributions import Categorical
import random
# import csv

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")  
print("device: ", device)

ITERNUM = 1000

RAALGO = 'CSMA'
# RAALGO = 'slottedaloha'

BUFALGO = 'prop'
# BUFALGO = 'forward'
# BUFALGO = 'sred'
# BUFALGO = 'random'
# BUFALGO = 'rlaqm'

forwardprobability = 0.5

writing = 1

p_sred = 0
p_max = 0.15
totaltime = 0
maxrep = 1

df = pd.DataFrame()

for rep in range(maxrep):
    # env = gym.make('CartPole-v1')
    env = ShowerEnv()
    pi = Policy().to(device)
    score = 0.0
    print_interval = 10
    # dflog = ra.randomaccess(NUMNODES, BEACONINTERVAL, FRAMETXSLOT, PER, RAALGO)
    # dflog = dflog[dflog['result'] == 'succ']
    # dflog = pd.read_csv(f'dflog_{RAALGO}.csv')
    # df = pd.DataFrame()

    print(f"RA algorithm:{RAALGO}, Buffer algorithm:{BUFALGO}, Packet error rate:{PER}, Numnodes:{NUMNODES}")
    print(f"Peak AoI threshold = {PEAKAOITHRES*100}ms")
    print(f"Repetition: ({rep+1}/{maxrep})")
    print("=============================================")

    for n_epi in range(ITERNUM):
        s = env.reset()
        # s = torch.from_numpy(s).float().to(device)
        done = False
        blockcount = 0
        compositeaoi = 0
        score = 0
        a_set = np.zeros(0)

        dflog = ra.randomaccess(NUMNODES, BEACONINTERVAL, FRAMETXSLOT, PER, RAALGO)
        dflog = dflog[dflog['result'] == 'succ']
        link_utilization = dflog.shape[0] * FRAMETXSLOT * 9 / BEACONINTERVAL

        # while not done:  # One beacon interval
        for countindex in dflog.index:  # One beacon interval
            prob = pi(torch.from_numpy(s).float().to(device))
            # prob = pi(torch.from_numpy(s).float())
            if BUFALGO == 'prop':
                m = Categorical(prob)
                a = m.sample().to(device)
            elif BUFALGO == 'random':
                unifrv = random.random()
                if unifrv < forwardprobability:
                    a = torch.tensor(1, device=device)
                else:
                    a = torch.tensor(0, device=device)
            else:
                if env.previous_action == 0:
                    a = torch.tensor(2, device=device)
                else:
                    a = torch.tensor(0, device=device)

            if BUFALGO == 'sred':
                if env.qpointer < BUFFERSIZE/6:
                    p_sred = 0
                elif (env.qpointer >= BUFFERSIZE/6) and (env.qpointer < BUFFERSIZE/3):
                    p_sred = p_max/4
                else:
                    p_sred = p_max
                if p_sred < random.random():
                    env.probenqueue(dflog)
            else:
                if not env.previous_action == 0:
                    env.probenqueue(dflog)
            a_set = np.append(a_set, a.item())
            if BUFALGO == 'rlaqm':
                s_prime, r, done, info = env.step_rlaqm(a.item(), dflog, countindex, link_utilization)
            else:
                s_prime, r, done, info = env.step(a.item(), dflog, countindex)
            r = torch.tensor([r], device=device)
            pi.put_data((r, prob[a]))
            env.previous_action = a.item()
            s = s_prime
            score += r
        # blockcount = env.getblockcount()
        # aoi = env.getaoi()

        pi.train_net()

        count_forward = sum(a_set == 0)
        count_discard = sum(a_set == 1)
        count_skip = sum(a_set == 2)

        df1 = pd.DataFrame(data=[[n_epi, count_forward, count_discard, count_skip, env.txed.sum(), (score / print_interval),
                                  env.aoi[env.aoi != 0].mean() * BEACONINTERVAL / 1000,
                                  env.aoi.max() * BEACONINTERVAL / 1000, env.consumedenergy / BEACONINTERVAL]], index=[rep])
        # df = df.append(df1, sort=True, ignore_index=True)
        df = pd.concat([df, df1])

        if n_epi % print_interval == 0 and n_epi != 0:  # print된 값들을 csv로 만들 것.
            unique, counts = np.unique(a_set, return_counts=True)
            print(f"# of episode:{n_epi}, Channel:{env.channel}, F,D,S:{counts}, txed:{env.txed.sum()}, avg score:{int(score / print_interval)}, meanAoI:{env.aoi[env.aoi != 0].mean()*BEACONINTERVAL/1000:.2f}ms, maxAoI:{env.aoi.max()*BEACONINTERVAL/1000:.2f}ms, consumedPower:{env.consumedenergy/BEACONINTERVAL:.2f} Watt")
            print(f"Probabilities: {prob}")
            score = 0.0

df.columns = ["N_epi", "Forward", "Discard", "Skip", "Txed", "AvgScore", "MeanAoI(ms)", "MaxAoI(ms)", "AveConsPower(Watts)"]

if writing == 1:
    if BUFALGO == "prop":
        filename = f'result_{RAALGO}_{BUFALGO}_{PER}_{NUMNODES}_{velocity}_{int(PEAKAOITHRES*100)}'
    else:
        filename = f'result_{RAALGO}_{BUFALGO}_{PER}_{NUMNODES}'
    print(filename + ".csv")
    df.to_csv(filename + ".csv")
    torch.save(pi, filename + '.pt')

env.close()
# 100번 반복해서 돌리고 shade plot 할 수 있도록 csv파일 뽑아볼 것.