import randomaccess as ra
from environment_rev import *
import numpy as np
import pandas as pd
import torch
from torch.distributions import Categorical
import csv

ITERNUM = 200
RAALGO = 'CSMA'
# RAALGO = 'slottedaloha'
BUFALGO = 'RED'  # RED, CODEL, PIE
thmin = int(BUFFERSIZE/3)
thmax = 2*int(BUFFERSIZE/3)
p_discard = 0.1
current_q_depth = 0

score = 0.0
print_interval = 10
df = pd.DataFrame()

print(f"RA algorithm:{RAALGO}, Buffer algorithm:{BUFALGO}, Packet error rate:{PER}")
print("=============================================")
for n_epi in range(ITERNUM+1):

    dflog = ra.randomaccess(NUMNODES, BEACONINTERVAL, FRAMETXSLOT, PER, RAALGO)
    dflog = dflog[dflog['result'] == 'succ']


    for countindex in dflog.index:  # One beacon interval
        if current_q_depth < thmin:


    #     prob = pi(torch.from_numpy(s).float())
    #     m = Categorical(prob)
    #     a = m.sample()
    #     if BUFALGO == 'prop':
    #         pass
    #     elif BUFALGO == 'forward':
    #         a = torch.tensor(0)
    #     env.probenqueue(dflog)
    #     a_set = np.append(a_set, a.item())
    #     s_prime, r, done, info = env.step(a.item(), dflog, countindex)
    #     pi.put_data((r, prob[a]))
    #     s = s_prime
    #     score += r
    # # blockcount = env.getblockcount()
    # # aoi = env.getaoi()
    #
    # pi.train_net()

    if n_epi % print_interval == 0 and n_epi != 0:  # print된 값들을 csv로 만들 것.
        # print(
        #     f"(POWERCOEFF:{POWERCOEFF}, AOICOEFF:{AOICOEFF})# of episode:{n_epi}, blockcount:{blockcount / print_interval}, meanAoI:{aoi.mean():.2f}, avg score:{score / print_interval:.2f}")
        # print(f"# of episode:{n_epi}, meanAoI:{(aoi.mean()):.2f}, avg score:{(score/print_interval):.2f}")
        unique, counts = np.unique(a_set, return_counts=True)
        df1 = pd.DataFrame(data=[[counts, env.txed.sum(), (score / print_interval), env.aoi.mean()*BEACONINTERVAL/1000, env.aoi.mean(axis=0).max()*BEACONINTERVAL/1000, env.consumedenergy/BEACONINTERVAL]], index=[n_epi])
        df = df.append(df1, sort=True, ignore_index=True)
        print(f"# of episode:{n_epi},"
              f"F,D,S:{counts},"
              f"txed:{env.txed.sum()},"
              f"avg score:{(score / print_interval):.2f},"
              f"meanAoI:{env.aoi.mean()*BEACONINTERVAL/1000:.2f}ms,"
              f"maxAoI:{env.aoi.mean(axis=0).max()*BEACONINTERVAL/1000:.2f}ms,"
              f"consumedPower:{env.consumedenergy/BEACONINTERVAL:.2f} Watt")

        score = 0.0

df.columns = ["Counts", "Txed", "AvgScore", "MeanAoI", "MaxAoI", "AveConsPower"]
df.to_csv(f'result_{RAALGO}_{BUFALGO}_{PER}.csv')
env.close()
