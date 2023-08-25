import setting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

currenttime = 0
txprob = 1/setting.numnodes
print(f"Data frame length: {setting.txslot}")
num_succ = 0
num_coll = 0
succ_timestamp = np.array([], dtype=int)
succ_namestamp = np.array([], dtype=int)
retx = np.zeros(setting.numnodes, dtype=int)
aoi = np.zeros(setting.numnodes, dtype=int)
df = pd.DataFrame(columns=['time', 'node', 'aoi'])

while currenttime < setting.binterval:
    txprobarray = (np.random.rand(setting.numnodes) < txprob).astype(int)
    aoi[retx == 1] += setting.txslot * setting.slottime
    # Tx succ
    if txprobarray.sum() == 1:
        ind, = np.where(txprobarray == 1)
        print(f"Time: {currenttime}, Tx success from {ind} with AoI {aoi[ind]}")
        retx[ind] = 0
        aoi[ind] = 0
        succ_timestamp = np.append(succ_timestamp, currenttime)
        succ_namestamp = np.append(succ_namestamp, ind[0])
        num_succ += 1
        df2 = pd.DataFrame({'time': currenttime, 'node': [ind], 'aoi': [aoi[ind]]})
        df = pd.concat([df, df2], ignore_index=True, axis=0)
    # Tx idle
    elif txprobarray.sum() == 0:
        pass
    # Tx coll
    else:
        ind, = np.where(txprobarray == 1)
        print(f"Time: {currenttime}, Tx collision from {ind} with AoI {aoi[ind]}")
        retx[ind] = 1
        num_coll += 1
        df2 = pd.DataFrame({'time': currenttime, 'node': [ind], 'aoi': [aoi[ind]]})
        df = pd.concat([df, df2], ignore_index=True, axis=0)

    currenttime += setting.txslot * setting.slottime
    # currenttime += 1

print(f"# of succ: {num_succ*setting.txslot*setting.slottime}/{setting.binterval}")
plt.hist(np.diff(succ_timestamp))
plt.tight_layout()
# plt.savefig('slottedaloha.png')
# plt.show()
df.to_csv('slottedaloha.csv', index=False)