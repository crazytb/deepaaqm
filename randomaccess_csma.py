import setting
import numpy as np
import pandas as pd


def csma(numnodes):
    interval = setting.binterval
    txslot = setting.txslot
    slottime = setting.slottime

    currenttime = 0
    print(f"Data frame length: {txslot}")
    contentionwindowsize = [2**x for x in range(5, 11)]
    # Backoff counter initialize
    bo_stage = np.zeros(numnodes, dtype=int)
    bo_counter = np.array([np.random.randint(contentionwindowsize[x]) for x in bo_stage])
    channel_state = 0  # 0: Idle, 1: busy

    num_succ = 0
    num_coll = 0
    succ_timestamp = np.array([], dtype=int)
    succ_namestamp = np.array([], dtype=int)
    retx = np.zeros(numnodes, dtype=int)
    aoi = np.zeros(numnodes, dtype=int)
    df = pd.DataFrame(columns=['time', 'node', 'aoi'])

    while currenttime < interval:
        # Idle
        if np.min(bo_counter) != 0:
            aoi[retx == 1] += np.min(bo_counter)
            currenttime += np.min(bo_counter)
            bo_counter = bo_counter - np.min(bo_counter)
        # Tx succ
        elif np.min(bo_counter) == 0 and (np.size(bo_counter) - np.count_nonzero(bo_counter) == 1):
            aoi[retx == 1] += txslot * slottime
            ind, = np.where(bo_counter == 0)
            bo_counter[ind] = np.random.randint(contentionwindowsize[0])
            print(f"Time: {currenttime}, Tx success from {ind} with AoI {aoi[ind]}")
            retx[ind] = 0
            aoi[ind] = 0
            currenttime += txslot*slottime
            succ_timestamp = np.append(succ_timestamp, currenttime)
            succ_namestamp = np.append(succ_namestamp, ind[0])
            num_succ += 1
            df2 = pd.DataFrame({'time': currenttime, 'node': [ind], 'aoi': [aoi[ind]]})
            df = pd.concat([df, df2], ignore_index=True, axis=0)
        # Tx coll
        elif np.min(bo_counter) == 0 and (np.size(bo_counter) - np.count_nonzero(bo_counter) > 1):
            aoi[retx == 1] += txslot * slottime
            ind, = np.where(bo_counter == 0)
            for x in ind:
                if bo_stage[x] < 5:
                    bo_stage[x] += 1
                bo_counter[x] = np.random.randint(contentionwindowsize[bo_stage[x]])
            print(f"Time: {currenttime}, Tx collision from {ind} with AoI {aoi[ind]}")
            retx[ind] = 1
            num_coll += 1
            currenttime += txslot*slottime
            df2 = pd.DataFrame({'time': currenttime, 'node': [ind], 'aoi': [aoi[ind]]})
            df = pd.concat([df, df2], ignore_index=True, axis=0)

    print(f"# of succ: {num_succ*txslot*slottime}/{interval}")
    df.to_csv('csma.csv', index=False)
