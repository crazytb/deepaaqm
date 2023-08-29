import numpy as np
import pandas as pd
import random


def randomaccess(numnodes, beaconinterval, frametxslot, per, raalgo):
    if raalgo == 'CSMA':
        slottime = 9
        currentslot = 0
        # print(f"Data frame length: {frametxslot}")
        contentionwindowsize = [2**x for x in range(5, 11)]
        # Backoff counter initialize
        bo_stage = np.zeros(numnodes, dtype=int)
        bo_counter = np.array([np.random.randint(contentionwindowsize[x]) for x in bo_stage])

        num_succ = 0
        num_fail = 0
        num_coll = 0
        succ_timestamp = np.array([], dtype=int)
        succ_namestamp = np.array([], dtype=int)
        retx = np.zeros(numnodes, dtype=int)
        aoitimestamp = np.zeros(numnodes, dtype=int)
        df = pd.DataFrame(columns=['time', 'node', 'aoi', 'result'])

        while currentslot < beaconinterval/slottime - frametxslot:
            per_rv = random.random()
            # Idle
            if np.min(bo_counter) != 0:
                currentslot += np.min(bo_counter)
                aoitimestamp[retx == 0] = currentslot
                bo_counter = bo_counter - np.min(bo_counter)
            # Tx succ
            elif (per_rv > per) and ((np.min(bo_counter) == 0) and (np.size(bo_counter) - np.count_nonzero(bo_counter) == 1)):
                currentslot += frametxslot
                ind, = np.where(bo_counter == 0)
                retx[ind] = 0
                # print(f"Time: {currentslot}, Tx success from {ind+1} with AoI {aoi[ind]}")
                df2 = pd.DataFrame({'time': currentslot*slottime, 'node': ind, 'aoi': aoitimestamp[ind]*slottime, 'result': 'succ'})
                df = pd.concat([df, df2], ignore_index=True, axis=0)
                aoitimestamp[retx == 0] = currentslot
                succ_timestamp = np.append(succ_timestamp, currentslot)
                succ_namestamp = np.append(succ_namestamp, ind[0])
                bo_stage[ind] = 0
                bo_counter[ind] = np.random.randint(contentionwindowsize[0])
                num_succ += 1
            # Tx failed
            elif (per_rv <= per) and ((np.min(bo_counter) == 0) and (np.size(bo_counter) - np.count_nonzero(bo_counter) == 1)):
                # aoi[retx == 0] += frametxslot
                currentslot += frametxslot
                ind, = np.where(bo_counter == 0)
                retx[ind] = 1
                for x in ind:
                    if bo_stage[x] < 5:
                        bo_stage[x] += 1
                    bo_counter[x] = np.random.randint(contentionwindowsize[bo_stage[x]])
                # print(f"Time: {currentslot}, Tx collision from {ind+1} with AoI {aoi[ind]}")
                df2 = pd.DataFrame({'time': currentslot*slottime, 'node': ind, 'aoi': aoitimestamp[ind]*slottime, 'result': 'fail'})
                df = pd.concat([df, df2], ignore_index=True, axis=0)
                aoitimestamp[retx == 0] = currentslot
                num_fail += 1
            # Tx coll
            elif np.min(bo_counter) == 0 and (np.size(bo_counter) - np.count_nonzero(bo_counter) > 1):
                # aoi[retx == 0] += frametxslot
                currentslot += frametxslot
                ind, = np.where(bo_counter == 0)
                retx[ind] = 1
                for x in ind:
                    if bo_stage[x] < 5:
                        bo_stage[x] += 1
                    bo_counter[x] = np.random.randint(contentionwindowsize[bo_stage[x]])
                # print(f"Time: {currentslot}, Tx collision from {ind+1} with AoI {aoi[ind]}")
                df2 = pd.DataFrame({'time': currentslot*slottime, 'node': ind, 'aoi': aoitimestamp[ind]*slottime, 'result': 'coll'})
                df = pd.concat([df, df2], ignore_index=True, axis=0)
                aoitimestamp[retx == 0] = currentslot
                num_coll += 1

        # print(f"# of succ: {num_succ * frametxslot * slottime}/{beaconinterval}")
        return df
        # df.to_csv('csma.csv', index=False)
    if raalgo == 'slottedaloha':
        slottime = 9
        currentslot = 0
        txprob = 1/numnodes
        # print(f"Data frame length: {txslot}")
        num_succ = 0
        num_coll = 0
        succ_timestamp = np.array([], dtype=int)
        succ_namestamp = np.array([], dtype=int)
        retx = np.zeros(numnodes, dtype=int)
        aoitimestamp = np.zeros(numnodes, dtype=int)
        df = pd.DataFrame(columns=['time', 'node', 'aoi', 'result'])

        while currentslot < beaconinterval/slottime - frametxslot:
            txprobarray = (np.random.rand(numnodes) < txprob).astype(int)
            currentslot += frametxslot
            # Tx idle
            if txprobarray.sum() == 0:
                pass
            # Tx succ
            elif (random.random() > per) and (txprobarray.sum() == 1):
                ind, = np.where(txprobarray == 1)
                # print(f"Time: {currentslot}, Tx success from {ind+1} with AoI {aoi[ind]}")
                df2 = pd.DataFrame({'time': currentslot*slottime, 'node': ind, 'aoi': aoitimestamp[ind]*slottime, 'result': 'succ'})
                df = pd.concat([df, df2], ignore_index=True, axis=0)
                retx[ind] = 0
                aoitimestamp[retx == 0] = currentslot
                succ_timestamp = np.append(succ_timestamp, currentslot)
                succ_namestamp = np.append(succ_namestamp, ind[0])
                num_succ += 1
            # Tx coll
            else:
                ind, = np.where(txprobarray == 1)
                # print(f"Time: {currentslot}, Tx collision from {ind+1} with AoI {aoi[ind]}")
                # df2 = pd.DataFrame({'time': currentslot*slottime, 'node': ind, 'aoi': aoitimestamp[ind]*slottime, 'result': 'coll'})
                # df = pd.concat([df, df2], ignore_index=True, axis=0)
                retx[ind] = 1
                num_coll += 1
            aoitimestamp[retx == 0] = currentslot
            # currentslot += 1
        # print(f"# of succ: {num_succ*txslot*slottime}/{interval}")
        return df
        # df.to_csv('slottedaloha.csv', index=False)