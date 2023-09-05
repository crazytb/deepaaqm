import numpy as np
from gymnasium import Env
from gymnasium import spaces
from gymnasium.spaces import Discrete, Box
import math
import random
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque
from numpy.random import default_rng

# Hyperparameters
learning_rate = 0.0001
gamma = 1

# Parameters
BUFFERSIZE = 5  # Def. 10
NUMNODES = 10
DIMSTATES = 2 * NUMNODES + 1
TIMEEPOCH = 300  # microseconds
FRAMETXSLOT = 30
BEACONINTERVAL = 100000  # microseconds
# MAXAOI = int(np.ceil(BEACONINTERVAL / TIMEEPOCH))
ACCESSPROB = 1 / NUMNODES
# ACCESSPROB = 1
POWERCOEFF = 1
AOIPENALTY = 1
PER = 0.1
PEAKAOITHRES = 20000   # That is, 5 000 for 5ms, (5,20)


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory():
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, round(n_observations/2))
        self.layer2 = nn.Linear(round(n_observations/2), round(n_observations/2))
        self.layer3 = nn.Linear(round(n_observations/2), n_actions)
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class ShowerEnv(Env):
    def __init__(self):
        super(ShowerEnv, self).__init__()
        # Actions we can take FORWARD, DISCARD, and SKIP
        self.action_space = Discrete(3)
        self.max_channel_quality = 2
        self.max_aois = 1
        
        self.observation_space = spaces.Dict({
            "channel_quality": spaces.Discrete(self.max_channel_quality),
            "current_aois": spaces.Box(low=0, high=1, shape=(1, NUMNODES)),
            "inbuffer_nodes": spaces.MultiBinary([NUMNODES, BUFFERSIZE]),
            "inbuffer_timestamps": spaces.Box(low=0, high=1, shape=(1, BUFFERSIZE)),
        })
        
        self.rng = default_rng()
        self.current_obs = None

    def _stepfunc(self, thres, x):
        if x > thres:
            return 1
        else:
            return 0
    
    def _get_obs(self):
        return {
            "channel_quality": self.channel_quality,
            "current_aois": self.current_aoi,
            "inbuffer_nodes": self.inbuffer_nodes,
            "inbuffer_timestamps": self.inbuffer_timestamps,
        }

    def _fill_first_zero(self, arr1, arr2):
        if not np.any(arr1 == 0):
            return arr1  # No zeros in arr1, return it as is

        zero_index = np.where(arr1 == 0)[0][0]
        remaining_zeros = np.count_nonzero(arr1 == 0) - zero_index  # Calculate the number of remaining zeros after the first zero
        
        if remaining_zeros >= len(arr2):
            arr1[zero_index:zero_index + len(arr2)] = arr2[:remaining_zeros]
        else:
            arr1[zero_index:zero_index + remaining_zeros] = arr2[:remaining_zeros]

        return arr1

    def _flatten_dict_values(self, dict):
        flattened = np.array([])
        for v in list(dict.values()):
            if isinstance(v, np.ndarray):
                flattened = np.concatenate([flattened, np.squeeze(np.reshape(v, [1, v.size]))])
            else:
                flattened = np.concatenate([flattened, np.array([v])])
        return flattened
    
    def _change_channel_quality(self):
        # State settings
        velocity = 100   # km/h
        snr_thr = 15
        snr_ave = snr_thr + 10
        f_0 = 5.9e9 # Carrier freq = 5.9GHz, IEEE 802.11bd
        speedoflight = 300000   # km/sec
        f_d = velocity/(3600*speedoflight)*f_0  # Hz
        packettime = 300    # us
        fdtp = f_d*packettime/1e6
        # 0: Good, 1: Bad
        TRAN_01 = (fdtp*math.sqrt(2*math.pi*snr_thr/snr_ave))/(np.exp(snr_thr/snr_ave)-1)
        TRAN_00 = 1 - TRAN_01
        # TRAN_11 = fdtp*math.sqrt((2*math.pi*snr_thr)/snr_ave)
        TRAN_10 = fdtp*math.sqrt((2*math.pi*snr_thr)/snr_ave)
        TRAN_11 = 1 - TRAN_10

        if self.channel_quality == 0:  # Bad state
            if self._stepfunc(TRAN_00, random.random()) == 0: # 0 to 0
                channel_quality = 0
            else:   # 0 to 1
                channel_quality = 1
        else:   # Good state
            if self._stepfunc(TRAN_11, random.random()) == 0: # 1 to 1
                channel_quality = 1
            else:   # 1 to 0
                channel_quality = 0
    
        return channel_quality
    
    def _is_buffer_empty(self):
        return self.inbuffer_nodes.sum(axis=0)[0] == 0
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.channel_quality = self.rng.integers(0, self.max_channel_quality)
        self.current_aoi = np.zeros(NUMNODES, dtype=float)
        self.inbuffer_nodes = np.zeros([NUMNODES, BUFFERSIZE], dtype=int)
        self.inbuffer_timestamps = np.zeros(BUFFERSIZE, dtype=float)
        
        self.leftslots = round(BEACONINTERVAL / TIMEEPOCH)
        self.leftbuffers = BUFFERSIZE
        self.currenttime = 0
            
        self.info = self._get_obs()
        self.current_obs = self._flatten_dict_values(self.info)

        return self.current_obs, self.info


    def probenqueue(self, dflog):
        self.currenttime += TIMEEPOCH / BEACONINTERVAL
        self.current_aoi += TIMEEPOCH / BEACONINTERVAL
        
        # Define condition that the elements of the dflog can enqueue.
        cond = (dflog.time >= self.currenttime*BEACONINTERVAL - TIMEEPOCH) & (dflog.time < self.currenttime*BEACONINTERVAL)
        
        # Extract target dflog
        self.leftbuffers = BUFFERSIZE - np.count_nonzero(self.inbuffer_timestamps)
        targetdflog = dflog[cond][:self.leftbuffers]
        tnodenumber = len(targetdflog)
        self.leftbuffers -= tnodenumber

        if tnodenumber == 0:
            pass
        else:
            enquenode = targetdflog.node.values.astype(int)
            enquenodetimestamp = targetdflog.timestamp.values.astype(int)

            repetitions = min((self.inbuffer_nodes.sum(axis=0)==0).sum(), tnodenumber)
            if repetitions != 0:
                insert_index = np.argwhere(self.inbuffer_nodes.sum(axis=1)==0)[0][0]
            
            for i in range(repetitions):
                arr = np.zeros(NUMNODES, dtype=int)
                arr[enquenode[i]] = 1
                self.inbuffer_nodes[:, insert_index] = arr
                self.inbuffer_timestamps[insert_index] = enquenodetimestamp[i] / BEACONINTERVAL
                insert_index += 1
            # self._fill_first_zero(self.inbuffer_nodes, enquenode)
            # self._fill_first_zero(self.inbuffer_timestamps, enquenodetimestamp / BEACONINTERVAL)
            
        self.info = self._get_obs()
        self.current_obs = self._flatten_dict_values(self.info)

    def step(self, action):  # 여기 해야 함.
        reward = 0
        # 0: FORWARD
        if action == 0:
            if self._is_buffer_empty():
                pass
            else:
                dequenode = np.nonzero(self.inbuffer_nodes[:, 0])[0]
                dequenodeaoi = self.currenttime - self.inbuffer_timestamps[0]
                
                if self.channel_quality == 0:
                    self.current_aoi[dequenode] = dequenodeaoi
                
                # Left-shift bufferinfo
                self.inbuffer_nodes[:, :-1] = self.inbuffer_nodes[:, 1:]
                self.inbuffer_nodes[-1] = 0
                self.inbuffer_timestamps[:-1] = self.inbuffer_timestamps[1:]
                self.inbuffer_timestamps[-1] = 0
                self.leftbuffers += 1
                reward -= 0.352

        # 1: DISCARD
        elif action == 1:
            if self._is_buffer_empty():
                pass
            else:
                # Left-shift bufferinfo
                self.inbuffer_nodes[:, :-1] = self.inbuffer_nodes[:, 1:]
                self.inbuffer_nodes[-1] = 0
                self.inbuffer_timestamps[:-1] = self.inbuffer_timestamps[1:]
                self.inbuffer_timestamps[-1] = 0
                self.leftbuffers += 1
                reward -= 0.154

        # 2: SKIP
        elif action == 2:
            pass

        self.channel_quality = self._change_channel_quality()
        self.info = self._get_obs()
        self.current_obs = self._flatten_dict_values(self.info)

        self.leftslots -= 1
        done = self.leftslots <= 0
        
        # if self.current_aoi.max() >= (PEAKAOITHRES / BEACONINTERVAL):
        reward -= np.clip(self.current_aoi - (PEAKAOITHRES / BEACONINTERVAL), 0, None).sum()
        # count the number of nodes whose aoi is less than PEAKAOITHRES / BEACONINTERVAL
        reward += np.count_nonzero(self.current_aoi < (PEAKAOITHRES / BEACONINTERVAL)) * (1/NUMNODES)
        
        # if done:
        #     print("Success!")
        #     if self.current_aoi.max() < (PEAKAOITHRES / BEACONINTERVAL):
        #         print("Congrats!")
        #         reward += 100
            

        return self.current_obs, reward, False, done, self.info

    def step_rlaqm(self, action, dflog, countindex, link_utilization):  # 여기 해야 함.
        ...

    def render(self):
        # Implement viz
        pass

    # def getblockcount(self):
    #     """
    #     :return: scalar
    #     """
    #     return self.blockcount

    def getaoi(self):
        """
        :return: scalar
        """
        return self.aoi

    
