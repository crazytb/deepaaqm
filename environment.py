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

# State settings
velocity = 100   # km/h
snr_ave = 15
snr_thr = snr_ave

f_0 = 5.9e9 # Carrier freq = 5.9GHz, IEEE 802.11bd
speedoflight = 300000   # km/sec
f_d = velocity/(3600*speedoflight)*f_0  # Hz
packettime = 300    # us
fdtp = f_d*packettime/1e6

TRAN_01 = (fdtp*math.sqrt(2*math.pi*snr_thr/snr_ave))/(np.exp(snr_thr/snr_ave)-1)
TRAN_00 = 1 - TRAN_01
TRAN_11 = fdtp*math.sqrt((2*math.pi*snr_thr)/snr_ave)
TRAN_10 = 1 - TRAN_11


def stepfunc(thres, x):
    if x > thres:
        return 1
    else:
        return 0

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
        self.layer3 = nn.Linear(round(n_observations/2), round(n_observations/2))
        self.layer4 = nn.Linear(round(n_observations/2), n_actions)
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)


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
            "inbuffer_nodes": spaces.MultiDiscrete([NUMNODES] * BUFFERSIZE),
            "inbuffer_timestamps": spaces.Box(low=0, high=1, shape=(1, BUFFERSIZE)),
        })
        
        self.rng = default_rng()
        self.current_obs = None
    
    def _get_obs(self):
        return {
            "channel_quality": self.channel,
            "current_aois": self.current_aoi,
            "inbuffer_nodes": self.inbuffer_nodes,
            "inbuffer_timestamps": self.inbuffer_timestamps,
        }
    
    def _fill_first_zero(self, arr, value):
        for i in range(len(arr)):
            if arr[i] == 0:
                arr[i] = value
                break
        return arr

    def _flatten_dict_values(self, dict):
        flattened = np.array([])
        for v in list(dict.values()):
            if isinstance(v, np.ndarray):
                flattened = np.concatenate([flattened, v])
            else:
                flattened = np.concatenate([flattened, np.array([v])])
        return flattened
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.channel = self.rng.integers(0, self.max_channel_quality)
        self.current_aoi = np.zeros(NUMNODES, dtype=float)
        self.inbuffer_nodes = 10 * np.ones(BUFFERSIZE, dtype=int)
        self.inbuffer_timestamps = np.zeros(BUFFERSIZE, dtype=float)
        
        self.leftslots = round(BEACONINTERVAL / TIMEEPOCH)
        self.leftbuffers = BUFFERSIZE
        self.currenttime = 0
            
        self.info = self._get_obs()
        self.current_obs = self._flatten_dict_values(self.info)

        return self.current_obs, self.info


    def probenqueue(self, dflog):
        self.currenttime += TIMEEPOCH
        self.current_aoi += TIMEEPOCH / BEACONINTERVAL
        
        # Define condition that the elements of the dflog can enqueue.
        cond = (dflog.time >= self.currenttime - TIMEEPOCH) & (dflog.time < self.currenttime)
        
        # Extract target dflog
        targetdflog = dflog[cond][:self.leftbuffers]
        tnodenumber = len(targetdflog)
        self.leftbuffers = BUFFERSIZE - tnodenumber

        if tnodenumber == 0:
            pass
        else:
            enquenodeentrytime = targetdflog.time.values.astype(int)
            enquenode = targetdflog.node.values.astype(int)
            enquenodetimestamp = targetdflog.timestamp.values.astype(int)

            self.inbuffernode[self.qpointer:self.qpointer + tnodenumber] = enquenode
            self.inbufferaoi[self.qpointer:self.qpointer + tnodenumber] = \
                ((self.currenttime - enquenodeentrytime / BEACONINTERVAL) + enquenodetimestamp / BEACONINTERVAL)
            self.qpointer += tnodenumber
            self.consumedenergy += 0.154 * TIMEEPOCH

        self.state = np.concatenate([self.current_aoi, self.bufferaoi])

    def step(self, action):  # 여기 해야 함.
        """
        Apply action
            0: FORWARD
                1. bufferinfo sliding
                2. AoIinfo(n) to zero, AoIinfo(-n) += 1
                3. whethertxed(n) to one
                4. Applying Reward(n) (Power consumption, AoI reward)
            1: DISCARD
                1. bufferinfo sliding
                2. AoIinfo(all) += 1
                3. whethertxed no change
            2: SKIP
                1. bufferinfo no change
                2. AoIinfo(all) += 1
                3. whethertxed no change
        :return: self.state, reward, done, info = (np.ndarray, scalar, bool, ...)
        """

        reward = 0
        # 0: FORWARD
        if action == 0:
            if (self.qpointer < 0) or (self.inbuffernode[0] == 0):
                # If the buffer is empty,
                # reward = -1*POWERCOEFF
                # reward = -0.506 * POWERCOEFF - self.current_aoi.max()
                gained_aoi = 0
                pass
            # 버퍼에 들어있지 않은 노드의 AoI를 어떻게 표현할 것인지? --> Inf로 표현.
            else:
                dequenode = self.inbuffernode[0]
                dequenodeaoi = self.inbufferaoi[0]
                # reward = -0.506 * POWERCOEFF + \
                #          1 * (self.current_aoi[dequenode - 1] - (dflog.loc[countindex].time - dflog.loc[countindex].aoi)/BEACONINTERVAL)
                # reward = -0.506 * POWERCOEFF - self.current_aoi.max()
                # Left-shift bufferinfo
                self.inbuffernode[:-1] = self.inbuffernode[1:]
                self.inbuffernode[-1] = 0
                self.inbufferaoi[dequenode - 1] = 0
                self.qpointer = max(0, self.qpointer - 1)

                self.txed[dequenode - 1] = 1

                # reward = (NUMNODES * np.e * TIMEEPOCH * (self.current_aoi[dequenode - 1]*BEACONINTERVAL - dequenodeaoi)) \
                #          / (BEACONINTERVAL**2)
                gained_aoi = self.current_aoi[dequenode - 1] - dequenodeaoi
                if self.channel == [0]:
                    self.current_aoi[dequenode - 1] = dequenodeaoi
                # self.txed[dequenode - 1] = 1
            # reward = -0.506 * POWERCOEFF - AOIPENALTY*max([self.current_aoi.max()-PEAKAOITHRES, 0])
            reward = -0.506 * POWERCOEFF
            # reward = -0.506 * POWERCOEFF + gained_aoi
            self.consumedenergy += 0.352 * TIMEEPOCH  # P.tx = 352mW, P.rx = 154mW.

        # 1: DISCARD
        elif action == 1:
            if self.qpointer == 0:
                # If the buffer is empty,
                # reward = -0.154 * POWERCOEFF - self.current_aoi.max()
                pass
            else:
                dequenode = self.inbuffernode[0]
                dequenodeaoi = self.inbufferaoi[0]

                # Left-shift bufferinfo
                self.inbuffernode[:-1] = self.inbuffernode[1:]
                self.inbuffernode[-1] = 0
                self.inbufferaoi[dequenode - 1] = 0
                self.qpointer -= 1

                # reward = -0.154 * POWERCOEFF - self.current_aoi.max()
                # self.current_aoi[dequenode - 1] = dequenodeaoi / BEACONINTERVAL
            # reward = -0.154 * POWERCOEFF - AOIPENALTY*max([self.current_aoi.max()-PEAKAOITHRES, 0])
            reward = -0.154 * POWERCOEFF
            self.consumedenergy += 0.154 * TIMEEPOCH  # P.rx = 154mW.
        # 2: SKIP
        elif action == 2:
            if self.qpointer == 0:
                # If the buffer is empty,
                # reward = -0.055 * POWERCOEFF - self.current_aoi.max()
                pass
            else:
                # reward = -0.055 * POWERCOEFF - self.current_aoi.max()
                pass
            # reward = -0.055 * POWERCOEFF - AOIPENALTY*max([self.current_aoi.max()-PEAKAOITHRES, 0])
            reward = -0.055 * POWERCOEFF
            self.consumedenergy += 0.055 * TIMEEPOCH  # P.listen = 55mW.

        # self.aoi = np.append(self.aoi, self.current_aoi)
        self.aoi = np.vstack((self.aoi, self.current_aoi))
        # self.state = np.concatenate((buffernodeindexstate, bufferaoistate, txedstate, aoiinfostate))

        info = {}

        if self.channel == [0]:
            if stepfunc(TRAN_00, random.random()) == 0:  # 0 to 0
                self.channel = [0]
            else:  # 0 to 1
                self.channel = [1]
        if self.channel == [1]:
            if stepfunc(TRAN_11, random.random()) == 0:  # 1 to 1
                self.channel = [1]
            else:  # 0 to 1
                self.channel = [0]

        # Return step information
        self.state = np.concatenate([self.channel, self.current_aoi, self.bufferaoi])
        self.qpointerhistory.append(self.qpointer)
        self.previous_action = action
        
        self.leftslots -= 1
        done = self.leftslots <= 0
        
        if self.current_aoi.max() >= (PEAKAOITHRES / BEACONINTERVAL):
            reward -= np.clip(self.current_aoi - (PEAKAOITHRES / BEACONINTERVAL), 0, None).sum()
        
        # if done:
            # reward += 1

        return self.state, reward, done, self.info

    def step_rlaqm(self, action, dflog, countindex, link_utilization):  # 여기 해야 함.
        """
        Apply action
            0: FORWARD
                1. bufferinfo sliding
                2. AoIinfo(n) to zero, AoIinfo(-n) += 1
                3. whethertxed(n) to one
                4. Applying Reward(n) (Power consumption, AoI reward)
            1: DISCARD
                1. bufferinfo sliding
                2. AoIinfo(all) += 1
                3. whethertxed no change
            2: SKIP
                1. bufferinfo no change
                2. AoIinfo(all) += 1
                3. whethertxed no change
        :return: self.state, reward, done, info = (np.ndarray, scalar, bool, ...)
        """

        # 0: FORWARD
        if action == 0:
            if (self.qpointer < 0) or (self.inbuffernode[0] == 0):
                # If the buffer is empty,
                pass
            # 버퍼에 들어있지 않은 노드의 AoI를 어떻게 표현할 것인지? --> Inf로 표현.
            else:
                dequenode = self.inbuffernode[0]
                dequenodeaoi = self.inbufferaoi[0]
                # Left-shift bufferinfo
                self.inbuffernode[:-1] = self.inbuffernode[1:]
                self.inbuffernode[-1] = 0
                self.inbufferaoi[dequenode - 1] = 0
                self.qpointer = max(0, self.qpointer - 1)

                self.txed[dequenode - 1] = 1
                self.current_aoi[dequenode - 1] = dequenodeaoi / BEACONINTERVAL
                # self.txed[dequenode - 1] = 1
            self.consumedenergy += 0.352*TIMEEPOCH  # P.tx = 352mW, P.rx = 154mW.

        # 1: DISCARD
        elif action == 1:
            if self.qpointer == 0:
                # If the buffer is empty,
                pass
            else:
                dequenode = self.inbuffernode[0]
                dequenodeaoi = self.inbufferaoi[0]

                # Left-shift bufferinfo
                self.inbuffernode[:-1] = self.inbuffernode[1:]
                self.inbuffernode[-1] = 0
                self.inbufferaoi[dequenode - 1] = 0
                self.qpointer -= 1

                self.current_aoi[dequenode - 1] = dequenodeaoi / BEACONINTERVAL
            # self.consumedenergy += 0.154*TIMEEPOCH  # P.rx = 154mW.
        # 2: SKIP
        elif action == 2:
            if self.qpointer == 0:
                # If the buffer is empty,
                pass
            else:
                pass
            # self.consumedenergy += 0.055*TIMEEPOCH  # P.listen = 55mW.

        # self.aoi = np.append(self.aoi, self.current_aoi)
        self.aoi = np.vstack((self.aoi, self.current_aoi))
        # self.state = np.concatenate((buffernodeindexstate, bufferaoistate, txedstate, aoiinfostate))

        if self.currenttime > 1:
            done = True
            # if self.aoi.mean(axis=0).max()*BEACONINTERVAL/1000 > 20:
            #     reward -= 10000
            # elif self.txed.sum() < NUMNODES:
            #     # reward -= POWERCOEFF*(NUMNODES-self.txed.sum())
            #     reward -= 10000
            # else:
            #     pass
        else:
            done = False

        self.info = {}

        if self.channel == [0]:
            if stepfunc(TRAN_00, random.random()) == 0:  # 0 to 0
                self.channel = [0]
            else:  # 0 to 1
                self.channel = [1]
        if self.channel == [1]:
            if stepfunc(TRAN_11, random.random()) == 0:  # 1 to 1
                self.channel = [1]
            else:  # 0 to 1
                self.channel = [0]

        # Return step information
        self.state = np.concatenate([self.channel, self.current_aoi, self.bufferaoi])
        reward = (link_utilization**2 - 0.5) + (2/(1+(self.aoi[self.aoi != np.inf].mean()*BEACONINTERVAL/1000)/5) - 1.5)

        self.qpointerhistory.append(self.qpointer)
        self.previous_action = action
        
        done = self.leftslots <= 0
        
        
        return self.state, reward, done, self.info

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

    
