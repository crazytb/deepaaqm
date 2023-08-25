import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters
learning_rate = 0.0001
gamma = 1

# Parameters
BUFFERSIZE = 100  # Def. 100
NUMNODES = 10
DIMSTATES = 2 * NUMNODES + 1
TIMEEPOCH = 250  # microseconds
FRAMETXSLOT = 30
BEACONINTERVAL = 100000  # microseconds
# MAXAOI = int(np.ceil(BEACONINTERVAL / TIMEEPOCH))
ACCESSPROB = 1 / NUMNODES
# ACCESSPROB = 1
POWERCOEFF = 1
AOIPENALTY = 100
PER = 0.1
PEAKAOITHRES = 0.05   # That is, 0.05 for 5ms, (5,20)

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


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []
        self.fc1 = nn.Linear(DIMSTATES, NUMNODES + 2)
        self.fc2 = nn.Linear(NUMNODES + 2, NUMNODES + 2)
        self.fc3 = nn.Linear(NUMNODES + 2, 3)
        # self.fc1 = nn.Linear(4, 128)
        # self.fc2 = nn.Linear(128, 2)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=0)
        # x = F.relu(self.fc2(x))
        return x

    def put_data(self, item):
        self.data.append(item)

    def normalize_rewards(self):
        reward_mean = np.mean([x[0] for x in self.data])
        reward_std = np.std([x[0] for x in self.data])
        eps = np.finfo(np.float32).eps.item()
        for i, x in enumerate(self.data):
            self.data[i][0] = (self.data[i][0] - reward_mean) / (reward_std + eps)

    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + gamma * R
            loss = -torch.log(prob) * R
            loss.backward()
        self.optimizer.step()
        self.data = []


class ShowerEnv(Env):
    def __init__(self):
        # Actions we can take FORWARD, DISCARD, and SKIP
        self.action_space = Discrete(3)
        # ResidualSlots, CurrentAoIs, TxCounter, RemainingSlots array
        low_currentaoi = np.array([0] * NUMNODES)
        low_bufferaoi = np.array([0] * NUMNODES)
        low_channel = np.array([0] * NUMNODES)
        # high_currentaoi = np.array([BEACONINTERVAL - 1] * NUMNODES)
        # high_bufferaoi = np.array([BEACONINTERVAL - 1] * NUMNODES)
        high_currentaoi = np.array([1] * NUMNODES)
        high_bufferaoi = np.array([1] * NUMNODES)
        high_channel = np.array([1] * NUMNODES)
        low = np.concatenate([low_channel, low_currentaoi, low_bufferaoi])
        high = np.concatenate([high_channel, high_currentaoi, high_bufferaoi])
        # self.observation_space = Box(low, high, dtype=np.float64)
        self.observation_space = Box(np.float32(low), np.float32(high))
        # self.observation_space = Tuple((Discrete(100), Discrete(100), Discrete(2), Discrete(100)))
        # Set start temp
        # self.state = 38 + random.randint(-3,3)

        # Set state (CurrentAoIinfo)
        self.channel = np.array([0])
        self.currentaoi = np.array([0] * NUMNODES, dtype=np.float32)
        self.bufferaoi = np.array([1] * NUMNODES, dtype=np.float32)
        self.state = np.concatenate([self.channel, self.currentaoi, self.bufferaoi])
        # self.bufferaoi[:] = BEACONINTERVAL - 1

        self.aoi = np.zeros(NUMNODES)
        # self.aoi = np.inf*np.ones(NUMNODES)
        self.txed = np.zeros(NUMNODES)
        self.consumedenergy = 0

        # Set buffer information (Not incorporated in state space.)
        self.inbufferaoi = np.array([np.inf] * BUFFERSIZE)
        self.inbuffernode = np.zeros(BUFFERSIZE, dtype=int)

        # Set shower length
        self.currenttime = 0
        self.qpointer = 0
        self.qpointerhistory = []
        self.holinfo = [None] * 3   # ID, aoi at the time, aoi at current time

        self.previous_action = 2

    def probenqueue(self, dflog):
        self.currenttime += TIMEEPOCH / BEACONINTERVAL
        self.currentaoi += TIMEEPOCH / BEACONINTERVAL
        self.inbufferaoi += TIMEEPOCH / BEACONINTERVAL
        cond = (dflog.time / BEACONINTERVAL >= self.currenttime - TIMEEPOCH / BEACONINTERVAL) & (
                    dflog.time / BEACONINTERVAL < self.currenttime)
        targetdflog = dflog[cond]
        tnodenumber = min([len(targetdflog), BUFFERSIZE - self.qpointer])  # 버퍼의 크기에 따라 들어올 수 있는 패킷 필터링.
        targetdflog = targetdflog[:tnodenumber]  # tnodenumber에 따라 DataFrame slicing

        if len(targetdflog) == 0:
        # if (len(targetdflog) == 0) or (self.previous_action == 0):
            pass
        else:
            enquenodeentrytime = targetdflog.time.values.astype(int)
            enquenode = targetdflog.node.values.astype(int)
            enquenodeaoi = targetdflog.aoi.values.astype(int)

            self.inbuffernode[self.qpointer:self.qpointer + tnodenumber] = enquenode
            self.inbufferaoi[self.qpointer:self.qpointer + tnodenumber] = \
                ((self.currenttime - enquenodeentrytime / BEACONINTERVAL) + enquenodeaoi / BEACONINTERVAL)
            self.qpointer += tnodenumber
            self.consumedenergy += 0.154 * TIMEEPOCH

        self.state = np.concatenate([self.currentaoi, self.bufferaoi])

    def step(self, action, dflog, countindex):  # 여기 해야 함.
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
                # reward = -0.506 * POWERCOEFF - self.currentaoi.max()
                gained_aoi = 0
                pass
            # 버퍼에 들어있지 않은 노드의 AoI를 어떻게 표현할 것인지? --> Inf로 표현.
            else:
                dequenode = self.inbuffernode[0]
                dequenodeaoi = self.inbufferaoi[0]
                # reward = -0.506 * POWERCOEFF + \
                #          1 * (self.currentaoi[dequenode - 1] - (dflog.loc[countindex].time - dflog.loc[countindex].aoi)/BEACONINTERVAL)
                # reward = -0.506 * POWERCOEFF - self.currentaoi.max()
                # Left-shift bufferinfo
                self.inbuffernode[:-1] = self.inbuffernode[1:]
                self.inbuffernode[-1] = 0
                self.inbufferaoi[dequenode - 1] = 0
                self.qpointer = max(0, self.qpointer - 1)

                self.txed[dequenode - 1] = 1

                # reward = (NUMNODES * np.e * TIMEEPOCH * (self.currentaoi[dequenode - 1]*BEACONINTERVAL - dequenodeaoi)) \
                #          / (BEACONINTERVAL**2)
                gained_aoi = self.currentaoi[dequenode - 1] - dequenodeaoi / BEACONINTERVAL
                if self.channel == 0:
                    self.currentaoi[dequenode - 1] = dequenodeaoi / BEACONINTERVAL
                # self.txed[dequenode - 1] = 1
            reward = -0.506 * POWERCOEFF - AOIPENALTY*max([self.currentaoi.max()-PEAKAOITHRES, 0])
            # reward = -0.506 * POWERCOEFF + gained_aoi
            self.consumedenergy += 0.352 * TIMEEPOCH  # P.tx = 352mW, P.rx = 154mW.

        # 1: DISCARD
        elif action == 1:
            if self.qpointer == 0:
                # If the buffer is empty,
                # reward = -0.154 * POWERCOEFF - self.currentaoi.max()
                pass
            else:
                dequenode = self.inbuffernode[0]
                dequenodeaoi = self.inbufferaoi[0]

                # Left-shift bufferinfo
                self.inbuffernode[:-1] = self.inbuffernode[1:]
                self.inbuffernode[-1] = 0
                self.inbufferaoi[dequenode - 1] = 0
                self.qpointer -= 1

                # reward = -0.154 * POWERCOEFF - self.currentaoi.max()
                # self.currentaoi[dequenode - 1] = dequenodeaoi / BEACONINTERVAL
            reward = -0.154 * POWERCOEFF - AOIPENALTY*max([self.currentaoi.max()-PEAKAOITHRES, 0])
            # self.consumedenergy += 0.154*TIMEEPOCH  # P.rx = 154mW.
        # 2: SKIP
        elif action == 2:
            if self.qpointer == 0:
                # If the buffer is empty,
                # reward = -0.055 * POWERCOEFF - self.currentaoi.max()
                pass
            else:
                # reward = -0.055 * POWERCOEFF - self.currentaoi.max()
                pass
            reward = -0.055 * POWERCOEFF - AOIPENALTY*max([self.currentaoi.max()-PEAKAOITHRES, 0])

            # self.consumedenergy += 0.055*TIMEEPOCH  # P.listen = 55mW.

        # self.aoi = np.append(self.aoi, self.currentaoi)
        self.aoi = np.vstack((self.aoi, self.currentaoi))
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
        self.state = np.concatenate([self.channel, self.currentaoi, self.bufferaoi])
        self.qpointerhistory.append(self.qpointer)
        self.previous_action = action

        return self.state, reward, done, info

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
                self.currentaoi[dequenode - 1] = dequenodeaoi / BEACONINTERVAL
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

                self.currentaoi[dequenode - 1] = dequenodeaoi / BEACONINTERVAL
            # self.consumedenergy += 0.154*TIMEEPOCH  # P.rx = 154mW.
        # 2: SKIP
        elif action == 2:
            if self.qpointer == 0:
                # If the buffer is empty,
                pass
            else:
                pass
            # self.consumedenergy += 0.055*TIMEEPOCH  # P.listen = 55mW.

        # self.aoi = np.append(self.aoi, self.currentaoi)
        self.aoi = np.vstack((self.aoi, self.currentaoi))
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
        self.state = np.concatenate([self.channel, self.currentaoi, self.bufferaoi])
        reward = (link_utilization**2 - 0.5) + (2/(1+(self.aoi[self.aoi != np.inf].mean()*BEACONINTERVAL/1000)/5) - 1.5)

        self.qpointerhistory.append(self.qpointer)
        self.previous_action = action
        return self.state, reward, done, info

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

    def reset(self):
        """
        :return: np.ndarray
        """
        self.channel = np.array([0])
        self.currentaoi = np.array([0] * NUMNODES, dtype=np.float32)
        self.bufferaoi = np.array([1] * NUMNODES, dtype=np.float32)
        self.state = np.concatenate([self.channel, self.currentaoi, self.bufferaoi])

        self.aoi = np.zeros(NUMNODES)
        self.txed = np.zeros(NUMNODES)
        self.consumedenergy = 0

        # Set shower length
        self.currenttime = 0
        self.qpointer = 0
        self.qpointerhistory = []

        self.previous_action = 2

        return self.state
