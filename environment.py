import numpy as np
from gym import Env
from gym.spaces import Discrete, Box

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters
learning_rate = 0.0005
gamma = 1

# Parameters
BUFFERSIZE = 3
NUMNODES = 5
DIMSTATES = 2 * BUFFERSIZE + 2 * NUMNODES
TIMEEPOCH = 300  # microseconds
FRAMETXSLOT = 30
BEACONINTERVAL = 100000  # microseconds
MAXAOI = int(np.ceil(BEACONINTERVAL / TIMEEPOCH))
ACCESSPROB = 1 / NUMNODES
# ACCESSPROB = 1
POWERCOEFF = 1
AOICOEFF = 0


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []
        self.fc1 = nn.Linear(DIMSTATES, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 3)
        # self.fc1 = nn.Linear(4, 128)
        # self.fc2 = nn.Linear(128, 2)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=0)
        return x

    def put_data(self, item):
        self.data.append(item)

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
        low_bufferinfo = np.array([0] * BUFFERSIZE)
        low_aoiinfo = np.array([0] * NUMNODES)
        low_whethertxed = np.array([0] * NUMNODES)
        low_currentaoiinfo = np.array([0] * NUMNODES)
        high_bufferinfo = np.array([NUMNODES - 1] * BUFFERSIZE)
        high_aoiinfo = np.array([BEACONINTERVAL] * NUMNODES)
        high_whethertxed = np.array([1] * NUMNODES)
        high_currentaoiinfo = np.array([BEACONINTERVAL] * NUMNODES)
        low = np.concatenate((low_bufferinfo, low_aoiinfo, low_whethertxed, low_currentaoiinfo))
        high = np.concatenate((high_bufferinfo, high_aoiinfo, high_whethertxed, high_currentaoiinfo))
        self.observation_space = Box(low, high, dtype=int)
        # self.observation_space = Tuple((Discrete(100), Discrete(100), Discrete(2), Discrete(100)))
        # Set start temp
        # self.state = 38 + random.randint(-3,3)

        # Set state (Bufferinfo, AoIinfo, Whethertxedinfo, CurrentAoIinfo)
        self.state = np.array([0] * DIMSTATES)
        self.buffernodeindexstate = self.state[:BUFFERSIZE:]
        self.bufferaoistate = self.state[BUFFERSIZE:2 * BUFFERSIZE:]
        self.txedstate = self.state[2 * BUFFERSIZE:2 * BUFFERSIZE + NUMNODES:]
        self.aoiinfostate = self.state[2 * BUFFERSIZE + NUMNODES::]
        # Set shower length
        self.restime = 0
        self.qpointer = 0
        self.aoi = np.zeros([0, NUMNODES], int)
        self.blockcount = 0

    def probenqueue(self, dflog):
        self.restime += TIMEEPOCH
        self.aoiinfostate += TIMEEPOCH
        cond = (dflog.time >= self.restime - TIMEEPOCH) & (dflog.time < self.restime)
        targetdflog = dflog[cond]
        # if (len(targetdflog) == 1) and (self.qpointer < BUFFERSIZE):
        if len(targetdflog) == 0:
            pass
        elif len(targetdflog) + self.qpointer <= BUFFERSIZE:
            tnode = targetdflog.node.values.tolist()
            tnodeaoi = (self.restime-targetdflog.time.values).tolist()
            self.buffernodeindexstate[self.qpointer:self.qpointer + len(tnode):] = tnode
            self.bufferaoistate[self.qpointer:self.qpointer + len(tnode):] = tnodeaoi
            self.qpointer += len(tnode)
        elif len(targetdflog) + self.qpointer > BUFFERSIZE:
            tnode = targetdflog.node.values.tolist()
            tnodeaoi = (self.restime-targetdflog.time.values).tolist()
            self.buffernodeindexstate[self.qpointer:self.qpointer + len(tnode):] \
                = tnode[:len(self.buffernodeindexstate) - self.qpointer:]
            self.bufferaoistate[self.qpointer:self.qpointer + len(tnode):] \
                = tnodeaoi[:len(self.buffernodeindexstate) - self.qpointer:]
            self.blockcount += (len(targetdflog) + self.qpointer - BUFFERSIZE)
            self.qpointer = BUFFERSIZE
        self.state = np.concatenate((self.buffernodeindexstate, self.bufferaoistate, self.txedstate, self.aoiinfostate))

    def step(self, action):
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
        # self.restime += TIMEEPOCH
        pointer = self.qpointer
        # buffernodeindexstate = self.state[:BUFFERSIZE:]
        # bufferaoistate = self.state[BUFFERSIZE:2 * BUFFERSIZE:]
        # txedstate = self.state[2 * BUFFERSIZE:2 * BUFFERSIZE + NUMNODES:]
        # aoiinfostate = self.state[2 * BUFFERSIZE + NUMNODES::]
        # self.bufferaoistate[self.bufferaoistate != 0] += TIMEEPOCH
        # self.aoiinfostate += TIMEEPOCH
        reward = 0

        # 0: FORWARD
        if action == 0:
            if self.qpointer == 0:
                # If the buffer is empty,
                # reward = 0
                pass
            else:
                targetnode = self.buffernodeindexstate[0]
                self.buffernodeindexstate = np.roll(self.buffernodeindexstate, -1)
                self.buffernodeindexstate[-1] = 0
                # self.aoiinfostate[targetnode - 1] = \
                #     min([self.aoiinfostate[targetnode - 1], (self.restime - self.bufferaoistate[0])])
                self.aoiinfostate[targetnode - 1] = self.bufferaoistate[0]
                self.bufferaoistate = np.roll(self.bufferaoistate, -1)
                self.bufferaoistate[-1] = 0
                self.txedstate[targetnode - 1] = 1
                # reward = (-1)*POWERCOEFF
                self.qpointer -= 1
        # 1: DISCARD
        elif action == 1:
            if self.qpointer == 0:
                # If the buffer is empty,
                # reward = 0
                pass
            else:
                self.buffernodeindexstate = np.roll(self.buffernodeindexstate, -1)
                self.buffernodeindexstate[-1] = 0
                # reward = 0
                self.qpointer -= 1
        # 2: SKIP
        elif action == 2:
            if self.qpointer == 0:
                # If the buffer is empty,
                # reward = 0
                pass
            else:
                # reward = 0
                pass

        self.aoi = np.append(self.aoi, [self.aoiinfostate], axis=0)
        # self.state = np.concatenate((buffernodeindexstate, bufferaoistate, txedstate, aoiinfostate))

        if self.restime > BEACONINTERVAL:
            done = True
            if (self.txedstate == np.ones(NUMNODES)).all():
                reward = BEACONINTERVAL - self.aoi.mean()
            else:
                pass
                # reward -= BEACONINTERVAL
        else:
            done = False

        info = {}

        # Return step information
        self.state = np.concatenate((self.buffernodeindexstate, self.bufferaoistate, self.txedstate, self.aoiinfostate))
        return self.state, reward, done, info

    def render(self):
        # Implement viz
        pass

    def getblockcount(self):
        """
        :return: scalar
        """
        return self.blockcount

    def getaoi(self):
        """
        :return: scalar
        """
        return self.aoi

    def reset(self):
        """
        :return: np.ndarray
        """
        # Set state (Bufferinfo, AoIinfo, Whethertxedinfo, CurrentAoIinfo)
        self.state = np.array([0] * DIMSTATES)
        self.buffernodeindexstate = self.state[:BUFFERSIZE:]
        self.bufferaoistate = self.state[BUFFERSIZE:2 * BUFFERSIZE:]
        self.txedstate = self.state[2 * BUFFERSIZE:2 * BUFFERSIZE + NUMNODES:]
        self.aoiinfostate = self.state[2 * BUFFERSIZE + NUMNODES::]
        # Set shower length
        self.restime = 0
        self.qpointer = 0
        self.aoi = np.zeros([0, NUMNODES], int)
        self.blockcount = 0
        return self.state