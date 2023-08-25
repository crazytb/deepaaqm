import math

numnodes = 10
L_data = 200     # Bytes
datarate = 8.1  # Mbit/s
N_DBPS = 117
T_sym = 16
# L_RTS = 20;
# L_CTS = 14;
# rtscts = 0;
# N_RTS = (22 + (L_RTS*8))/N_DBPS_legacy
# T_RTS = 40 + 4*ceil(T_sym*N_RTS/4) # in us
# N_CTS = (22 + (L_CTS*8))/N_DBPS_legacy
# T_CTS = 40 + 4*ceil(T_sym*N_CTS/4) # in us
N_data = (22 + (L_data*8))/N_DBPS
T_data = 40 + 4*math.ceil(T_sym*N_data/4) # in us
# N_ACK = (22 + (L_ACK*8))/N_DBPS_control
# T_ACK = 40 + 4*ceil(T_sym*N_ACK/4) # in us

slottime = 9    # in microsec
binterval = 100000    # in microsec
txslot = math.ceil(T_data/slottime)
