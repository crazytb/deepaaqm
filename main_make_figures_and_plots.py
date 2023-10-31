import matplotlib.pyplot as plt
import pandas as pd

raalgos = ["slottedaloha", "CSMA"]
numnodes = [5, 10]

for raalgo in raalgos:
    for numnode in numnodes:
        # Read files
        deepaaqm = pd.read_csv('test_log_deepaaqm_' + raalgo + '_' + str(numnode) + '.csv')
        codel = pd.read_csv('test_log_codel_' + raalgo + '_' + str(numnode) + '.csv')
        sred = pd.read_csv('test_log_sred_' + raalgo + '_' + str(numnode) + '.csv')
        deepaaqm.drop(columns=["Unnamed: 0"], inplace=True)
        codel.drop(columns=["Unnamed: 0"], inplace=True)
        sred.drop(columns=["Unnamed: 0"], inplace=True)
        
        # Make 1D array that contains all aois of the dataframe
        aois_deepaaqm = deepaaqm.iloc[:, 7:].values.flatten()
        aois_codel = codel.iloc[:, 7:].values.flatten()
        aois_sred = sred.iloc[:, 7:].values.flatten()
               
