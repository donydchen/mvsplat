import os
import numpy as np

if os.path.exists('datasets/Replica'):
    replica_instance = np.loadtxt('configs/replica_instance_split.txt',dtype=str).tolist()
    replica_train = np.loadtxt('configs/replica_train_split.txt',dtype=str).tolist()
    replica_test = np.loadtxt('configs/replica_test_split.txt',dtype=str).tolist()
    