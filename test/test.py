import pyshark
import os
import numpy as np
import pickle
import sys

train_size=100
all_time=[]
with open("../data/sites",'r') as f:
    sites = f.readlines()
    for site in sites:
        s = site.split()
        if s[0] == "#":
            continue
        f = open("../data/features/all_time/"+s[1],"rb")
        all_size = pickle.load(f)
        all_size_v = []
        print(all_size)
        """
        for size in all_size:
            all_size_v.append(-size) 
        f.close()

        f = open("../data/features/all_time/"+s[1], 'wb')
        pickle.dump(all_size_v,f)
        f.close()
        """
