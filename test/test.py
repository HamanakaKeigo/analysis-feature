import pyshark
import os
import numpy as np
import pickle
import sys
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.distributions import norm
from sklearn.neighbors import KernelDensity
import seaborn as sns
from scipy.integrate import cumtrapz
import scipy.io
import math
import itertools

def cmp(a, b):
    return (a > b) - (a < b) 

#times = [1,2,3,4,6,5,7]
sizes = [10,2,5,8,12,20,3]

data = pyshark.FileCapture("../data/train/www.osaka-u.ac.jp/1.pcap")
sizes = list(map(lambda x:x-sizes[0], sizes))
sizes.extend([1,2,3])
print(data[4000].sniff_time-data[0].sniff_time)
print(float(data[4000].sniff_timestamp)-float(data[0].sniff_timestamp))
#for packet in data:
#    if "TCP" in packet: