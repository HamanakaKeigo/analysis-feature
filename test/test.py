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

def cmp(a, b):
    return (a > b) - (a < b) 

times = [1,2,3,4,6,5,7]
sizes = [10,2,5,8,12,20,3]

tmp = sorted(zip(times,sizes))
        
times = [x for x,_ in tmp]
sizes = [x for _,x in tmp]
PktSize=5
for i in range(len(sizes)):
    sizes[i] = ( abs(sizes[i])/PktSize )*cmp(sizes[i],0)

for i in range(len(sizes)):
    for x in range(abs(i)):
        print(cmp(sizes[i],0))