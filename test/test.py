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


Feature_data=[]
test1 = np.array([0,2,3,4,5])
test2 = np.array([0,4,9,16,25])
t1 = [1,2,3]
t2 = [4,5,6]
t3 = [7,8,9]

T = [t1,t2,t3]
print(T)
T = list(itertools.chain.from_iterable([t1,t2,t3]))
print(T)
