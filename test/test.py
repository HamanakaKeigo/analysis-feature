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


with open("../data/features/all_size/www.amazon.co.jp","rb") as f3:
    a = pickle.load(f3)
    a = np.array(a)
    a = np.reshape(a,(-1,1))
    print(len(a))
    #print(Feature_data[feature])