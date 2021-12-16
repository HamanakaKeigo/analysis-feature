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
import csv

import chromedriver_binary
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.chrome.options import Options
import subprocess
from subprocess import PIPE

filename = "../data/amazon/Amazonjp0/amazonjp000.pcap"
data = pyshark.FileCapture(filename)

with open('../data/plot/diff.csv') as f:
    data = list(csv.reader(f))
    data = np.array(data[1:],dtype=np.float32)
    #print(data)

    arg=np.argsort(data[:,2])
    data = data[arg][::-1]
    print(data)

    with open('../data/plot/sort_diff.csv',"w") as f:
        writer = csv.writer(f)
        writer.writerows(data)
        #print(data)

    pl = data.transpose()[1]
    plt.plot(pl)
    plt.show()