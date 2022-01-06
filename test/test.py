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

def save_Cumul50(Size):
    featureCount=50
    feature=[]
    cum = []
    insize=0
    outsize=0
    incount=0
    outcount=0

    for size in Size:
        size = -size
        if len(cum) == 0:
            cum.append(size)
        else:
            cum.append(cum[-1]+size)


    
    """
    Features = np.interp(np.linspace(0, len(cum), featureCount), cum)
    for el in itertools.islice(Features, None):
        feature.append(el)
    """

    return cum

if __name__ == "__main__":
    https = 443
    http  = 80
    Time=[]
    Size=[]
    IP=[]

    with open("../data/train/icn/Amazon.com/0.csv") as f:
        data = csv.reader(f)

        for packet in data:
            if(packet[0] == "index"):
                continue
            Size.append(int(packet[1]))
            Time.append(float(packet[2]))
            IP.append(packet[3])

    
    y = save_Cumul50(Size)
    x = np.linspace(0, len(y)-1, 50)
    x = np.round(x)
    y2=[]
    for i in x:
        print(i)
        y2.append(y[i])
    
    
    plt.plot(y)
    plt.plot(x,y2)
    plt.show()
    print(x)
    print(len(y))