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

with open("../data/amazon/amazonjp0/amazonjp000.csv") as f:
    reader = csv.reader(f)

    Size=[]
    Time=[]
    for packet in reader:
        if(len(packet)<1):
            continue
        Size.append(int(packet[1]))
        Time.append(float(packet[2]))

print(sum(Time))