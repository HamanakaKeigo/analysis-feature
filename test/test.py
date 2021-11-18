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



data = pyshark.FileCapture("../data/train/www.osaka-u.ac.jp/0.pcap")

https = 443
http  = 80

Time=[]
Size=[]

for packet in data:
    if "TCP" in packet:
        #to server
        if(int(packet.tcp.dstport) == https or packet.tcp.dstport == http):
            Time.append(float(packet.sniff_timestamp))
            Size.append(int(packet.length))

        #from server
        elif(int(packet.tcp.srcport) == https or packet.tcp.srcport == http):
            Time.append(float(packet.sniff_timestamp))
            Size.append(-int(packet.length))
data.close()

for i in range(len(Size)):
    Size[i] = ( abs(Size[i])//500 )*cmp(Size[i],0)

sizes=[]
for i in range(len(Size)):
    for j in range(abs(Size[i])):
        sizes.append(i)
print(Size)
print(sizes)