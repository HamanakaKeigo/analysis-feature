import pyshark
import os
import numpy as np
import pickle
import sys
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.distributions import norm
from scipy.integrate import cumtrapz
import scipy.io
import math
import itertools
import csv


https = 443
http  = 80

for i in range(10):
    filename = "../data/train/www.amazon.co.jp/"+str(i)
    data = pyshark.FileCapture(filename+".pcap")

    with open(filename+".csv","w") as f:
        writer = csv.writer(f)
        #header = ["index","time","size"]
        #writer.writerow(header)

        rows = []
        for j,packet in enumerate(data):
            row = [j]
            if "TCP" in packet:
                #to server
                if(int(packet.tcp.dstport) == https or packet.tcp.dstport == http):
                    row.append(float(packet.sniff_timestamp))
                    row.append(int(packet.length))
                    rows.append(row)

                #from server
                elif(int(packet.tcp.srcport) == https or packet.tcp.srcport == http):
                    row.append(float(packet.sniff_timestamp))
                    row.append(-int(packet.length))
                    rows.append(row)
                

        #writer.writerows(rows)

        file = open(filename+".pickle", 'wb')
        pickle.dump(rows,file)
        file.close()
        data.close()


    