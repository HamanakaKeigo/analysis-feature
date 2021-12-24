import pyshark
import os
import numpy as np
import pickle
import sys
import scipy.io
import itertools
import math
import csv


def to_csv(filename=""):
    print(filename)

    https = 443
    http  = 80
    data = pyshark.FileCapture(filename+".pcap")
    out = [["index","size","time","ip"]]


    i=0
    for packet in data:
        #print(i)
        if "TCP" in packet:
            t = []
            t.append(i)
            if(int(packet.tcp.dstport) == https or packet.tcp.dstport == http):
                t.append(int(packet.length))
            elif(int(packet.tcp.srcport) == https or packet.tcp.srcport == http):
                t.append(-int(packet.length))
            else:
                continue
            t.append(packet.sniff_timestamp)
            t.append(packet.ip.host)
            out.append(t)
            i+=1

    data.close()
    with open(filename+".csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerows(out)

if __name__ == "__main__":
    
    train_size = 100

    with open("../data/sites",'r') as f:
        sites = f.readlines()
        for site in sites:
            s = site.split()
            if s[0]=="#":
                continue
            

            features=[]
            for i in range(train_size):
                if not os.path.isfile("../data/train/lib/"+s[1]+"/"+str(i)+".pcap"):
                    break
                #print("../data/train/"+s[1]+"/"+str(i)+".pcap")
                to_csv("../data/train/lib/"+s[1]+"/"+str(i))

