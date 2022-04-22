import pyshark
import os
import numpy as np
import pickle
import sys
import scipy.io
import itertools
import math
import csv
from scapy.all import *


def to_csv_pyshark(filename=""):
    print(filename)

    https = 443
    http  = 80
    data = pyshark.FileCapture(filename+".pcap")
    with open(filename+".csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerows(data)

    out = [["index","size","time","ip"]]


    i=0
    for packet in data:
        #print(i)
        if "TCP" in packet:
            t = []
            t.append(i)
            if(int(packet.tcp.dstport) == https or int(packet.tcp.dstport) == http):
                t.append(int(packet.length))
            elif(int(packet.tcp.srcport) == https or int(packet.tcp.srcport) == http):
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

def to_csv(filename=""):
    print(filename)

    https = 443
    http  = 80
    data = rdpcap(filename+".pcap")
    with open(filename+".csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerows(data)

    out = [["index","size","time","ip"]]


    i=0
    for packet in data:
        #print(i)
        if "TCP" in packet and "IP" in packet:
            #print(packet.wirelen,packet.time,packet["IP"].src)
            t = []
            t.append(i)
            if(int(packet["TCP"].dport) == https or int(packet["TCP"].dport) == http):
                t.append(int(packet.wirelen))
            elif(int(packet["TCP"].sport) == https or int(packet["TCP"].sport) == http):
                t.append(-int(packet.wirelen))
            else:
                continue
            t.append(packet.time)
            t.append(packet["IP"].src)
            out.append(t)
            i+=1

    with open(filename+".csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerows(out)

if __name__ == "__main__":
    
    loc = "wsfodins"
    train_size = 200

    with open("../data/wsfsites",'r') as f:
        sites = f.readlines()
        for site in sites:
            s = site.split()
            if s[0]=="#":
                continue
            

            features=[]
            for i in range(train_size):
                if not os.path.isfile("../data/dataset/origin/"+loc+"/"+s[1]+"/"+str(i)+".pcap"):
                    continue
                #print("../data/train/"+s[1]+"/"+str(i)+".pcap")
                to_csv("../data/dataset/origin/"+loc+"/"+s[1]+"/"+str(i))

