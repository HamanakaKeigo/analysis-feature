import pyshark
import os
import numpy as np
import pickle
import sys


if __name__ == "__main__":
    #args = sys.argv 

    train_size=50
    site_data =[]
    with open("../data/sites",'r') as f:
        sites = f.readlines()
        for site in sites:
            s = site.split()
            if s[0] == "#":
                continue
            site_data.append(s)
    
    for site in site_data:
        for i in range(train_size):
            get_feature("../data/train/"+site[1]+"/"+i+"pcap")

        
        print(site[1]+"is exist")

    #print(site_data)


def get_features(filename = False):

    if not os.path.isfile(filename):
        print("file not found in")
        return

    data = pyshark.FileCapture(filename)
    all_size=0

    if "TCP" in packet:
        if(packet.ip.src == ip):
            packets.append([packet.sniff_timestamp , int(packet.tcp.len) + int(packet.tcp.hdr_len)])
        elif(packet.ip.dst == ip):
            packets.append([packet.sniff_timestamp , -int(packet.tcp.hdr_len) - int(packet.tcp.len)])