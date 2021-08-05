import pyshark
import os
import numpy as np
import pickle
import sys





def get_features(filename = "",ip=""):

    data = pyshark.FileCapture(filename)
    all_size=0
    packet_trace=[]

    https = 443
    http  = 80
    for packet in data:
        if "TCP" in packet:

            if(int(packet.tcp.srcport) == https or packet.tcp.srcport == http):
                packet_trace.append([packet.sniff_timestamp , int(packet.tcp.len) + int(packet.tcp.hdr_len)])
                all_size += int(packet.length)
            elif(int(packet.tcp.dstport) == https or packet.tcp.dstport == http):
                packet_trace.append([packet.sniff_timestamp , -int(packet.tcp.hdr_len) - int(packet.tcp.len)])
                all_size += int(packet.length)
        #close(packet)
    print(all_size)
    data.close()
    return(all_size)


if __name__ == "__main__":
    #args = sys.argv 

    train_size=100

    with open("../data/sites",'r') as f:
        sites = f.readlines()
        for site in sites:
            all_size=[]
            s = site.split()
            if s[0] == "#":
                continue

            for i in range(train_size):
                if not os.path.isfile("../data/train/"+s[1]+"/"+str(i)+".pcap"):
                    break
                size = get_features("../data/train/"+s[1]+"/"+str(i)+".pcap",s[2])
                all_size.append(size)

            f = open('../data/features/all_size/'+s[1], 'wb')
            pickle.dump(all_size,f)
            f.close()
        
            print("get feature of :" + s[1])

    #print(site_data)