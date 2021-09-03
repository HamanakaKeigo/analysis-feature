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
    first_time=0
    for packet in data:
        if "TCP" in packet:
            if(int(packet.tcp.srcport) == (https or http) or int(packet.tcp.dstport) == (http or https)):
                first_time = packet.sniff_time
                break


    for packet in data:
        if "TCP" in packet:

            if(int(packet.tcp.srcport) == https or packet.tcp.srcport == http):
                packet_trace.append([packet.sniff_timestamp , int(packet.tcp.len) + int(packet.tcp.hdr_len)])
                all_size += int(packet.length)
                end_time = packet.sniff_time
            elif(int(packet.tcp.dstport) == https or packet.tcp.dstport == http):
                packet_trace.append([packet.sniff_timestamp , -int(packet.tcp.hdr_len) - int(packet.tcp.len)])
                all_size += int(packet.length)
                end_time = packet.sniff_time

        #close(packet)
    print(all_size)
    all_time = (end_time - first_time).total_seconds()
    data.close()
    return(all_size,all_time)


if __name__ == "__main__":
    #args = sys.argv 

    train_size=100

    with open("../data/sites",'r') as f:
        sites = f.readlines()
        for site in sites:
            all_size=[]
            all_time=[]
            s = site.split()
            if s[0] == "#":
                continue
            

            for i in range(train_size):
                if not os.path.isfile("../data/train/"+s[1]+"/"+str(i)+".pcap"):
                    break
                size,time = get_features("../data/train/"+s[1]+"/"+str(i)+".pcap",s[2])
                all_size.append(size)
                all_time.append(time)

            f = open('../data/features/all_size/'+s[1], 'wb')
            pickle.dump(all_size,f)
            f.close()
            f = open('../data/features/all_time/'+s[1], 'wb')
            pickle.dump(all_time,f)
            f.close()
        
            print("get feature of :" + s[1])

    #print(site_data)