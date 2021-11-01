import pyshark
import os
import numpy as np
import pickle
import sys
import scipy.io


def save_burst(burst=[]):
    bur = {"g5":0,"g10":0,"g15":0,"0":0,"1":0,"2":0,"3":0,"4":0}
    bur["max"] = max(burst)
    bur["ave"] = sum(burst)/len(burst)
    bur["len"] = len(burst)
    for i in burst:
        if i > 5:
            bur["g5"] +=1
        if i > 10:
            bur["g10"] +=1
        if i > 15:
            bur["g15"] +=1
    for i in range(0,5):
        try:
            bur[str(i)]=burst[i]
        except:
            bur[str(i)]="X"
    #print(bur)
    return bur

def get_features(filename = "",ip=""):

    #packet size (positive means outgoing and, negative, incoming.)
    data = pyshark.FileCapture(filename)
    all_size=0
    packet_num=0
    packet_trace=[]

    https = 443
    http  = 80
    first_time=0
    packet_in=0
    packet_out=0
    max_size=0
    burst=[]
    curburst=0
    stopped=0

    for packet in data:
        if "TCP" in packet:
            if(int(packet.tcp.srcport) == (https or http) or int(packet.tcp.dstport) == (http or https)):
                first_time = packet.sniff_time
                break


    for packet in data:
        if "TCP" in packet:

            #to server
            if(int(packet.tcp.dstport) == https or packet.tcp.dstport == http):
                packet_trace.append([packet.sniff_timestamp , int(packet.tcp.len) + int(packet.tcp.hdr_len)])
                all_size += int(packet.length)
                stopped=0
                curburst+= int(packet.length)
                packet_num+=1
                packet_in+=1
                end_time = packet.sniff_time
                if(int(packet.length)>max_size):
                    max_size=int(packet.length)
            #from server
            elif(int(packet.tcp.srcport) == https or packet.tcp.srcport == http):
                packet_trace.append([packet.sniff_timestamp , -int(packet.tcp.hdr_len) - int(packet.tcp.len)])
                all_size += int(packet.length)
                packet_num+=1
                packet_out+=1
                end_time = packet.sniff_time
                if(int(packet.length)>max_size):
                    max_size=int(packet.length)
                if stopped==0:
                    stopped=1
                elif stopped==1:
                    stopped=0
                    if curburst!=0:
                        burst.append(curburst)
                        curburst=0

    bur = save_burst(burst)
    #print(all_size)
    all_time = (end_time - first_time).total_seconds()
    data.close()
    return(all_size,all_time,packet_num,packet_in,max_size,all_time/packet_out,bur)



if __name__ == "__main__":
    #args = sys.argv 

    train_size=100

    with open("../data/sites",'r') as f:
        sites = f.readlines()
        for site in sites:
            all_size=[]
            all_time=[]
            packet_num=[]
            packet_in=[]
            in_rate=[]
            max_size=[]
            out_ps=[]
            burst=[]

            s = site.split()
            if s[0] == "#":
                continue
            

            for i in range(train_size):
                if not os.path.isfile("../data/train/"+s[1]+"/"+str(i)+".pcap"):
                    break
                size,time,num,p_in,m,o_ps,bur = get_features("../data/train/"+s[1]+"/"+str(i)+".pcap",s[2])
                all_size.append(size)
                all_time.append(time)
                packet_num.append(num)
                in_rate.append(p_in/num)
                max_size.append(m)
                out_ps.append(o_ps)
                burst.append(bur)


                print(str(i)+" times of " + s[1])


            f = open('../data/features/all_size/'+s[1], 'wb')
            pickle.dump(all_size,f)
            f.close()
            f = open('../data/features/all_time/'+s[1], 'wb')
            pickle.dump(all_time,f)
            f.close()
            f = open('../data/features/packet_num/'+s[1], 'wb')
            pickle.dump(packet_num,f)
            f.close()
            f = open('../data/features/in_rate/'+s[1], 'wb')
            pickle.dump(in_rate,f)
            f.close()
            f = open('../data/features/max_size/'+s[1], 'wb')
            pickle.dump(max_size,f)
            f.close()
            f = open('../data/features/out_ps/'+s[1], 'wb')
            pickle.dump(out_ps,f)
            f.close()
            #f = open('../data/features/burst/'+s[1]+".mat", 'wb')
            scipy.io.savemat("../data/features/burst/"+s[1]+".mat", bur)

        
            print("get feature of :" + s[1])

    #print(site_data)