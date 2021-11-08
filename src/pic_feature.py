import pyshark
import os
import numpy as np
import pickle
import sys
import scipy.io


def save_burst(burst=[]):
    feature = []

    feature.append(max(burst))
    feature.append(sum(burst)/len(burst))
    feature.append(len(burst))

    x=0
    y=0
    z=0
    for i in burst:
        if i > 5:
            x +=1
        if i > 10:
            y +=1
        if i > 15:
            z +=1
    feature.append(x)
    feature.append(y)
    feature.append(z)

    for i in range(0,5):
        try:
            feature.append(burst[i])
        except:
            feature.append("X")
    
    return feature

def save_PktCount(Count={}):
    feature=[]

    feature.append(Count["total"])
    feature.append(Count["out"])
    feature.append(Count["in"])
    
    out_t = float(Count["out"])/Count["total"]
    in_t = float(Count["in"])/Count["total"]

    feature.append(out_t*100)
    feature.append(in_t*100)

    feature.append(int(15*round(float(Count["total"])/15)))
    feature.append(int(15*round(float(Count["out"])/15)))
    feature.append(int(15*round(float(Count["in"])/15)))
    
    feature.append(int(5*round(float(out_t*100)/5)))
    feature.append(int(5*round(float(in_t*100)/5)))

    feature.append(Count["total"]*512)
    feature.append(Count["out"]*512)
    feature.append(Count["in"]*512)
    

    return feature

def save_time(Time={}):

    feature=[]
    pre=0
    
    ty=["total","out","in"]
    for t in ty:
        interval=[]
        for time in Time[t]:
            if(pre!=0):
                interval.append(time-pre)
            pre=time

        feature.extend([np.max(interval),np.mean(interval),np.std(interval),np.percentile(interval,75)])
        
    for t in ty:
        feature.extend([np.percentile(Time[t],25),np.percentile(Time[t],50),np.percentile(Time[t],75),np.percentile(Time[t],100)])

    return feature

def NgramLoc(sample,n):
    index=0
    bit=0
    for i in range(0,n):
        if sample[i]>0:
            bit = 1
        else:
            bit = 0
        index = index + bit*( 2**(n-i-1) )

    return index

def save_ngram(Size={},n=0):
    buckets = [0]*(2**n)
    for i in range(0,len(Size["total"])-n+1):
        index = NgramLoc(Size["total"][i:i+n],n)
        buckets[index] += 1

    return buckets

def get_features(filename = "",ip=""):

    #packet size (positive means outgoing and, negative, incoming.)
    data = pyshark.FileCapture(filename)

    https = 443
    http  = 80
    start_time=0
    burst=[]
    Count={"total":0,"in":0,"out":0}
    Time={"total":[],"in":[],"out":[]}
    Size={"total":[],"in":[],"out":[]}
    curburst=0
    stopped=0

    for packet in data:
        if "TCP" in packet:

            #to server
            if(int(packet.tcp.dstport) == https or packet.tcp.dstport == http):


                stopped=0
                curburst+= int(packet.length)
                Count["total"] += 1
                Count["out"] += 1
                Time["total"].append(float(packet.sniff_timestamp))
                Time["out"].append(float(packet.sniff_timestamp))
                Size["total"].append(int(packet.length))
                Size["out"].append(int(packet.length))
                
            #from server
            elif(int(packet.tcp.srcport) == https or packet.tcp.srcport == http):


                Count["total"] += 1
                Count["in"] += 1
                Time["total"].append(float(packet.sniff_timestamp))
                Time["in"].append(float(packet.sniff_timestamp))
                Size["total"].append(int(packet.length))
                Size["in"].append(-int(packet.length))

                if stopped==0:
                    stopped=1
                elif stopped==1:
                    stopped=0
                    if curburst!=0:
                        burst.append(curburst)
                        curburst=0
    data.close()
    start_time=Time["total"][0]
    Time["total"] = list(map(lambda x:x-start_time, Time["total"]))
    Time["in"] = list(map(lambda x:x-start_time, Time["in"]))
    Time["out"] = list(map(lambda x:x-start_time, Time["out"]))

    bur = save_burst(burst)
    pktcount = save_PktCount(Count)
    time_f = save_time(Time)
    ngram=[]
    for n in range(2,7):
        ngram.extend(save_ngram(Size,n))
    
    return(bur,pktcount,time_f,ngram)



if __name__ == "__main__":
    #args = sys.argv 

    train_size=100

    with open("../data/sites",'r') as f:
        sites = f.readlines()
        for site in sites:

            s = site.split()
            if s[0] == "#":
                continue
            
            features=[]
            for i in range(train_size):
                if not os.path.isfile("../data/train/"+s[1]+"/"+str(i)+".pcap"):
                    break
                get = get_features("../data/train/"+s[1]+"/"+str(i)+".pcap",s[2])
                features.append(get)

                print(str(i)+" times of " + s[1])

            f = open('../data/features/total/'+s[1], 'wb')
            pickle.dump(features,f)
            f.close()
        
            print("get feature of :" + s[1])

    #print(site_data)