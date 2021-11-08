import pyshark
import os
import numpy as np
import pickle
import sys
import scipy.io


def save_burst(Size):

    feature = []
    burst = []
    stopped=0
    curburst=0

    for size in Size["total"]:
        if size>0:
            stopped=0
            curburst += size
        elif size<0:
            if stopped==0:
                stopped=1
            elif stopped==1:
                stopped=0
                if curburst!=0:
                    burst.append(curburst)
                    curburst=0

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

def save_transpos(Size={}):
    in_count=0
    out_count=0
    in_feature=[]
    out_feature=[]
    feature=[]

    for i in range(len(Size["total"])):
        if out_count>=300 and in_count>=300:
            break

        if (Size["total"][i]>0):
            if(out_count<300):
                out_count+=1
                out_feature.append(i)
        elif(Size["total"][i]<0):
            if(in_count<300):
                in_count+=1
                in_feature.append(i)

    for i in range(in_count,300):
        in_feature.append("X")
    for i in range(out_count,300):
        out_feature.append("X")
    
    feature.extend(out_feature)
    feature.append(np.std(out_feature))
    feature.append(np.mean(out_feature))

    feature.extend(in_feature)
    feature.append(np.std(in_feature))
    feature.append(np.mean(in_feature))

    return feature


def save_intI(Size):
    feature=[]
    in_count=0
    in_pre=0
    in_feature=[]
    out_count=0
    out_pre=0
    out_feature[]

    for i in range(0,len(Size["total"])):
        if in_count>=300 and out_count>=300:
            break
        
        if Size["total"][i]>0:
            if(out_count<300):
                out_count+=1
                out_feature.append(i-out_pre)
                out_pre=i
        elif Size["total"][i]<0:
            if(in_count<300):
                in_count+=1
                in_feature.append(i-in_pre)
                in_pre=i

    for i in range(in_count,300):
        in_feature.append("X")
    for i in range(out_count,300):
        out_feature.append("X")

    feature.append(out_feature)
    feature.append(in_feature)

def save_intII_III(Size):
    feature=[]
    MAX=300
    in_feature=0
    out_feature=0
    in_pre=0
    out_pre=0

    interval_freq_in = [0] * (MAX+1)
    interval_freq_out = [0] * (MAX+1)

    for i in range(0,len(Size["total"])):
        if Size["total"][i]>0:
            interval_freq_out[min([MAX,i-out_pre-1])]
            out_pre=i
        elif Size["total"][i]<0:
            interval_freq_in[min([MAX,i-in_pre-1])]
            in_pre=i
    feature.extend(interval_freq_out)
    feature.extend(interval_freq_in)

    feature.extend(interval_freq_out[0:3])
    feature.append( sum(interval_freq_out[3:6]))
    feature.append( sum(interval_freq_out[6:9]))
    feature.append( sum(interval_freq_out[9:14]))
    feature.extend(interval_freq_out[14:])

    feature.extend(interval_freq_in[0:3])
    feature.append( sum(interval_freq_in[3:6]))
    feature.append( sum(interval_freq_in[6:9]))
    feature.append( sum(interval_freq_in[9:14]))
    feature.extend(interval_freq_in[14:])

def save_dist(Size,Time):
    count = 0
    feature=[]
    tmp=[]

    for i in range(0, min(len(Size["total"]),6000)):
        if Size["total"][i] > 0:
            count+=1
        if (i%30) == 29:
            tmp.append(count)
            count = 0
    
    for i range(len(Size["total"]/30),200)
        tmp.append(0)

    feature.extend(tmp)
    feature.append(np.std(tmp))
    feature.append(np.mean(tmp))
    feature.append(np.median(tmp))
    feature.append(np.max(tmp))

    bucket = [0]*20
    for i in range(0,200):
        ib = i/10
        bucket[ib] = bucket[ib] + tmp[i]
    feature.extend(bucket)
    feature.append(np.sum(bucket))

    return feature

def save_ht(Size):
    feature=[]
    for i in range(0,20):
        feature.append(Size["total"][i]+1500)

    in_count=0
    out_count=0
    for i in range(0,30):
        if i < len(Size["total"]):
            if(Size["total"][i]>0):
                out_count += 1
            elif(Size["total"][i]<0):
                in_count += 1
    feature.append(out_count)
    feature.append(in_count)

    in_count=0
    out_count=0
    for i in range(1,31):
        if i < len(Size["total"]):
            if(Size["total"][-i]>0):
                out_count += 1
            elif(Size["total"][-i]<0):
                in_count += 1
    feature.append(out_count)
    feature.append(in_count)

    return feature


def get_features(filename = "",ip=""):

    #packet size (positive means outgoing and, negative, incoming.)
    data = pyshark.FileCapture(filename)

    https = 443
    http  = 80
    burst=[]
    Count={"total":0,"in":0,"out":0}
    Time={"total":[],"in":[],"out":[]}
    Size={"total":[],"in":[],"out":[]}

    for packet in data:
        if "TCP" in packet:
            #to server
            if(int(packet.tcp.dstport) == https or packet.tcp.dstport == http):

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

    data.close()
    start_time=Time["total"][0]
    Time["total"] = list(map(lambda x:x-start_time, Time["total"]))
    Time["in"] = list(map(lambda x:x-start_time, Time["in"]))
    Time["out"] = list(map(lambda x:x-start_time, Time["out"]))

    
    pktcount = save_PktCount(Count)
    time = save_time(Time)
    ngram=[]
    for n in range(2,7):
        ngram.extend(save_ngram(Size,n))
    trans = save_transpos(Size)
    IntervalI = save_intI(Size)
    IntervalII = save_intII_III(Size)
    dist = save_dist(Size,Time)
    bur = save_burst(Size)
    ht = save_ht(Size)
    PktSec = save_PktSec(Time,Size)
    
    return(pktcount,time,ngram,trans,IntervalI,IntervalII_III,dist,bur)



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