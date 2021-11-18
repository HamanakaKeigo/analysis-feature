import pyshark
import os
import numpy as np
import pickle
import sys
import scipy.io
import itertools
import math

def save_burst(Size):
    
    feature = []
    burst = []
    stopped=0
    curburst=0

    for size in Size:
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
            feature.append(0)
    
    return feature

def save_PktCount(Size=[]):
    feature=[]
    Count={"total":0,"out":0,"in":0}

    for i in range(len(Size)):
        Count["total"] += 1
        if (Size[i]>0):
            Count["out"] += 1
        elif (Size[i]<0):
            Count["in"] += 1

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

def save_time(Time=[],Size=[]):

    feature=[]
    
    
    #ty=["total","out","in"]
    #for t in ty:
    int_all=[]
    int_in=[]
    int_out=[]

    pre_all=-1
    pre_in=-1
    pre_out=-1

    time_in=[]
    time_out=[]
    
    for i in range(len(Time)):
        if(pre_all!=-1):
            int_all.append(Time[i]-pre_all)
        pre_all=Time[i]

        if(Size[i]>0):
            time_out.append(Time[i])
            if(pre_out!=-1):
                int_out.append(Time[i]-pre_out)
            pre_out=Time[i]
        elif(Size[i]<0):
            time_in.append(Time[i])
            if(pre_in!=-1):
                int_in.append(Time[i]-pre_in)
            pre_in=Time[i]

    feature.extend([np.max(int_all),np.mean(int_all),np.std(int_all),np.percentile(int_all,75)])
    feature.extend([np.max(int_out),np.mean(int_out),np.std(int_out),np.percentile(int_out,75)])
    feature.extend([np.max(int_in),np.mean(int_in),np.std(int_in),np.percentile(int_in,75)])
 

    feature.extend([np.percentile(Time,25),np.percentile(Time,50),np.percentile(Time,75),np.percentile(Time,100)])
    feature.extend([np.percentile(time_out,25),np.percentile(time_out,50),np.percentile(time_out,75),np.percentile(time_out,100)])
    feature.extend([np.percentile(time_in,25),np.percentile(time_in,50),np.percentile(time_in,75),np.percentile(time_in,100)])

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

def save_ngram(Size=[],n=0):
    buckets = [0]*(2**n)
    for i in range(0,len(Size)-n+1):
        index = NgramLoc(Size[i:i+n],n)
        buckets[index] += 1

    return buckets

def save_transpos(Size=[]):
    in_count=0
    out_count=0
    in_feature=[]
    out_feature=[]
    feature=[]

    for i in range(len(Size)):
        if out_count>=300 and in_count>=300:
            break

        if (Size[i]>0):
            if(out_count<300):
                out_count+=1
                out_feature.append(i)
        elif(Size[i]<0):
            if(in_count<300):
                in_count+=1
                in_feature.append(i)

    for i in range(in_count,300):
        in_feature.append(0)
    for i in range(out_count,300):
        out_feature.append(0)
    
    feature.extend(out_feature)
    feature.append(np.std(out_feature))
    feature.append(np.mean(out_feature))

    #print(in_feature)
    feature.extend(in_feature)
    feature.append(np.std(in_feature))
    feature.append(np.mean(in_feature))

    return feature


def save_intI(Size=[]):
    feature=[]
    in_count=0
    in_pre=0
    in_feature=[]
    out_count=0
    out_pre=0
    out_feature=[]

    for i in range(0,len(Size)):
        if in_count>=300 and out_count>=300:
            break
        
        if Size[i]>0:
            if(out_count<300):
                out_count+=1
                out_feature.append(i-out_pre)
                out_pre=i
        elif Size[i]<0:
            if(in_count<300):
                in_count+=1
                in_feature.append(i-in_pre)
                in_pre=i

    for i in range(in_count,300):
        in_feature.append(0)
    for i in range(out_count,300):
        out_feature.append(0)

    feature.extend(out_feature)
    feature.extend(in_feature)
    return feature

def save_intII_III(Size=[]):
    feature=[]
    MAX=300
    in_feature=0
    out_feature=0
    in_pre=0
    out_pre=0

    interval_freq_in = [0] * (MAX+1)
    interval_freq_out = [0] * (MAX+1)

    for i in range(0,len(Size)):
        if Size[i]>0:
            interval_freq_out[min([MAX,i-out_pre-1])] += 1
            out_pre=i
        elif Size[i]<0:
            interval_freq_in[min([MAX,i-in_pre-1])] +=1
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

    return feature

def save_dist(Size=[]):
    count = 0
    feature=[]
    tmp=[]

    for i in range(0, min(len(Size),6000)):
        if Size[i] > 0:
            count+=1
        if (i%30) == 29:
            tmp.append(count)
            count = 0
    
    for i in range(int(len(Size)/30),200):
        tmp.append(0)

    feature.extend(tmp)
    feature.append(np.std(tmp))
    feature.append(np.mean(tmp))
    feature.append(np.median(tmp))
    feature.append(np.max(tmp))

    bucket = [0]*20
    for i in range(0,200):
        ib = int(i/10)
        bucket[ib] = bucket[ib] + tmp[i]
    feature.extend(bucket)
    feature.append(np.sum(bucket))

    return feature

def save_ht(Size):
    feature=[]
    for i in range(0,20):
        feature.append(Size[i]+1500)

    in_count=0
    out_count=0
    for i in range(0,30):
        if i < len(Size):
            if(Size[i]>0):
                out_count += 1
            elif(Size[i]<0):
                in_count += 1
    feature.append(out_count)
    feature.append(in_count)

    in_count=0
    out_count=0
    for i in range(1,31):
        if i < len(Size):
            if(Size[-i]>0):
                out_count += 1
            elif(Size[-i]<0):
                in_count += 1
    feature.append(out_count)
    feature.append(in_count)

    return feature

def save_PktSec(Time=[],Size=[]):
    feature=[]
    count = [0]*100

    for i in range(0,len(Size)):
        t = int( np.floor(Time[i]))
        if t < 100:
            count[t] += 1
    feature.extend(count)

    feature.append( np.mean(count))
    feature.append( np.std(count))
    feature.append( np.min(count))
    feature.append( np.max(count))
    feature.append( np.median(count))
    #print(count)
    bucket = [0] * 20
    for i in range(0,100):
        ib = int(i/5)
        bucket[ib] += count[i]
    feature.extend(bucket)
    feature.append(np.sum(bucket))
    
    return feature

def save_cumul(Size):
    feature=[]
    total=[]
    cum = []
    pos=[]
    neg=[]
    insize=0
    outsize=0
    incount=0
    outcount=0

    for size in Size:
        size = -size

        if size>0:
            insize += size
            incount += 1

            if len(cum) == 0:
                cum.append(size)
                total.append(size)
                pos.append(size)
                neg.append(0)
            else:
                cum.append(cum[-1]+size)
                total.append(total[-1]+abs(size))
                pos.append(pos[-1]+size)
                neg.append(neg[-1]+0)

        elif size<0:
            outsize += abs(size)
            outcount += 1

            if len(cum) == 0:
                cum.append(size)
                total.append(size)
                pos.append(0)
                neg.append(size)
            else:
                cum.append(cum[-1]+size)
                total.append(total[-1]+abs(size))
                pos.append(pos[-1]+0)
                neg.append(neg[-1]+size)

    feature.append(incount)
    feature.append(outcount)
    feature.append(outsize)
    feature.append(insize)
    featureCount=100

    posFeatures = np.interp(np.linspace(total[0], total[-1], int(featureCount/2)), total, pos)
    negFeatures = np.interp(np.linspace(total[0], total[-1], int(featureCount/2)), total, neg)
    for el in itertools.islice(posFeatures, None):
        feature.append(el)
    for el in itertools.islice(negFeatures, None):
        feature.append(el)

    return feature

def cmp(a, b):
    return (a > b) - (a < b) 

def normalize_data(Time=[],Size=[]):
    tmp = sorted(zip(Time,Size))

    Time = [x for x,_ in tmp]
    Size = [x for _,x in tmp]

    start_time=Time[0]
    PktSize=500

    for i in range(len(Time)):
        Time[i] = Time[i] - start_time

    for i in range(len(Size)):
        Size[i] = (abs(Size[i])/PktSize) * cmp(Size[i],0)

    new_Time=[]
    new_Size=[]

    for t,s in zip(Time,Size):
        numcell = abs(s)
        onecell = cmp(s,0)

        for r in range(numcell):
            new_Time.append(t)
            new_Size.append(onecell)

    return new_Time,new_Size

def get_features(filename = "",ip=""):

    #packet size (positive means outgoing and, negative, incoming.)
    data = pyshark.FileCapture(filename)

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


    #Time normarize
    start_time=Time[0]
    Time = list(map(lambda x:x-start_time, Time))

     
    #normalize 
    #Time,Size = normalize_data(Time,Size)
    
    pktcount = save_PktCount(Size)
    time = save_time(Time,Size)
    ngram=[]
    for n in range(2,7):
        ngram.extend(save_ngram(Size,n))
    trans = save_transpos(Size)
    IntervalI = save_intI(Size)
    IntervalII = save_intII_III(Size)
    dist = save_dist(Size)
    bur = save_burst(Size)
    ht = save_ht(Size)
    PktSec = save_PktSec(Time,Size)
    cumul = save_cumul(Size)

    feature = [pktcount,time,ngram,trans,IntervalI,IntervalII,dist,bur,ht,PktSec,cumul]
    """
    x=0
    for f in feature:
        x += len(f)
        print(str(x) + " : (" +str(len(f)) + ")")
    """
    return(feature)



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

                feature=[]
                for g in get:
                    feature.extend(g)
                features.append(feature)
                print(len(feature))

                print(str(i)+" times of " + s[1])

            f = open('../data/features/total/'+s[1], 'wb')
            pickle.dump(features,f)
            f.close()
        
            print("get feature of :" + s[1])

    #print(site_data)