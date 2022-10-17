import numpy as np
import matplotlib.pyplot as plt
import pickle
import matlab
import math
import scipy.io
import os


loc = "wsfodins"
"""
if not os.path.isdir("../data/plot/category/"+loc):
    os.makedirs("../data/plot/category/"+loc)
data = pickle.load(open("../data/plot/feature_info/"+loc,"rb"))
"""
data=[]
odins_data=[]
i=0
while True:
    i += 1
    if not os.path.isfile("../data/plot/kernel/"+loc+"/"+str(i)+".csv"):
        #print("break",i)
        break
    with open("../data/plot/kernel/"+loc+"/"+str(i)+".csv") as f:
        get=float(f.read())
        #print(get)
        data.append(get)
i=0
while True:
    i += 1
    if not os.path.isfile("../data/plot/kernel/odins/"+str(i)+".csv"):
        #print("break",i)
        break
    with open("../data/plot/kernel/odins/"+str(i)+".csv") as f:
        get=float(f.read())
        #print(get)
        odins_data.append(get)

data = np.array(data)
odins_data= np.array(odins_data)
#data = data[0]
print("len=",len(data))

if not os.path.isdir("../data/plot/category/"+loc):
    os.makedirs("../data/plot/category/"+loc)

fig = plt.figure()
fig.set_figheight(6)
fig.set_figwidth(9)
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")

"""
for i in range(len(data)):
    print(str(i+1) + "th = " + str(data[i]))
"""
srt = sorted(data[:3093],reverse=True)
art = np.argsort(data[:3093])

for i in range(3094):
    print(i,srt[i])
    if srt[i]<0.7:
        
        break

print("Best 100",sorted(art[-100:]))
#print(data[3031],srt[99])
ax1.plot(odins_data)
ax1.plot(data*0.5)
#ax1.plot([2]*len(data))
#ax1.plot([0.8]*len(data))
#plt.show()
plt.close("all")


fig = plt.figure()
fig.set_figheight(6)
fig.set_figwidth(10)
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.scatter(range(len(data[:3093])),data[:3093],color="r")
ax1.plot(range(len(data[:3093])),[0.7]*len(data[:3093]),color="b")
#ax1.plot(range(len(data[:3093])),[0.1]*len(data[:3093]))
#plt.ylim(0,1.4)
plt.xlabel("Feature Index",fontsize=21)
plt.ylabel("Information Leakage",fontsize=21)
#plt.yticks([0,0.1,0.5,0.7,1,1.2,1.4])
#plt.legend(fontsize=17)
plt.tick_params(labelsize=17)
plt.tight_layout()

fig.savefig("../data/plot/category/"+loc+"/all.pdf")
#plt.show()

Count = data[0:13]
fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(15)
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.scatter(range(1,14),data[0:13])
ax1.scatter(range(1,14),odins_data[0:13])
ax1.plot(range(1,14),[0.7]*len(range(1,14)),color="g")
ax1.plot(range(1,14),[0.1]*len(range(1,14)),color="g")
plt.xlim(1,13)
plt.ylim(0,1.4)
fig.savefig("../data/plot/category/"+loc+"/count.png")


Time = data[13:37]
fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(15)
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.scatter(range(14,38),data[13:37])
ax1.scatter(range(14,38),odins_data[13:37])
ax1.plot(range(14,38),[0.7]*len(range(14,38)),color="g")
ax1.plot(range(14,38),[0.1]*len(range(14,38)),color="g")
plt.xlim(14,37)
plt.ylim(0,1.4)
fig.savefig("../data/plot/category/"+loc+"/Time.png")

Ngram = data[37:161]
fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(15)
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.scatter(range(38,162),data[37:161])
ax1.scatter(range(38,162),odins_data[37:161])
ax1.plot(range(38,162),[0.7]*len(range(38,162)),color="g")
ax1.plot(range(38,162),[0.1]*len(range(38,162)),color="g")
plt.xlim(38,161)
plt.ylim(0,1.4)
fig.savefig("../data/plot/category/"+loc+"/Ngram.png")

Transposition = data[161:765]
fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(15)
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.scatter(range(162,766),data[161:765])
ax1.scatter(range(162,766),odins_data[161:765])
ax1.plot(range(162,766),[0.7]*len(range(162,766)),color="g")
ax1.plot(range(162,766),[0.1]*len(range(162,766)),color="g")
plt.xlim(162,765)
plt.ylim(0,1.4)
fig.savefig("../data/plot/category/"+loc+"/Transposition.png")

IntervalI = data[765:1365]
fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(15)
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.scatter(range(766,1366),data[765:1365])
ax1.scatter(range(766,1366),odins_data[765:1365])
ax1.plot(range(766,1366),[0.7]*len(range(766,1366)),color="g")
ax1.plot(range(766,1366),[0.1]*len(range(766,1366)),color="g")
plt.xlim(766,1365)
plt.ylim(0,1.4)
fig.savefig("../data/plot/category/"+loc+"/Interval-I.png")

plt.close("all")

IntervalII = data[1365:1967]
fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(15)
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.scatter(range(1366,1968),data[1365:1967])
ax1.scatter(range(1366,1968),odins_data[1365:1967])
ax1.plot(range(1366,1968),[0.7]*len(range(1366,1968)),color="g")
ax1.plot(range(1366,1968),[0.1]*len(range(1366,1968)),color="g")
plt.xlim(1366,1967)
plt.ylim(0,1.4)
fig.savefig("../data/plot/category/"+loc+"/Interval-II.png")

IntervalIII = data[1967:2553]
fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(15)
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.scatter(range(1968,2554),data[1967:2553])
ax1.scatter(range(1968,2554),odins_data[1967:2553])
ax1.plot(range(1967,2553),[0.7]*len(range(1967,2553)),color="g")
ax1.plot(range(1967,2553),[0.1]*len(range(1967,2553)),color="g")
plt.xlim(1968,2553)
plt.ylim(0,1.4)
fig.savefig("../data/plot/category/"+loc+"/Interval-III.png")

plt.close("all")

Distribution = data[2553:2778]
fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(15)
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.scatter(range(2554,2779),data[2553:2778])
ax1.scatter(range(2554,2779),odins_data[2553:2778])
ax1.plot(range(2553,2778),[0.7]*len(range(2553,2778)),color="g")
ax1.plot(range(2553,2778),[0.1]*len(range(2553,2778)),color="g")
plt.xlim(2554,2778)
plt.ylim(0,1.4)
#plt.show()
fig.savefig("../data/plot/category/"+loc+"/Distribution.png")

Burst = data[2778:2789]
fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(15)
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.scatter(range(2779,2790),data[2778:2789])
ax1.scatter(range(2779,2790),odins_data[2778:2789])
ax1.plot(range(2779,2790),[0.7]*len(range(2779,2790)),color="g")
ax1.plot(range(2779,2790),[0.1]*len(range(2779,2790)),color="g")
plt.xlim(2779,2789)
plt.ylim(0,1.4)
fig.savefig("../data/plot/category/"+loc+"/Burst.png")

First20 = data[2789:2809]
fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(15)
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.scatter(range(2790,2810),data[2789:2809])
ax1.scatter(range(2790,2810),odins_data[2789:2809])
ax1.plot(range(2790,2810),[0.7]*len(range(2790,2810)),color="g")
ax1.plot(range(2790,2810),[0.1]*len(range(2790,2810)),color="g")
plt.xlim(2790,2809)
plt.ylim(0,1.4)
ax1.xaxis.set_ticks([2790,2795,2800,2805,2809])
fig.savefig("../data/plot/category/"+loc+"/First20.png")

First30c = data[2809:2811]
fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(15)
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.scatter(range(2810,2812),data[2809:2811])
ax1.scatter(range(2810,2812),odins_data[2809:2811])
ax1.plot(range(2809,2811),[0.7]*len(range(2809,2811)),color="g")
ax1.plot(range(2809,2811),[0.1]*len(range(2809,2811)),color="g")
plt.xlim(2810,2811)
plt.ylim(0,1.4)
fig.savefig("../data/plot/category/"+loc+"/First30c.png")

Last30c = data[2811:2813]
fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(15)
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.scatter(range(2812,2814),data[2811:2813])
ax1.scatter(range(2812,2814),odins_data[2811:2813])
ax1.plot(range(2811,2813),[0.7]*len(range(2811,2813)),color="g")
ax1.plot(range(2811,2813),[0.1]*len(range(2811,2813)),color="g")
plt.xlim(2812,2813)
plt.ylim(0,1.4)
fig.savefig("../data/plot/category/"+loc+"/Last30c.png")

plt.close("all")

PerSec = data[2813:2939]
fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(15)
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.scatter(range(2814,2940),data[2813:2939])
ax1.scatter(range(2814,2940),odins_data[2813:2939])
ax1.plot(range(2814,2940),[0.7]*len(range(2813,2939)),color="g")
ax1.plot(range(2814,2940),[0.1]*len(range(2813,2939)),color="g")
plt.xlim(2814,2939)
plt.ylim(0,1.4)
#plt.show()
fig.savefig("../data/plot/category/"+loc+"/perSec.png")

plt.close("all")

CUMUL = data[2939:3043]
fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(15)
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.scatter(range(2940,3044),data[2939:3043])
ax1.scatter(range(2940,3044),odins_data[2939:3043])
ax1.plot(range(2940,3044),[0.7]*len(range(2939,3043)),color="g")
ax1.plot(range(2940,3044),[0.1]*len(range(2939,3043)),color="g")
plt.xlim(2940,3043)
plt.ylim(0,1.4)
#plt.show()
fig.savefig("../data/plot/category/"+loc+"/CUMUL.png")

CDNBurst = data[3043:3093]
fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(15)
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.scatter(range(3044,3094),data[3043:3093])
ax1.scatter(range(3044,3094),odins_data[3043:3093])
ax1.plot(range(3044,3094),[0.7]*len(range(3043,3093)),color="g")
ax1.plot(range(3044,3094),[0.1]*len(range(3043,3093)),color="g")
plt.xlim(3044,3093)
plt.ylim(0,1.4)
fig.savefig("../data/plot/category/"+loc+"/CDNBurst.png")

CUMUL50 = data[3093:3143]
fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(15)
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.scatter(range(50),data[3093:3143],label="WSF")
ax1.scatter(range(50),odins_data[3093:3143],label="WPF")
ax1.plot(range(3093,3143),[0.7]*len(range(3093,3143)),color="g")
ax1.plot(range(3093,3143),[0.1]*len(range(3093,3143)),color="g")
#plt.xlim(3094,3143)
plt.ylim(0,1.4)
fig.savefig("../data/plot/category/"+loc+"/CUMUL50.png")


fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(15)
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.scatter(range(3143,3193),data[3143:3193],label="WSF")
ax1.scatter(range(3143,3193),odins_data[3143:3193],label="WPF")
ax1.plot(range(3143,3193),[0.7]*len(range(3143,3193)),color="g")
ax1.plot(range(3143,3193),[0.1]*len(range(3143,3193)),color="g")
#plt.xlim(3144,3193)
plt.ylim(0,1.4)
fig.savefig("../data/plot/category/"+loc+"/CDNburst.png")


fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(15)
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.scatter(range(3193,3243),data[3193:3243],label="WSF")
ax1.scatter(range(3193,3243),odins_data[3193:3243],label="WPF")
ax1.plot(range(3193,3243),[0.7]*len(range(3193,3243)),color="g")
#plt.xlim(3194,3243)
plt.ylim(0,1.4)
fig.savefig("../data/plot/category/"+loc+"/TimeCUMUL.png")

fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.scatter(range(3394,3494),data[3393:3493])
plt.xlim(3394,3493)
plt.ylim(0,1.4)
fig.savefig("../data/plot/category/"+loc+"/Per0.02Sec.png")

fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.scatter(range(3494,3544),data[3493:3543])
ax1.scatter(range(3494,3544),odins_data[3493:3543])
plt.xlim(3494,3543)
plt.ylim(0,1.4)
fig.savefig("../data/plot/category/"+loc+"/TimeBurst.png")


"""
fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.scatter(range(50),range(3205,3430),data[3204:3430])
plt.xlim(3204,3429)
plt.ylim(0,1.4)
fig.savefig("../data/plot/category/"+loc+"/inDist.png")
"""

plt.close("all")
fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(15)
#plt.rcParams['figure.subplot.bottom'] = 0.5
ax1 = fig.add_subplot(111,xlabel="feature index",ylabel="information leakage")

ax1.scatter(range(50),data[3093:3143],label="CUMUL")
plt.ylim(0,1.4)
srt = sorted(data[3093:3143],reverse=True)
art = np.argsort(data[3093:3143])
print("CUMUL",srt[0],max(data[3093:3143]),art[-1],data[3093:3143][art[-1]])

ax1.scatter(range(50),data[3143:3193],label="CDN")
plt.ylim(0,1.4)
srt = sorted(data[3143:3193],reverse=True)
art = np.argsort(data[3143:3193])
print("CDN",srt[0],max(data[3143:3193]),art[-1],data[3143:3193][art[-1]])

#ax2 = fig.add_subplot(132,xlabel="index",label="CDNburst")
ax1.scatter(range(50),data[3193:3243],label="Time")
srt = sorted(data[3193:3243],reverse=True)
art = np.argsort(data[3193:3243])
print("Time cumul",srt[0],max(data[3193:3243]),art[-1],data[3193:3243][art[-1]])

"""
#ax3 = fig.add_subplot(133,xlabel="index",label="timesplit")
ax1.scatter(range(50),data[3243:3293],label="burst")
srt = sorted(data[3243:3293],reverse=True)
art = np.argsort(data[3243:3293])
print("burst",srt[0],max(data[3243:3293]),art[-1],data[3243:3293][art[-1]])
"""

plt.xlabel("feature index",fontsize=17)
plt.ylabel("information leakage",fontsize=17)
#plt.legend(fontsize=17)
plt.tick_params(labelsize=17)
plt.tight_layout()

#plt.show()
fig.savefig("../data/plot/category/"+loc+"/evaluate.png")





fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(15)
#plt.rcParams['figure.subplot.bottom'] = 0.5
ax1 = fig.add_subplot(111,xlabel="feature index",ylabel="information leakage")

ax1.scatter(range(50),data[3093:3143],label="CUMUL")
plt.ylim(0,1.4)
srt = sorted(data[3093:3143],reverse=True)
art = np.argsort(data[3093:3143])

ax1.scatter(range(50),data[2943:2993],label="inCUMUL")
plt.ylim(0,1.4)
srt = sorted(data[2943:2993],reverse=True)
art = np.argsort(data[2943:2993])

#ax2 = fig.add_subplot(132,xlabel="index",label="CDNburst")
ax1.scatter(range(50),data[2993:3043],label="outCUMUL")
srt = sorted(data[2993:3043],reverse=True)
art = np.argsort(data[2993:3043])

plt.xlabel("feature index",fontsize=17)
plt.ylabel("information leakage",fontsize=17)
plt.legend(fontsize=17)
plt.tick_params(labelsize=17)
plt.tight_layout()
fig.savefig("../data/plot/category/"+loc+"/evCUMUL.png")



fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(15)
#plt.rcParams['figure.subplot.bottom'] = 0.5
ax1 = fig.add_subplot(111,xlabel="feature index",ylabel="information leakage")

ax1.scatter(range(50),data[3093:3143],label="CUMUL")
plt.ylim(0,1.4)
srt = sorted(data[3093:3143],reverse=True)
art = np.argsort(data[3093:3143])

print(len(data[3493:3543]))
ax1.scatter(range(50),data[3493:3543],label="inCUMUL")

#ax2 = fig.add_subplot(132,xlabel="index",label="CDNburst")
ax1.scatter(range(50),data[3543:3593],label="outCUMUL")

plt.xlabel("feature index",fontsize=17)
plt.ylabel("information leakage",fontsize=17)
plt.legend(fontsize=17)
plt.tick_params(labelsize=17)
plt.tight_layout()
fig.savefig("../data/plot/category/"+loc+"/evCUMUL2.png")

plt.close("all")
fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(15)
#plt.rcParams['figure.subplot.bottom'] = 0.5
ax1 = fig.add_subplot(111,xlabel="feature index",ylabel="information leakage")
plt.ylim(0,1.4)

ax1.scatter(range(50),data[3193:3243],label="Time cumul")
ax1.scatter(range(50),data[3293:3343],label="in time")
ax1.scatter(range(50),data[3343:3393],label="out time")

plt.xlabel("feature index",fontsize=17)
plt.ylabel("information leakage",fontsize=17)
plt.legend(fontsize=17)
plt.tick_params(labelsize=17)
plt.tight_layout()
fig.savefig("../data/plot/category/"+loc+"/evTimecumul.png")
plt.close("all")

PacketCount = []
PacketCount.extend(Count)
PacketCount.extend(Distribution)
PacketCount.extend(Burst)
PacketCount.extend(First30c)
PacketCount.extend(Last30c)
PacketCount.extend(PerSec)
print(len(PacketCount))

PacketArrivalTime = []
PacketArrivalTime.extend(Time)
print(len(PacketArrivalTime))

PacketSize = []
PacketSize.extend(CDNBurst)
PacketSize.extend(First20)
PacketSize.extend(CUMUL)
print(len(PacketSize))

PacketOrder = []
PacketOrder.extend(Ngram)
PacketOrder.extend(Transposition)
PacketOrder.extend(IntervalI)
PacketOrder.extend(IntervalII)
PacketOrder.extend(IntervalIII)
print(len(PacketOrder))

print(len(PacketCount) + len(PacketArrivalTime) + len(PacketSize) + len(PacketOrder))

fig = plt.figure()
fig.set_figheight(4)
fig.set_figwidth(10)
ax1 = fig.add_subplot(111)
ax1.scatter(range(len(PacketCount)),PacketCount,color="r")
ax1.plot(range(len(PacketCount)),[0.7]*len(PacketCount),color="b")
plt.xlabel("Feature Index",fontsize=21)
plt.ylabel("Information Leakage",fontsize=21)
plt.tick_params(labelsize=17)
plt.tight_layout()
ax1.set_xlim(0, len(PacketCount))
ax1.set_ylim(0, 1.5)
fig.savefig("../data/plot/category/"+loc+"/PacketCount.pdf")
plt.show()

fig = plt.figure()
fig.set_figheight(4)
fig.set_figwidth(10)
ax1 = fig.add_subplot(111)
ax1.scatter(range(len(PacketOrder)),PacketOrder,color="r")
ax1.plot(range(len(PacketOrder)),[0.7]*len(PacketOrder),color="b")
plt.xlabel("Feature Index",fontsize=21)
plt.ylabel("Information Leakage",fontsize=21)
plt.tick_params(labelsize=17)
plt.tight_layout()
ax1.set_xlim(0, len(PacketOrder))
ax1.set_ylim(0, 1.5)
fig.savefig("../data/plot/category/"+loc+"/PacketOrder.pdf")
plt.show()

fig = plt.figure()
fig.set_figheight(4)
fig.set_figwidth(10)
ax1 = fig.add_subplot(111)
ax1.scatter(range(len(PacketArrivalTime)),PacketArrivalTime,color="r")
ax1.plot(range(len(PacketArrivalTime)),[0.7]*len(PacketArrivalTime),color="b")
plt.xlabel("Feature Index",fontsize=21)
plt.ylabel("Information Leakage",fontsize=21)
plt.tick_params(labelsize=17)
plt.tight_layout()
ax1.set_xlim(0, len(PacketArrivalTime))
ax1.set_ylim(0, 1.5)
fig.savefig("../data/plot/category/"+loc+"/PacketArrivalTime.pdf")
plt.show()

fig = plt.figure()
fig.set_figheight(4)
fig.set_figwidth(10)
ax1 = fig.add_subplot(111)
ax1.scatter(range(len(PacketSize)),PacketSize,color="r")
ax1.plot(range(len(PacketSize)),[0.7]*len(PacketSize),color="b")
plt.xlabel("Feature Index",fontsize=21)
plt.ylabel("Information Leakage",fontsize=21)
plt.tick_params(labelsize=17)
plt.tight_layout()
ax1.set_xlim(0, len(PacketSize))
ax1.set_ylim(0, 1.5)
fig.savefig("../data/plot/category/"+loc+"/PacketSize.pdf")
plt.show()


"""
fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.scatter(range(50),data)
plt.ylim(0,1.4)
fig.savefig("../data/plot/category/"+loc+"/All.png")
"""
#print(data[1365:1380])

