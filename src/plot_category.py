import numpy as np
import matplotlib.pyplot as plt
import pickle
import matlab
import math
import scipy.io
import os


loc = "odins"
"""
if not os.path.isdir("../data/plot/category/"+loc):
    os.makedirs("../data/plot/category/"+loc)
data = pickle.load(open("../data/plot/feature_info/"+loc,"rb"))
"""
data=[]
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

#data = data[0]
print("len=",len(data))

if not os.path.isdir("../data/plot/category/"+loc):
    os.makedirs("../data/plot/category/"+loc)



"""
for i in range(len(data)):
    print(str(i+1) + "th = " + str(data[i]))
"""
srt = sorted(data[:3093],reverse=True)
art = np.argsort(data[:3093])
print("Best 100",sorted(art[-100:]))
#print(data[3031],srt[99])

fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(8)
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(len(data[:3093])),data[:3093])
ax1.plot(range(len(data[:3093])),[srt[99]]*len(data[:3093]))
plt.ylim(0,1.5)
plt.xlabel("feature index",fontsize=17)
plt.ylabel("information leakage",fontsize=17)
#plt.legend(fontsize=17)
plt.tick_params(labelsize=17)
plt.tight_layout()

fig.savefig("../data/plot/category/"+loc+"/all.png")
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(1,14),data[0:13])
plt.xlim(1,13)
plt.ylim(0,1.5)
fig.savefig("../data/plot/category/"+loc+"/count.png")

fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(14,38),data[13:37])
plt.xlim(14,37)
plt.ylim(0,1.5)
fig.savefig("../data/plot/category/"+loc+"/Time.png")

fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(38,162),data[37:161])
plt.xlim(38,161)
plt.ylim(0,1.5)
fig.savefig("../data/plot/category/"+loc+"/Ngram.png")


fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(162,766),data[161:765])
plt.xlim(162,765)
plt.ylim(0,1.5)
fig.savefig("../data/plot/category/"+loc+"/Transposition.png")

fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(766,1366),data[765:1365])
plt.xlim(766,1365)
plt.ylim(0,1.5)
fig.savefig("../data/plot/category/"+loc+"/Interval-I.png")

fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(1366,1968),data[1365:1967])
plt.xlim(1366,1967)
plt.ylim(0,1.5)
fig.savefig("../data/plot/category/"+loc+"/Interval-II.png")

fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(1968,2554),data[1967:2553])
plt.xlim(1968,2553)
plt.ylim(0,1.5)
fig.savefig("../data/plot/category/"+loc+"/Interval-III.png")

fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(2554,2779),data[2553:2778])
plt.xlim(2554,2778)
plt.ylim(0,1.5)
fig.savefig("../data/plot/category/"+loc+"/Distribution.png")

fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(2779,2790),data[2778:2789])
plt.xlim(2779,2789)
plt.ylim(0,1.5)
fig.savefig("../data/plot/category/"+loc+"/Burst.png")

fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(2790,2810),data[2789:2809])
plt.xlim(2790,2809)
plt.ylim(0,1.5)
ax1.xaxis.set_ticks([2790,2795,2800,2805,2809])
fig.savefig("../data/plot/category/"+loc+"/First20.png")
plt.close()

fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(2810,2812),data[2809:2811])
plt.xlim(2810,2811)
plt.ylim(0,1.5)
fig.savefig("../data/plot/category/"+loc+"/First30c.png")

fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(2812,2814),data[2811:2813])
plt.xlim(2812,2813)
plt.ylim(0,1.5)
fig.savefig("../data/plot/category/"+loc+"/Last30c.png")

fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(2814,2940),data[2813:2939])
plt.xlim(2814,2939)
plt.ylim(0,1.5)
fig.savefig("../data/plot/category/"+loc+"/perSec.png")

fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(2940,3044),data[2939:3043])
plt.xlim(2940,3043)
plt.ylim(0,1.5)
fig.savefig("../data/plot/category/"+loc+"/CUMUL.png")

fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(3044,3094),data[3043:3093])
plt.xlim(3044,3093)
plt.ylim(0,1.5)
fig.savefig("../data/plot/category/"+loc+"/CDNBurst.png")


fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(3094,3144),data[3093:3143])
plt.xlim(3094,3143)
plt.ylim(0,1.5)
fig.savefig("../data/plot/category/"+loc+"/CUMUL50.png")
plt.close()

fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(3144,3194),data[3143:3193])
plt.xlim(3144,3193)
plt.ylim(0,1.5)
fig.savefig("../data/plot/category/"+loc+"/CDNburst.png")

fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(3194,3244),data[3193:3243])
plt.xlim(3194,3243)
plt.ylim(0,1.5)
fig.savefig("../data/plot/category/"+loc+"/TimeCUMUL.png")


"""
fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(3205,3430),data[3204:3430])
plt.xlim(3204,3429)
plt.ylim(0,1.5)
fig.savefig("../data/plot/category/"+loc+"/inDist.png")
"""

plt.clf()
fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(8)
#plt.rcParams['figure.subplot.bottom'] = 0.5
ax1 = fig.add_subplot(111,xlabel="feature index",ylabel="information leakage")

ax1.scatter(range(50),data[3093:3143],label="CUMUL")
plt.ylim(0,1.5)
srt = sorted(data[3093:3143],reverse=True)
art = np.argsort(data[3093:3143])
print("CUMUL",srt[0],max(data[3093:3143]),art[-1],data[3093:3143][art[-1]])

ax1.scatter(range(50),data[3143:3193],label="CDN")
plt.ylim(0,1.5)
srt = sorted(data[3143:3193],reverse=True)
art = np.argsort(data[3143:3193])
print("CDN",srt[0],max(data[3143:3193]),art[-1],data[3143:3193][art[-1]])

#ax2 = fig.add_subplot(132,xlabel="index",label="CDNburst")
ax1.scatter(range(50),data[3193:3243],label="Time")
srt = sorted(data[3193:3243],reverse=True)
art = np.argsort(data[3193:3243])
print("Time cumul",srt[0],max(data[3193:3243]),art[-1],data[3193:3243][art[-1]])


#ax3 = fig.add_subplot(133,xlabel="index",label="timesplit")
ax1.scatter(range(50),data[3243:3293],label="burst")
srt = sorted(data[3243:3293],reverse=True)
art = np.argsort(data[3243:3293])
print("burst",srt[0],max(data[3243:3293]),art[-1],data[3243:3293][art[-1]])

plt.xlabel("feature index",fontsize=17)
plt.ylabel("information leakage",fontsize=17)
plt.legend(fontsize=17)
plt.tick_params(labelsize=17)
plt.tight_layout()

#plt.show()
fig.savefig("../data/plot/category/"+loc+"/evaluate.png")




plt.clf()
fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(8)
#plt.rcParams['figure.subplot.bottom'] = 0.5
ax1 = fig.add_subplot(111,xlabel="feature index",ylabel="information leakage")

ax1.scatter(range(50),data[3093:3143],label="CUMUL")
plt.ylim(0,1.5)
srt = sorted(data[3093:3143],reverse=True)
art = np.argsort(data[3093:3143])

ax1.scatter(range(50),data[2943:2993],label="inCUMUL")
plt.ylim(0,1.5)
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

plt.clf()
fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(8)
#plt.rcParams['figure.subplot.bottom'] = 0.5
ax1 = fig.add_subplot(111,xlabel="feature index",ylabel="information leakage")
plt.ylim(0,1.5)

ax1.scatter(range(50),data[3193:3243],label="Time cumul")
ax1.scatter(range(50),data[3293:3343],label="in time")
ax1.scatter(range(50),data[3343:3393],label="out time")

plt.xlabel("feature index",fontsize=17)
plt.ylabel("information leakage",fontsize=17)
plt.legend(fontsize=17)
plt.tick_params(labelsize=17)
plt.tight_layout()
fig.savefig("../data/plot/category/"+loc+"/evTimecumul.png")


"""
fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(data)
plt.ylim(0,1.5)
fig.savefig("../data/plot/category/"+loc+"/All.png")
"""
#print(data[1365:1380])

