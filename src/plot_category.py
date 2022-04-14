from scipy.stats import gaussian_kde
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import pickle
import matlab
from scipy import integrate
from scipy.integrate import cumtrapz
import sympy as sym
import math
import scipy.io
import os


loc = "icn"
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
print(art[-100:])
print(data[3031],srt[99])
fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(len(data[:3093])),data[:3093])
ax1.plot(range(len(data[:3093])),[srt[99]]*len(data[:3093]))
plt.ylim(0,1)
fig.savefig("../data/plot/category/"+loc+"/all.png")
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(1,14),data[0:13])
plt.xlim(1,13)
plt.ylim(0,1)
fig.savefig("../data/plot/category/"+loc+"/count.png")

fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(14,38),data[13:37])
plt.xlim(14,37)
plt.ylim(0,1)
fig.savefig("../data/plot/category/"+loc+"/Time.png")

fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(38,162),data[37:161])
plt.xlim(38,161)
plt.ylim(0,1)
fig.savefig("../data/plot/category/"+loc+"/Ngram.png")


fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(162,766),data[161:765])
plt.xlim(162,765)
plt.ylim(0,1)
fig.savefig("../data/plot/category/"+loc+"/Transposition.png")

fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(766,1366),data[765:1365])
plt.xlim(766,1365)
plt.ylim(0,1)
fig.savefig("../data/plot/category/"+loc+"/Interval-I.png")

fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(1366,1968),data[1365:1967])
plt.xlim(1366,1967)
plt.ylim(0,1)
fig.savefig("../data/plot/category/"+loc+"/Interval-II.png")

fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(1968,2554),data[1967:2553])
plt.xlim(1968,2553)
plt.ylim(0,1)
fig.savefig("../data/plot/category/"+loc+"/Interval-III.png")

fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(2554,2779),data[2553:2778])
plt.xlim(2554,2778)
plt.ylim(0,1)
fig.savefig("../data/plot/category/"+loc+"/Distribution.png")

fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(2779,2790),data[2778:2789])
plt.xlim(2779,2789)
plt.ylim(0,1)
fig.savefig("../data/plot/category/"+loc+"/Burst.png")

fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(2790,2810),data[2789:2809])
plt.xlim(2790,2809)
plt.ylim(0,1)
ax1.xaxis.set_ticks([2790,2795,2800,2805,2809])
fig.savefig("../data/plot/category/"+loc+"/First20.png")

fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(2810,2812),data[2809:2811])
plt.xlim(2810,2811)
plt.ylim(0,1)
fig.savefig("../data/plot/category/"+loc+"/First30c.png")

fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(2812,2814),data[2811:2813])
plt.xlim(2812,2813)
plt.ylim(0,1)
fig.savefig("../data/plot/category/"+loc+"/Last30c.png")

fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(2814,2940),data[2813:2939])
plt.xlim(2814,2939)
plt.ylim(0,1)
fig.savefig("../data/plot/category/"+loc+"/perSec.png")

fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(2940,3044),data[2939:3043])
plt.xlim(2940,3043)
plt.ylim(0,1)
fig.savefig("../data/plot/category/"+loc+"/CUMUL.png")

fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(3044,3094),data[3043:3093])
plt.xlim(3044,3093)
plt.ylim(0,1)
fig.savefig("../data/plot/category/"+loc+"/CDNBurst.png")

"""
fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(3094,3194),data[3093:3193])
plt.xlim(3094,3194)
plt.ylim(0,1)
fig.savefig("../data/plot/category/"+loc+"/Time100.png")

fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(3194,3205),data[3193:3204])
plt.xlim(3194,3204)
plt.ylim(0,1)
fig.savefig("../data/plot/category/"+loc+"/inburst.png")
"""

"""
fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(3205,3430),data[3204:3430])
plt.xlim(3204,3429)
plt.ylim(0,1)
fig.savefig("../data/plot/category/"+loc+"/inDist.png")
"""

fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(8)
#plt.rcParams['figure.subplot.bottom'] = 0.5
ax1 = fig.add_subplot(111,xlabel="feature index",ylabel="information leakage")
ax1.scatter(range(50),data[-150:-100],label="CUMUL")
plt.ylim(0,1)

srt = sorted(data[-150:-100],reverse=True)
art = np.argsort(data[-150:-100])
print(srt[0],max(data[-150:-100]),art[-1],data[-150:-101][art[-1]])

#ax2 = fig.add_subplot(132,xlabel="index",label="CDNburst")
ax1.scatter(range(50),data[-100:-50],label="CDNburst")


#ax3 = fig.add_subplot(133,xlabel="index",label="timesplit")
ax1.scatter(range(50),data[-50:],label="Time cumul")


plt.xlabel("feature index",fontsize=17)
plt.ylabel("information leakage",fontsize=17)
plt.legend(fontsize=17)
plt.tick_params(labelsize=17)
plt.tight_layout()

#plt.show()
fig.savefig("../data/plot/evaluate.png")


"""
fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(data)
plt.ylim(0,1)
fig.savefig("../data/plot/category/"+loc+"/All.png")
"""
#print(data[1365:1380])

