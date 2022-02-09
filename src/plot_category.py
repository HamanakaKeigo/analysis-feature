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
import math
import scipy.io


data = pickle.load(open("../data/plot/origin","rb"))

fig = plt.figure()
for i in range(len(data)):
    print(str(i+1) + "th = " + str(data[i]))

ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(1,14),data[0:13])
plt.xlim(1,13)
plt.ylim(0,2)
fig.savefig("../data/plot/count.png")

fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(14,38),data[13:37])
plt.xlim(14,37)
plt.ylim(0,2)
fig.savefig("../data/plot/Time.png")

fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(38,162),data[37:161])
plt.xlim(38,161)
plt.ylim(0,2)
fig.savefig("../data/plot/Ngram.png")


fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(162,766),data[161:765])
plt.xlim(162,765)
plt.ylim(0,2)
fig.savefig("../data/plot/Transposition.png")

fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(766,1366),data[765:1365])
plt.xlim(766,1365)
plt.ylim(0,2)
fig.savefig("../data/plot/Interval-I.png")

fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(1366,1968),data[1365:1967])
plt.xlim(1366,1967)
plt.ylim(0,2)
fig.savefig("../data/plot/Interval-II.png")

fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(1968,2554),data[1967:2553])
plt.xlim(1968,2553)
plt.ylim(0,2)
fig.savefig("../data/plot/Interval-III.png")

fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(2554,2779),data[2553:2778])
plt.xlim(2554,2778)
plt.ylim(0,2)
fig.savefig("../data/plot/Distribution.png")

fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(2779,2790),data[2778:2789])
plt.xlim(2779,2789)
plt.ylim(0,2)
fig.savefig("../data/plot/Burst.png")

fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(2790,2810),data[2789:2809])
plt.xlim(2790,2809)
plt.ylim(0,2)
ax1.xaxis.set_ticks([2790,2795,2800,2805,2809])
fig.savefig("../data/plot/First20.png")

fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(2810,2812),data[2809:2811])
plt.xlim(2810,2811)
plt.ylim(0,2)
fig.savefig("../data/plot/First30c.png")

fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(2812,2814),data[2811:2813])
plt.xlim(2812,2813)
plt.ylim(0,2)
fig.savefig("../data/plot/Last30c.png")

fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(2814,2940),data[2813:2939])
plt.xlim(2814,2939)
plt.ylim(0,2)
fig.savefig("../data/plot/perSec.png")

fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
ax1.plot(range(2940,3044),data[2939:3043])
plt.xlim(2940,3043)
plt.ylim(0,2)
fig.savefig("../data/plot/CUMUL.png")

#print(data[1365:1380])

