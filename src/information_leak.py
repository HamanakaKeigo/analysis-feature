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





information_leakage=[]
Feature_data = []

with open("../data/sites",'r') as f1:
    sites = f1.readlines()
    for site in sites:
        s = site.split()
        if s[0] == "#":
            continue


        with open("../data/features/total/"+s[1],"rb") as f2:
            data = pickle.load(f2)
            #[site][feature] >> [feature][site]

            for i in range(len(data)):
                for j in range(len(data[i])):
                    if(len(Feature_data)<=j):
                        Feature_data.append({})
                    
                    if s[1] not in Feature_data[j]:
                        Feature_data[j][s[1]] = []
                    if "all" not in Feature_data[j]:
                        Feature_data[j]["all"] = []
                    Feature_data[j][s[1]].append(data[i][j])
                    Feature_data[j]["all"].append(data[i][j])



        #Feature_data[key][s[1]] = np.reshape(Feature_data[key][s[1]],(-1,1))


    for key in range(len(Feature_data)):

        fd = np.array(Feature_data[key]["all"]).reshape(-1,1)
        fac = len(fd) ** -0.2
        width = (fac**2)*np.var(fd,ddof=1)
        bw = 1.06*(width**0.5)
        print("bw = " + str(bw))
        xticks = np.linspace(fd.min()-bw*4, fd.max()+bw*4, 10000)
        Xticks = np.reshape(xticks,(-1,1))

        kde = KernelDensity(kernel="gaussian",bandwidth=bw).fit(fd)
        estimate = np.exp(kde.score_samples(Xticks))

        """
        fig = plt.figure()
        ax1 = fig.add_subplot(111,xlabel="total",ylabel="rate")
        ax1.plot(xticks,estimate*len(fd))
        """
        

        Hcf=0
        Hc=0
        for site in sites:
            s = site.split()
            if s[0] == "#":
                continue
            name = s[1]

            sfd = np.array(Feature_data[key][name]).reshape(-1,1)
            kde_s = KernelDensity(kernel="gaussian",bandwidth=bw).fit(sfd)
            estimate_s = np.exp(kde_s.score_samples(Xticks))
            #print(name +" of len : " + str(len(Feature_data[name])))

            rate = len(sfd)/len(fd)
            Hc += rate*math.log2(rate)
            rate = (estimate_s*len(sfd))/(estimate*len(fd))
            rate = np.nan_to_num(rate)
            #print("est = "+str(estimate))
            #print("est_s = "+str(estimate_s))
            pc = rate*np.log2(rate)
            pc = np.nan_to_num(pc)
            x = pc*estimate
            mutual = cumtrapz(x,xticks)

            #print("mutual of " + name + " : " + str(mutual[-1]))
            Hcf+=mutual[-1]
            
            #ax1.plot(xticks,estimate_s*len(sfd),label=name)

        print("total Hcf : " + str(-Hcf))
        print("total Hc : " + str(-Hc))
        print("total mutual : "+str(Hcf-Hc))
        information_leakage.append(Hcf-Hc)
        #plt.legend()
        #plt.show()
        #fig.savefig("../data/plot/"+str(key)+".png")
        print("--------------------")



print(information_leakage)
test=[]
#test.append(information_leakage["max"])

fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel="feature",ylabel="rate")
ax1.plot(information_leakage)
plt.legend()
plt.show()
fig.savefig("../data/test.png")