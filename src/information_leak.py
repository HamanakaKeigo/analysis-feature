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



infomation_leakage={}

with open("../data/feature",'r') as f1:
    features = f1.readlines()
    keys=[]
    for feature in features:
        f = feature.split()
        if f[0] == "#":
            continue
        Feature_data = {}
        with open("../data/sites",'r') as f2:
            sites = f2.readlines()
            for site in sites:
                s = site.split()
                if s[0] == "#":
                    continue

                with open("../data/features/"+f[0]+"/"+s[1],"rb") as f3:
                    data = pickle.load(f3)
                    for p in data:
                        for key in p.keys():
                            if key not in Feature_data:
                                Feature_data[key]={}
                                Feature_data[key]["all"]=np.empty(0)
                            if s[1] not in Feature_data[key]:
                                Feature_data[key][s[1]] = np.empty(0)
                            #print(p)
                            x = np.array(p[key])
                            #x = np.reshape(x,(-1,1))
                            Feature_data[key][s[1]] = np.append(Feature_data[key][s[1]],x)
                            Feature_data[key]["all"] = np.append(Feature_data[key]["all"],x)


                Feature_data[key][s[1]] = np.reshape(Feature_data[key][s[1]],(-1,1))


        for key in Feature_data.keys():
            print("key="+key)
            fd = np.array(Feature_data[key]["all"]).reshape(-1,1)
            #print(fd)
            fac = len(fd) ** -0.2
            width = (fac**2)*np.var(fd,ddof=1)
            bw = 1.06*(width**0.5)
            print("bw = " + str(bw))
            xticks = np.linspace(fd.min()-bw*4, fd.max()+bw*4, 10000)
            Xticks = np.reshape(xticks,(-1,1))

            kde = KernelDensity(kernel="gaussian",bandwidth=bw).fit(fd)
            estimate = np.exp(kde.score_samples(Xticks))

            fig = plt.figure()
            ax1 = fig.add_subplot(111,xlabel=f[0],ylabel="rate")
            ax1.plot(xticks,estimate*len(fd))
            #ax1.set_title(f[0])
        
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
                pc = rate*np.log2(rate)
                x = pc*estimate
                #print("rate = " + str(rate))
                #print("pc = "+str(pc))
                #print("x = " + str(x))
                mutual = cumtrapz(x,xticks)

                print("mutual of " + name + " : " + str(mutual[-1]))
                Hcf+=mutual[-1]

                ax1.plot(xticks,estimate_s*len(sfd),label=name)
                if(name=="www.osaka-u.ac.jp"):
                    mut = cumtrapz((estimate_s*len(sfd))/(estimate*len(fd)),xticks)
                    #print("mut = :"+str(mut[-1]))
                    #ax2 = fig.add_subplot(212,xlabel=f[0],ylabel="rate")
                    #ax2.plot(xticks,(estimate_s*len(sfd))/(estimate*len(fd)),label=name)

            print("total Hcf : " + str(-Hcf))
            print("total Hc : " + str(-Hc))
            print("total mutual : "+str(Hcf-Hc))
            infomation_leakage[key] = Hcf-Hc
            plt.legend()
            #plt.show()
            fig.savefig("../data/"+f[0]+".png")
            print("--------------------")
print(infomation_leakage)
test=[]
test.append(infomation_leakage["max"])
test.append(infomation_leakage["ave"])
test.append(infomation_leakage["len"])
test.append(infomation_leakage["g5"])
test.append(infomation_leakage["g10"])
test.append(infomation_leakage["g15"])
test.append(infomation_leakage["0"])
test.append(infomation_leakage["1"])
test.append(infomation_leakage["2"])
test.append(infomation_leakage["3"])
test.append(infomation_leakage["4"])
print(test)
fig = plt.figure()
ax1 = fig.add_subplot(111,xlabel=f[0],ylabel="rate")
ax1.plot(test,label=name)
plt.legend()
plt.show()
fig.savefig("../data/test.png")