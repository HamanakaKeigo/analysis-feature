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


def calc_kernel(sites=[],target=""):

    information_leakage=[]
    Feature_data = []


    for site in sites:

        with open("../data/features/total/"+site,"rb") as feature_set:
            data = pickle.load(feature_set)
            #[site][feature] >> [feature][site]

            for i in range(len(data)):
                for j in range(len(data[i])):

                    if(len(Feature_data)<=j):
                        Feature_data.append({})
                    if site not in Feature_data[j]:
                        Feature_data[j][site] = []
                    if "all" not in Feature_data[j]:
                        Feature_data[j]["all"] = []

                    Feature_data[j][site].append(data[i][j])
                    Feature_data[j]["all"].append(data[i][j])



            #Feature_data[key][s[1]] = np.reshape(Feature_data[key][s[1]],(-1,1))


    for key in range(len(Feature_data)):

        fd = np.array(Feature_data[key]["all"]).reshape(-1,1)
        fac = len(fd) ** -0.2
        width = abs((fac**2)*np.var(fd,ddof=1))
        if(width==0):
            information_leakage.append(0)
            continue
        bw = 1.06*(width**0.5)
        print(str(key) + "th bw = " + str(bw))
        xticks = np.linspace(fd.min()-bw*4, fd.max()+bw*4, 10000)
        Xticks = np.reshape(xticks,(-1,1))

        kde = KernelDensity(kernel="gaussian",bandwidth=bw).fit(fd)
        estimate = np.exp(kde.score_samples(Xticks))

        
        fig = plt.figure()
        ax1 = fig.add_subplot(111,xlabel=key,ylabel="rate",label="all")
        ax1.plot(xticks,estimate*len(fd))
    
    

        Hcf=0
        Hc=0
        for site in sites:
        
            sfd = np.array(Feature_data[key][site]).reshape(-1,1)
            kde_s = KernelDensity(kernel="gaussian",bandwidth=bw).fit(sfd)
            estimate_s = np.exp(kde_s.score_samples(Xticks))
            #print(site +" of len : " + str(len(Feature_data[site])))

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

            #print("mutual of " + site + " : " + str(mutual[-1]))
            Hcf+=mutual[-1]
            
            ax1.plot(xticks,estimate_s*len(sfd),label=site)

        #print("total Hcf : " + str(-Hcf))
        #print("total Hc : " + str(-Hc))
        print("total mutual : "+str(Hcf-Hc))
        information_leakage.append(Hcf-Hc)
        plt.legend()
        #plt.show()
        fig.savefig("../data/plot/kernel/"+target+"/"+str(key+1)+".png")
        print("--------------------")
    return(information_leakage)


if __name__ == "__main__":
    sites = []
    target = "WSF"

    if(target == "WSF"):
        with open("../data/sites",'r') as f1:
            site_list = f1.readlines()
            for site in site_list:
                s = site.split()
                if s[0] == "#":
                    continue
                sites.append(s[1])

    elif(target == "WPF"):
        for i in range(12):
            sites.append("Amazonjp"+str(i))


    data = calc_kernel(sites,target)



    #print(information_leakage)
    test=[]
    #test.append(information_leakage["max"])

    fig = plt.figure()
    ax1 = fig.add_subplot(111,xlabel="feature",ylabel="rate")
    ax1.plot(data)
    #plt.show()
    fig.savefig("../data/all_data.png")

    if(target == "WSF"):
        f = open('../data/plot/feature_info/wsf', 'wb')
        pickle.dump(data,f)
        f.close()
    elif(target == "WPF"):
        f = open('../data/plot/feature_info/wpf', 'wb')
        pickle.dump(data,f)
        f.close()