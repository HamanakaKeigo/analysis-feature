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
import csv
import os
import shutil


def calc_kernel(sites=[],place="",savedir=""):

    Feature_data = []


    for loc in place:
        for i,site in enumerate(sites):
        
            #default
            with open("../data/features/"+loc+"/"+site,"rb") as feature_set:
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

    """
    with open("../data/features/odins/Amazon_music1","rb") as feature_set:
        data = pickle.load(feature_set)
        #[site][feature] >> [feature][site]
        
        for i in range(len(data)):
            if(i==100):
                break
            for j in range(len(data[i])):

                
                if(len(Feature_data)<=j):
                    Feature_data.append({})
                if site not in Feature_data[j]:
                    Feature_data[j][site] = []
                if "all" not in Feature_data[j]:
                    Feature_data[j]["all"] = []

                Feature_data[j][site].append(data[i][j])
                Feature_data[j]["all"].append(data[i][j])
    """
    print(len(Feature_data[1]["all"]))
    infos = []
    for i in range(2900,len(Feature_data)):
    #for i in [3097,3137,2947,2987,2997,3037]:
        infos.append(info(Feature_data,i,sites,loc,savedir))

    return([infos])
        

        #Feature_data[key][s[1]] = np.reshape(Feature_data[key][s[1]],(-1,1))
def info(Feature_data,key,sites,loc,savedir):
    var=[]
    fd = np.array(Feature_data[key]["all"]).reshape(-1,1)
    fac = len(fd) ** -0.2
    width = abs((fac**2)*np.var(fd,ddof=1))
    bw = 1.06*(width**0.5)
    #var.append("all var ="+str(np.std(fd,ddof=1)))
    if bw==0:
        Hc=0
        print("bw=",bw)
        with open("../data/plot/kernel/"+savedir+"/"+str(key+1)+".csv", 'w') as f:
            writer = csv.writer(f)
            writer.writerow([0])
        return(0)
        

    #print(str(key) + "th bw = " + str(bw))
    xticks = np.linspace(fd.min()-bw*4, fd.max()+bw*4, 1000)
    Xticks = np.reshape(xticks,(-1,1))

    kde = KernelDensity(kernel="gaussian",bandwidth=bw).fit(fd)
    estimate = np.exp(kde.score_samples(Xticks))
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111,xlabel=key,ylabel="rate")
    fig.set_figheight(6)
    fig.set_figwidth(9)
    plt.xlabel("Feature Value",fontsize=21)
    plt.ylabel("Probability Density",fontsize=21)
    plt.tick_params(labelsize=19)
    
    #ax1.plot(xticks,estimate*len(fd))


    #print(estimate)
    Hcf=0
    Hc=0
    for i,site in enumerate(sites):
        sfd = np.array(Feature_data[key][site]).reshape(-1,1)
        #print(sfd)
        kde_s = KernelDensity(kernel="gaussian",bandwidth=bw).fit(sfd)
        estimate_s = np.exp(kde_s.score_samples(Xticks))
        var.append(np.std(sfd,ddof=1))
        
        #print(site +" of len : " + str(len(Feature_data[site])))

        rate = len(sfd)/len(fd)
        Hc += rate*math.log2(rate)
        rate = (estimate_s*len(sfd))/(estimate*len(fd))
        rate = np.nan_to_num(rate)

        grap = (estimate_s*len(sfd))/(estimate*len(fd))
        grap = np.nan_to_num(grap)
        pc = rate*np.log2(rate)
        pc = np.nan_to_num(pc)
        x = pc*estimate
        mutual = cumtrapz(x,xticks)

        #print("mutual of " + site + " : " + str(mutual[-1]))
        Hcf+=mutual[-1]
        ax1.plot(xticks,estimate_s)
        #ax1.plot(xticks,grap)
    print("all var=",np.std(fd,ddof=1))
    print(np.mean(var))
    #print("total Hcf : " + str(-Hcf))
    #print("total Hc : " + str(-Hc))
    info = Hcf-Hc
    print(key+1,"/",len(Feature_data),"th total mutual : ",info)
    #plt.show()
    
    plt.tight_layout()
    fig.savefig("../data/plot/kernel/"+savedir+"/"+str(key+1)+".pdf")
    with open("../data/plot/kernel/"+savedir+"/"+str(key+1)+".csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow([info])
    
    print("--------------------")
    return(info)


if __name__ == "__main__":
    sites = []
    siteslist=[]
    place = ["odins"]
    savedir = "odins"
    odins11=[83, 65, 96, 8, 16, 67, 36, 41, 94, 73, 61]


    with open("../data/sites",'r') as f1:
        site_list = f1.readlines()
        for i,site in enumerate(site_list):
            s = site.split()
            if s[0] == "#":
                continue
            siteslist.append(s[1])
    """
    for i in odins11:
        sites.append(siteslist[i])
    print(sites)"""
    sites = siteslist

    if not os.path.isdir("../data/plot/kernel/"+savedir):
        os.makedirs("../data/plot/kernel/"+savedir)
    data = calc_kernel(sites,place,savedir)

    #f = open('../data/plot/feature_info/'+target, 'wb')
    f = open('../data/plot/feature_info/'+savedir, 'wb')
    pickle.dump(data,f)
    f.close()
