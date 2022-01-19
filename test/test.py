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


        #default
        with open("../data/features/icn/"+site,"rb") as feature_set:
            data = pickle.load(feature_set)
            #[site][feature] >> [feature][site]
            
            for i in range(len(data)):
                if(i==50):
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
        
    data=[]
    for i in range(3):
        data.append(Feature_data[i]["all"])
    data = np.array(data)

    fd = np.array(Feature_data[0]["all"][::]).reshape(-1,)
    #print(data.shape)


    fig = plt.figure()
    ax1 = fig.add_subplot(111,xlabel="feature(x)",ylabel="feature(y)")
    ax1.scatter(data[0][:50],data[1][:50],label="feature data",marker="x", alpha=0.4)
    
    """
    weight=[1]*len(fd)
    kdex = gaussian_kde(fd,bw_method="silverman",weights=weight)
    bwx = math.sqrt(kdex.covariance[0,0])
    fcx = kdex.factor
    xticks = np.linspace(fd.min()-bwx*4, fd.max()+bwx*4, 10000)
    print(bwx,fcx)
    estimatex = kdex(xticks)
    ax1.plot(xticks,estimatex,label="Kernel")
    
    
    weight[50:] = [0]*len(weight[50:])
    kde = gaussian_kde(fd,bw_method="silverman",weights=weight)
    bw = math.sqrt(kde.covariance[0,0])
    fc = kde.factor
    print(bw,fc)
    fc = bwx/(bw/fc)
    kde = gaussian_kde(fd,bw_method=fc,weights=weight)
    bw = math.sqrt(kde.covariance[0,0])
    fc = kde.factor
    print(bw,fc)
    ticks = np.linspace(fd.min()-bw*4, fd.max()+bw*4, 10000)
    estimate = kde(ticks)
    #ax1.plot(ticks,estimate,label="Kernel(Site0)",color="g")
    """
    weight = np.ones(500)
    #print(weight)
    #weight[50:] = 0
    kde = gaussian_kde(data,bw_method="silverman",weights=weight)
    bw = np.sqrt(kde.covariance)
    #print(kde.evaluate([100000,400000,600000]))
    print(bw)
    xticks = np.linspace(data[0].min(), data[0].max(), 100)
    yticks = np.linspace(data[1].min(), data[1].max(), 100)
    xx,yy = np.meshgrid(xticks,yticks)
    mesh = np.vstack([xx.ravel(),yy.ravel()])
    #z = kde(mesh)
    #print(mesh)
    #data = [fd,sd]
    
    
    
    
    #ax2 = fig.add_subplot(111,xlabel="fd",ylabel="sd")
    
    #ax1.plot(yticks,estimatey)
    #ax1.scatter(estimatex,estimatey)
    #ax1.scatter(fd,sd)
    """
    ax1.contourf(xx,yy,z.reshape(len(yticks),len(xticks)),15,cmap="Blues", alpha=0.5)
    PCM=ax1.get_children()[2] #get the mappable, the 1st and the 2nd are the x and y axes
    plt.colorbar(PCM, ax=ax1,cmap="Blues",label="Probabirity density") 
    """

    """
    for site in sites:
        fd = np.array(Feature_data[key][site]).reshape(-1,)

        kdex = gaussian_kde(fd,bw_method=1)
        xticks = np.linspace(fd.min()-bwx*4, fd.max()+bwx*4, 10000)
        estimatex = kdex(xticks)
        
    """
    plt.legend()
    plt.savefig("../data/Kernel.png")
    plt.show()
    
    return(information_leakage)


if __name__ == "__main__":
    sites = []
    target = "wpf"


    with open("../data/sites",'r') as f1:
        site_list = f1.readlines()
        for site in site_list:
            s = site.split()
            if s[0] == "#":
                continue
            if (s[2] == target):
                sites.append(s[1])


    data = calc_kernel(sites,target)