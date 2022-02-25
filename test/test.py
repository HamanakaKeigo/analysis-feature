from scipy.stats import gaussian_kde
from scipy.stats import norm
<<<<<<< HEAD
=======
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
>>>>>>> main
import numpy as np
import matplotlib.pyplot as plt
import pickle
import matlab
from scipy import integrate
from scipy.integrate import cumtrapz
<<<<<<< HEAD
import math
import scipy.io
import csv


information_leakage=[]
Feature_data = []


def calc(filename):
    file = open(filename, 'rb')
    datas = pickle.load(file)

    times = []
    sizes = []
    for data in datas:
        times.append(data[1])
        sizes.append(data[0])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(times,sizes)
    #plt.show()

    dataset = [times,sizes]
    kde = gaussian_kde(dataset)

    x = np.linspace(min(times)-1, max(times)+1, 50)
    y = np.linspace(min(sizes)-10000, max(sizes)+10000, 50)
    xx,yy = np.meshgrid(x,y)
    meshdata = np.vstack([xx.ravel(),yy.ravel()])
    z = kde.evaluate(meshdata)

    ax.contourf(xx,yy,z.reshape(len(y),len(x)),alpha=0.5)
    plt.show()
=======
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
>>>>>>> main

    fd = np.array(Feature_data[0]["all"][::]).reshape(-1,)
    #print(data.shape)

<<<<<<< HEAD
    print("calc")
=======
>>>>>>> main

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

<<<<<<< HEAD
def pic(filename):
    #data = pyshark.FileCapture(filename)
    file = open(filename, 'rb')
    data = pickle.load(file)

    all_size = 0
    first_time=0
    
    for i,packet in enumerate(data):
        all_size += int(packet[2])

    all_time = data[-1][1] - data[0][1]
    file.close()
    save=[all_size,all_time]
    return save



if __name__ == "__main__":

    datas = []
    c = False
    if(c):
        for i in range(10):
            filename = "../data/train/www.amazon.co.jp/"+str(i)+".pickle"
            datas.append(pic(filename))

        f = open('../data/plot/data', 'wb')
        pickle.dump(datas,f)
        f.close()
    
    calc('../data/plot/data')
=======
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
>>>>>>> main
