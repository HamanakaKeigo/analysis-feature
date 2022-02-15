from sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib.pyplot as plt
import pickle
import matlab
import math
import sys 
sys.path.append('../')
from my_scipy.stats import gaussian_kde
from my_scipy.stats import norm
#print(sys.path)

def kde_1d(Feature_data=[]):
    #1次元データ
    data = []
    data.extend(Feature_data[0]["Amazon1"])
    data.extend(Feature_data[0]["Amazon2"])
    data1 = Feature_data[0]["Amazon1"]
    data2 = Feature_data[0]["Amazon2"]

    xticks = np.linspace(min(data), max(data), 100)

    #グラフの準備
    fig = plt.figure()
    ax1 = fig.add_subplot(111,xlabel="fd",ylabel="sd")

    kde = gaussian_kde(data,bw_method="silverman")
    z = kde(xticks)
    ax1.plot(xticks,z*len(data),color="k")
    #ax1.scatter(data,[0]*len(data))

    kde1 = gaussian_kde(data1,cov_inv=[kde.inv_cov,kde.covariance])
    z1 = kde1(xticks)
    ax1.plot(xticks,z1*len(data1))
    ax1.scatter(data1,[0]*len(data1))

    kde2 = gaussian_kde(data2,cov_inv=[kde.inv_cov,kde.covariance])
    z2 = kde2(xticks)
    ax1.plot(xticks,z2*len(data2))
    ax1.scatter(data2,[0]*len(data2))

    plt.show()

def kde_multi(Feature_data=[]):
    #2次元データ
    datax = []
    datax.extend(Feature_data[0]["Amazon1"])
    datax.extend(Feature_data[0]["Amazon2"])
    datay = []
    datay.extend(Feature_data[1]["Amazon1"])
    datay.extend(Feature_data[1]["Amazon2"])
    data = [datax,datay]
    data1 = []
    data1.append(Feature_data[0]["Amazon1"])
    data1.append(Feature_data[1]["Amazon1"])
    data2 = []
    data2.append(Feature_data[0]["Amazon2"])
    data2.append(Feature_data[1]["Amazon2"])

    xticks = np.linspace(min(data[0]), max(data[0]), 100)
    yticks = np.linspace(min(data[1]), max(data[1]), 100)
    xx,yy = np.meshgrid(xticks,yticks)
    mesh = np.vstack([xx.ravel(),yy.ravel()])

    #グラフの準備
    fig = plt.figure()
    ax1 = fig.add_subplot(111,xlabel="fd",ylabel="sd")

    kde = gaussian_kde(data)
    z = kde(mesh)
    Z = z.reshape(len(yticks),len(xticks))
    ax1.contourf(xx,yy,Z,cmap="Greens", alpha=0.5)

    kde1 = gaussian_kde(data1,cov_inv=[kde.inv_cov,kde.covariance])
    z1 = kde1(mesh)
    Z1 = z1.reshape(len(yticks),len(xticks))
    ax1.contourf(xx,yy,Z1,cmap="Greens", alpha=0.5)

    kde2 = gaussian_kde(data2,cov_inv=[kde.inv_cov,kde.covariance])
    z2 = kde2(mesh)
    Z2 = z2.reshape(len(yticks),len(xticks))
    ax1.contourf(xx,yy,Z2,cmap="Reds", alpha=0.5)

    ax1.scatter(data2[0],data2[1])
    ax1.scatter(data1[0],data1[1])

    diff = z*len(data[0]) - (z1*len(data1[0])+z2*len(data2[0]))
    print(max(diff))
    print(min(diff))
    print(diff)


    plt.show()

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


    #kde_multi(Feature_data)
    kde_1d(Feature_data)

    
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