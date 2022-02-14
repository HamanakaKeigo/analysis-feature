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
import sys 
sys.path.append('../')
from my_scipy.stats import gaussian_kde
from my_scipy.stats import norm
#print(sys.path)


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
        
    #データ選択
    """
    data=[]
    for i in range(3):
        data.append(Feature_data[i]["all"])
    data = np.array(data)
    """
    data = Feature_data[0]["all"]
    data1 = Feature_data[0]["Amazon0"]

    xticks = np.linspace(min(data), max(data), 100)
    #xx,yy = np.meshgrid(xticks,yticks)
    #mesh = np.vstack([xx.ravel(),yy.ravel()])

    #グラフの準備
    fig = plt.figure()
    ax1 = fig.add_subplot(111,xlabel="fd",ylabel="sd")
    
    kde = gaussian_kde(data,bw_method="silverman")
    z = kde(xticks)
    ax1.plot(xticks,z*len(data),color="k")
    ax1.scatter(data,[0]*len(data))

    for site in sites:
        kde1 = gaussian_kde(Feature_data[0][site],cov_inv=[kde.inv_cov,kde.covariance])
        z1 = kde1(xticks)
        ax1.plot(xticks,z1*len(Feature_data[0][site]))
        #ax1.scatter(data1,[0]*len(data1))
    

    #ax1.scatter(estimatex,estimatey)
    #ax1.scatter(fd,sd)
    #ax1.contourf(xx,yy,z.reshape(len(yticks),len(xticks)),cmap="Blues", alpha=0.5)
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