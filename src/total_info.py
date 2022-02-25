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
    #print(data.shape)


    kde = gaussian_kde(data,bw_method="silverman")
    print(kde.covariance)

    #xx,yy = np.meshgrid(xticks,yticks)
    #mesh = np.vstack([xx.ravel(),yy.ravel()])
    plot = [0]*3
    plot[2] += 1
    z = kde.evaluate(plot)
    print(z)
    
    #fig = plt.figure()
    #ax1 = fig.add_subplot(111,xlabel="fd",ylabel="sd")
    #ax1.scatter(estimatex,estimatey)
    #ax1.scatter(fd,sd)
    #ax1.contourf(xx,yy,z.reshape(len(yticks),len(xticks)),cmap="Blues", alpha=0.5)
    #plt.show()
    
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