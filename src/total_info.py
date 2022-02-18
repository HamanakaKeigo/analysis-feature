
import numpy as np
import matplotlib.pyplot as plt
import pickle
import matlab
import math
import sys 
import csv

sys.path.append('../')
from scipy import integrate
from my_scipy.stats import gaussian_kde
#print(sys.path)


def test_1d(Feature_data=[]):
    
    #1次元データ
    data = []
    data.extend(Feature_data[0]["Amazon1"])
    data.extend(Feature_data[0]["Amazon2"])
    data1 = Feature_data[0]["Amazon1"]
    data2 = Feature_data[0]["Amazon2"]
    minx = min(data) - 50000
    maxx = max(data) + 50000

    xticks = np.linspace(minx, maxx, 100)

    #グラフの準備
    fig = plt.figure()
    ax1 = fig.add_subplot(111,xlabel="data",ylabel="rate")

    kde = gaussian_kde(data,bw_method="silverman")
    z = kde(xticks)
    ax1.plot(xticks,z,color="k")
    integ = kde.integrate_box_1d(minx,maxx)
    print(integ)

    kde1 = gaussian_kde(data1,cov_inv=[kde.inv_cov,kde.covariance])
    z1 = kde1(xticks)
    ax1.plot(xticks,z1)
    ax1.scatter(data1,[0]*len(data1))

    kde2 = gaussian_kde(data2,cov_inv=[kde.inv_cov,kde.covariance])
    z2 = kde2(xticks)
    ax1.plot(xticks,z2)
    ax1.scatter(data2,[0]*len(data2))

    integral = lambda x: kde(x) * (( kde1(x)*len(data1) )/( kde(x)*len(data) )) * np.log2(( kde1(x)*len(data1) )/( kde(x)*len(data) ))
    val, err = integrate.quad(integral,minx,maxx)
    print(val)
    print(err)

    plt.show()

def test_multi(Feature_data=[]):
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
    minx = min(data[0])-(max(data[0])-min(data[0]))/2
    miny = min(data[1])-(max(data[1])-min(data[1]))/2
    maxx = max(data[0])+(max(data[0])-min(data[0]))/2
    maxy = max(data[1])+(max(data[1])-min(data[1]))/2
    min_box = [minx,miny]
    max_box = [maxx,maxy]

    xticks = np.linspace(minx, maxx, 100)
    yticks = np.linspace(miny, maxy, 100)
    xx,yy = np.meshgrid(xticks,yticks)
    mesh = np.vstack([xx.ravel(),yy.ravel()])

    #グラフの準備
    fig = plt.figure()
    ax1 = fig.add_subplot(111,xlabel="fd",ylabel="sd")

    kde = gaussian_kde(data)
    integ = kde.integrate_box(min_box,max_box)
    print(integ)
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
    ax1.contourf(xx,yy,Z2,cmap="Greens", alpha=0.5)

    ax1.scatter(data2[0],data2[1])
    ax1.scatter(data1[0],data1[1])

    diff = z*len(data[0]) - (z1*len(data1[0])+z2*len(data2[0]))
    print(max(diff/(z*len(data[0]))))
    print(min(diff/(z*len(data[0]))))
    print(diff/z)
    #print(min(z1))


    plt.show()

def kde_1d(Feature_data=[],sites=None,id=0):
    
    #1次元データ
    data = Feature_data[id]["all"]

    #グラフの準備
    fig = plt.figure()
    ax1 = fig.add_subplot(111,xlabel="fd",ylabel="sd")

    kde = gaussian_kde(data,bw_method="silverman")
    minx = min(data) - (max(data)-min(data))/2
    maxx = max(data) + (max(data)-min(data))/2
    xticks = np.linspace(minx, maxx, 1000)
    
    z = kde(xticks)
    ax1.plot(xticks,z*len(data),color="k")
    
    

    Hcf = 0
    Hc = 0
    for site in sites:
        data1 = Feature_data[id][site]
        if len(data1)==0:
            print("break")
            break

        rate = len(data1) / len(data)
        Hc += rate*math.log2(rate)
        kde1 = gaussian_kde(data1,cov_inv=[kde.inv_cov,kde.covariance])
        z1 = kde1(xticks)
        ax1.plot(xticks,z1*len(data1))
        ax1.scatter(data1,[0]*len(data1))
        
        #minx = min(data1) - (max(data1)-min(data1))
        #maxx = max(data1) + (max(data1)-min(data1))

        integral = lambda x: kde(x) * (( kde1(x)*len(data1) )/( kde(x)*len(data)  )) * np.log2(( kde1(x)*len(data1) )/( kde(x)*len(data) )) if (kde(x)>0 and kde1(x)>0) else 0
        val, err = integrate.quad(integral,minx,maxx)
        
        if np.isnan(val):
            print(site)
            fig = plt.figure()
            ax1 = fig.add_subplot(111,xlabel="fd",ylabel="sd")

            xticks = np.linspace(minx, maxx, 100)
            z = kde(xticks)
            ax1.plot(xticks,z*len(data),color="k")
            z1 = kde1(xticks)
            ax1.plot(xticks,z1*len(data1))
            plt.show()
            continue
        #print(site," val:",val)

            
        #print("val = ",val)
        Hcf += val
        #print(val)
        #print(err)

    fig.savefig("../data/plot/kernel/total/"+str(id+1)+".png")


    print(id,"th mutual info = ",Hcf - Hc)
    return([Hcf-Hc])

    #plt.show()

def kde_multi(Feature_data=[],sites=None):
    #2次元データ
    dim = 2 #(参照する変数の数)
    data = []
    min_box = []
    max_box = []
    box = []
    for i in range(dim):
        d = Feature_data[i]["all"]
        data.append(d)
        
        minx = min(d) - (max(d)-min(d))/2
        min_box.append(minx)
        maxx = max(d) + (max(d)-min(d))/2
        max_box.append(maxx)
        box.append([minx,maxx])
    print(box)
    kde = gaussian_kde(data)

    

    #グラフ
    """xticks = np.linspace(min_box[0], max_box[0], 1000)
    yticks = np.linspace(min_box[1], max_box[1], 1000)
    xx,yy = np.meshgrid(xticks,yticks)
    mesh = np.vstack([xx.ravel(),yy.ravel()])

    fig = plt.figure()
    ax1 = fig.add_subplot(111,xlabel="fd",ylabel="sd")

    z = kde(mesh)
    Z = z.reshape(len(yticks),len(xticks))
    ax1.contourf(xx,yy,Z,cmap="Greens")"""

    
    n = len(data[0])
    
    Hcf = 0
    Hc = 0
    for i,site in enumerate(sites):
        data1 = []
        for j in range(dim):
            d1 = Feature_data[j][site]
            data1.append(d1)
        rate = len(data1[0]) / len(data[0])
        n1 = len(data1[0])
        
        Hc += rate*math.log2(rate)

        kde1 = gaussian_kde(data1,cov_inv=[kde.inv_cov,kde.covariance])
        integral = lambda *x: kde(x) * ( (kde1(x)*n1) / (kde(x)*n) ) * np.log2(( kde1(x)*n1)/( kde(x)*n )) if (kde(x)>0 and kde1(x)>0) else 0
        #integral = lambda x,y:kde1([x,y])
        val, err = integrate.nquad(integral,box)
        print(val)
        Hcf += val
        """z1 = kde1(mesh)
        Z1 = z1.reshape(len(yticks),len(xticks))
        ax1.contourf(xx,yy,Z1,cmap="Greens", alpha=0.5)
        ax1.scatter(data1[0],data1[1])"""

    #range(2): mtual = 1.3904793915869877
    print("mutual :",Hcf - Hc)
    plt.show()

    return ([Hcf - Hc])

def calc_info(sites=[],target=""):

    information_leakage=[]
    Feature_data = []


    for i,site in enumerate(sites):
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


    #test_multi(Feature_data)
    info=[]
    #for i in range (len(Feature_data)):
        #info.append(kde_1d(Feature_data,sites,i))
    info.append(kde_multi(Feature_data,sites))
    #test_1d(Feature_data)
    
    return(info)


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

    data = calc_info(sites,target)
    with open("../data/info.csv","w") as f:
        writer = csv.writer(f)
        writer.writerows(data)
