import numpy as np
import matplotlib.pyplot as plt
import pickle
import matlab
import math
import scipy.io
import os

import calc_info

def plot_trans(sites,place,savename):

    Feature_data=[]
    graph = {}

    for loc in place:
        for site in sites:
            #default
            plot = []
            with open("../data/features/"+loc+"/"+site,"rb") as feature_set:
                data = pickle.load(feature_set)
                #[site][feature] >> [feature][site]
                
                for i in range(len(data)):
                    if(i==10):
                        break
                    #plot.append(data[i][2943:2993]) #inCUMUL
                    plot.append(data[i][2993:3043]) #outCUMUL
                    #plot.append(data[i][3093:3143]) #CUMUL
                    #plot.append(data[i][3143:3193]) #CDN
                    #plot.append(data[i][3193:3243]) #Time
                    #plot.append(data[i][3293:3343]) #inTime
                    
            graph[site]=plot

    
    plot = []
    with open("../data/features/wsfodins/google","rb") as feature_set:
        data = pickle.load(feature_set)
        #[site][feature] >> [feature][site]
        
        for i in range(len(data)):
            if(i==10):
                break
            #plot.append(data[i][2943:2993]) #inCUMUL
            plot.append(data[i][2993:3043]) #outCUMUL
            #plot.append(data[i][3093:3143]) #CUMUL
            #plot.append(data[i][3143:3193]) #CDN
            #plot.append(data[i][3193:3243]) #Time
    graph["google"]=plot
    
    #print(graph["Amazon_audio1"])

    show_fig(graph,sites)
    return

    
def show_fig(graph,sites):
    fig = plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(8)
    ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
    color=["b","g","r","c","m","y","k","w"]
    style=["solid","dashed","dashdot","dotted"]
    c=0
    for i,site in enumerate(sites):
        if i%20 != 0:
            continue
        print(site)
        for j in range(len(graph[site])):
            ax1.plot(graph[site][j],linestyle=style[int(c/7)],color=color[c%7])
        c+=1
        #plt.show()
        #fig = plt.figure()
        #ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")
    #for j in range(len(graph["google"])):
    #    ax1.plot(graph["google"][j],linestyle=style[int(c/7)],color=color[c%7],)
    plt.xlabel("feature index",fontsize=17)
    plt.ylabel("size",fontsize=17)
    #plt.legend(fontsize=17)
    plt.tick_params(labelsize=17)
    plt.tight_layout()

    plt.show()




if __name__ == "__main__":
    sites = []
    place = ["odins"]
    savename = "odins"


    with open("../data/sites",'r') as f1:
        site_list = f1.readlines()
        for i,site in enumerate(site_list):
            s = site.split()
            if s[0] == "#":
                continue
            sites.append(s[1])

    if not os.path.isdir("../data/plot/trans"):
        os.makedirs("../data/plot/trans")
    data = plot_trans(sites,place,savename)
