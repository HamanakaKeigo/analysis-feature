import numpy as np
import csv
import pic_feature
import os
import shutil

def get_alldata(train_size,place):
    

    for loc in place:
        with open("../data/sites",'r') as f:
            sites = f.readlines()
            size = {}
            time = {}
            for site in sites:
                s = site.split()
                if s[0] == "#":
                    continue
                path = "../data/dataset/origin/"+loc+"/"+s[1]+"/"

                datasize = [0]*train_size
                datatime = [0]*train_size

                for i in range(train_size):
                    if not os.path.isfile(path +str(i)+ ".pcap"):
                        continue
                    if os.path.isfile(path+str(i)+".csv"):
                        Time,Size,IP = pic_feature.get_csv(path+str(i)+".csv")
                    else:
                        Time,Size,IP = pic_feature.get_pcap(path+str(i)+".pcap")
                        print("pic pcap")
                    
                    Size = np.abs(Size)
                    datasize[i] = sum(Size)
                    datatime[i] = Time[-1]-Time[0]
                
                size[s[1]] = datasize
                time[s[1]] = datatime

            size_list = np.array(list(size.values())).ravel()
            return(size,time,size_list)

def remove(train_size,place,size,time):

    for loc in place:
        with open("../data/sites",'r') as f:
            sites = f.readlines()
            for site in sites:
                s = site.split()
                if s[0] == "#":
                    continue

                origin_path = "../data/dataset/origin/"+loc+"/"+s[1]+"/"
                copy_path = "../data/dataset/processed/"+loc+"/"+s[1]
                if os.path.exists(copy_path):
                    os.remove(copy_path)
                os.makedirs(copy_path)

                size75, size25 = np.percentile(size[s[1]], [75 ,25])
                sizelow = size25 - 1.5*(size75-size25)
                sizehigh = size75 + 1.5*(size75-size25)
                print("size low,high",sizehigh,sizelow)

                time75, time25 = np.percentile(time[s[1]], [75 ,25])
                timelow = time25 - 1.5*(time75-time25)
                timehigh = time75 + 1.5*(time75-time25)
                print("time low,high",timehigh,timelow)
                
                for i in range(train_size):
                    if (size[s[1]][i] < sizelow or size[s[1]][i] > sizehigh):
                        print("remove for size",origin_path,i,size[s[1]][i])
                    elif(time[s[1]][i] < timelow or time[s[1]][i] > timehigh):
                        print("remove for time",origin_path,i,time[s[1]][i])
                    else:
                        shutil.copy(origin_path+str(i)+".csv",copy_path)
              



if __name__ == "__main__":
    train_size=150
    place = ["icn"]

    size,time,size_list = get_alldata(train_size,place)

    remove(train_size,place,size,time)
