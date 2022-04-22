import numpy as np
import csv
import pic_feature
import os
import shutil

def get_alldata(train_size,place):
    

    for loc in place:
        with open("../data/wsfsites",'r') as f:
            sites = f.readlines()
            size = {}
            time = {}
            for site in sites:
                s = site.split()
                if s[0] == "#":
                    continue
                path = "../data/dataset/origin/"+loc+"/"+s[1]+"/"

                datasize = []
                datatime = []

                for i in range(train_size):
                    if os.path.isfile(path+str(i)+".csv"):
                        Time,Size,IP = pic_feature.get_csv(path+str(i)+".csv")
                    elif os.path.isfile(path +str(i)+ ".pcap"):
                        Time,Size,IP = pic_feature.get_pcap(path+str(i)+".pcap")
                        print("pic pcap")
                    
                    Size = np.abs(Size)
                    datasize.append(sum(Size))
                    datatime.append(Time[-1]-Time[0])
                
                size[s[1]] = datasize
                time[s[1]] = datatime

            size_list = np.array(list(size.values())).ravel()
            return(size,time,size_list)

def remove(train_size,place,size,time):

    for loc in place:
        with open("../data/wsfsites",'r') as f:
            sites = f.readlines()
            for site in sites:
                s = site.split()
                if s[0] == "#":
                    continue

                origin_path = "../data/dataset/origin/"+loc+"/"+s[1]+"/"
                copy_path = "../data/dataset/processed/"+loc+"/"+s[1]
                if os.path.exists(copy_path):
                    shutil.rmtree(copy_path)
                os.makedirs(copy_path)

                size75, size25 = np.percentile(size[s[1]], [75 ,25])
                sizelow = size25 - 1.5*(size75-size25)
                sizehigh = size75 + 1.5*(size75-size25)
                print("size low,high",sizehigh,sizelow)

                time75, time25 = np.percentile(time[s[1]], [75 ,25])
                timelow = time25 - 1.5*(time75-time25)
                timehigh = time75 + 1.5*(time75-time25)
                print("time low,high",timehigh,timelow)
                
                skip=0
                for i in range(train_size):
                    if not os.path.isfile(origin_path +str(i)+ ".csv"):
                        skip+=1
                        continue
                    if (size[s[1]][i-skip] < sizelow or size[s[1]][i-skip] > sizehigh):
                        print("remove for size",origin_path,i,size[s[1]][i-skip])
                    elif(time[s[1]][i-skip] < timelow or time[s[1]][i-skip] > timehigh):
                        print("remove for time",origin_path,i,time[s[1]][i-skip])
                    else:
                        shutil.copy(origin_path+str(i)+".csv",copy_path)
              



if __name__ == "__main__":
    train_size=200
    place = ["wsfodins"]

    size,time,size_list = get_alldata(train_size,place)

    remove(train_size,place,size,time)
