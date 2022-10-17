import json
import csv
import matplotlib.pyplot as plt
import os




def get_har(filename):
    with open(filename) as f:
        data = json.load(f)
    maxb=0

    startTime = float(data["log"]["entries"][0]["startedDateTime"][-7:-1]) + float(data["log"]["entries"][0]["startedDateTime"][-10:-8])*60
    http = []
    for i in range(len(data["log"]["entries"])):
        har = {}
        if "serverIPAddress" in data["log"]["entries"][i]:
            har["IP"]   = data["log"]["entries"][i]["serverIPAddress"]
        else:
            har["IP"]   = "cache"
        har["URL"]      = data["log"]["entries"][i]["request"]["url"]
        har["Type"]     = data["log"]["entries"][i]["response"]["content"]["mimeType"]
        har["inSize"]   = data["log"]["entries"][i]["response"]["headersSize"] + data["log"]["entries"][i]["response"]["bodySize"]
        har["outSize"]  = data["log"]["entries"][i]["request"]["headersSize"] + data["log"]["entries"][i]["request"]["bodySize"]
        har["Time"]     = data["log"]["entries"][i]["timings"]
        har["dilay"]    = round(float(data["log"]["entries"][i]["startedDateTime"][-7:-1]) + float(data["log"]["entries"][i]["startedDateTime"][-10:-8])*60 - startTime,4)
        har["StartTime"]= data["log"]["entries"][i]["startedDateTime"]
        har["allTime"]  = data["log"]["entries"][i]["time"]
        http.append(har)

    #print(data["log"]["entries"][3]["request"]["headersSize"])
    #print(data["log"]["entries"][3]["request"]["bodySize"])
    #print(http[0]["URL"])
    return http

def count_request(data):
    count=0
    for i in range(len(data)):
        if data[i]["inSize"]>0:
            count+=1
    return count

def count_filetype(data):
    types={}
    for i in range(len(data)):
        if data[i]["inSize"]>0:
            if data[i]["Type"] in types.keys():
                types[data[i]["Type"]] += 1
            else:
                types[data[i]["Type"]]=1
    
    return types

def print_name(data):
    names = {}
    for i in range(len(data)):
        if data[i]["inSize"]>0:
            if data[i]["URL"].split("?")[0].split("/")[-1] not in names.keys():
                names[data[i]["URL"].split("?")[0].split("/")[-1]] = 1
            else:
                names[data[i]["URL"].split("?")[0].split("/")[-1]] += 1

    for i in names.keys():
        print(names[i],i)

def total_insize(data):
    total = 0

    for i in range(len(data)):
        if data[i]["inSize"]>0:
            total+=data[i]["inSize"]

    return total

def total_outsize(data):
    total = 0

    for i in range(len(data)):
        if data[i]["outSize"]>0:
            total+=data[i]["outSize"]

    return total


def split_filetype(data):
    file = {}
    for i in range(len(data)):
        if data[i]["Type"] not in file.keys():
            file[data[i]["Type"]] = [i,[data[i]]]
        else:
            file[data[i]["Type"]].append([i,data[i]])
    return file

def return_type(data):
    t = data["Type"]
    if(t.split("/")[0] == "image"):
        t = "image"
    elif(t.split("/")[-1] == "css"):
        t = "css"
    elif(t.split("/")[-1].split("-")[-1] == "javascript"):
        t="js"
    elif(t.split("/")[-1] == "html"):
        t="html"
    elif(t.split("/")[-1] == "font-woff2"):
        t="font"
    elif(t.split("/")[-1] == "json"):
        #print(t,data["URL"])
        t="json"
    else:
        #print(t,data["URL"])
        t="others"

    return t

def print_bigrequest(data):
    s = []
    for d in data:
        s.append(d["inSize"])
    s.sort(reverse=True)
    print(s[:10])

    return

def graph_timing(data,site,num):
    colorlist = ["r", "g", "b", "c", "m", "y", "k", "lightgray", "lightsalmon", "darkorange", "greenyellow", "aquamarine","navy","violet","pink"]
    filetype=["html","js","css","image","font","json","others"]
    IP=[]
    fig = plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(8)
    ax1 = fig.add_subplot(111,xlabel="index",ylabel="rate")

    sc={}
    adr={}
    for i in range(len(data)):
        if data[i]["IP"]=="cache":
            continue
        t = return_type(data[i])
        ip= data[i]["IP"]

        if ip not in IP:
            IP.append(ip)
            adr[ip]=[[],[]]
        if t not in sc.keys():
            sc[t]=[[],[]]
        sc[t][0].extend([data[i]["dilay"]+round(data[i]["Time"]["blocked"]/1000,3),data[i]["dilay"]+data[i]["allTime"]/1000])
        sc[t][1].extend([i]*2)
        adr[ip][0].extend([data[i]["dilay"]+round(data[i]["Time"]["blocked"]/1000,3),data[i]["dilay"]+data[i]["allTime"]/1000])
        adr[ip][1].extend([i]*2)

        ax1.plot([data[i]["dilay"]+round(data[i]["Time"]["blocked"]/1000,3),data[i]["dilay"]+(data[i]["allTime"]/1000)],[i]*2,color=colorlist[filetype.index(t)])
        #ax1.plot([data[i]["dilay"]+round(data[i]["Time"]["blocked"]/1000,3),data[i]["dilay"]+(data[i]["allTime"]/1000)],[i]*2,color=colorlist[IP.index(data[i]["IP"])])


    for t in sc.keys():
    #for ip in IP:
        ax1.scatter(sc[t][0],sc[t][1],color=colorlist[filetype.index(t)],label=t)
        #ax1.scatter(adr[ip][0],adr[ip][1],color=colorlist[IP.index(ip)],label=ip)
    plt.legend()
    #plt.show()
    fig.savefig("../data/plot/httptiming/"+site+"/"+str(num))
    plt.clf()
    plt.close()

if __name__ == "__main__":
    sites = ["Amazon_home1","Amazon_watch4","Amazon_toy1","Amazon_soft2","Amazon_audio2"]#,"google","youtube","qq"
    datas = {}
    files=[]
    for site in sites:
        print("----------",site,"----------")
        if not os.path.isdir("../data/plot/httptiming/"+site):
            os.makedirs("../data/plot/httptiming/"+site)
        datas[site] = []
        
        filecount=1
        
        for i in range(filecount):
            files.append([])
            datas[site].append(get_har("../data/dataset/har/"+site+"/"+str(i)+".har"))
            print(site,count_request(datas[site][-1]),"->",count_filetype(datas[site][-1]))
            print("time:",datas[site][-1][-1]["dilay"],"r/t:",count_request(datas[site][-1])/datas[site][-1][-1]["dilay"])
            #print(site,"inSize:",total_insize(datas[site][-1]))
            #print(site,"outSize:",total_outsize(datas[site][-1]))
            print_bigrequest(datas[site][-1])
            
            #print(files)
            for j in range(len(datas[site][-1])):
                if(len(files[-1]) < 10):
                #if float(datas[site][-1][j]["dilay"])<0.5:
                    #datas[site][-1][j]["allTime"],
                    files[-1].append(datas[site][-1][j]["URL"])
                #files[-1].append(datas[site][-1][j]["URL"].split("?")[0].split("/")[-1])
                    print(j,datas[site][-1][j]["URL"].split("?")[0].split("/")[-1])

            #graph_timing(datas[site][i],site,i)

    print("^^^^^^^^^^^^")
    ind=[]
    d=list(set(files[-1]))
    count=0
    for i,x in enumerate(d):
        flag=0
        for j in range(len(files)):
            if x not in files[j]:
                flag=1
                break

        if flag==0:
            ind.append(files[-1].index(x))
            print(files[-1].index(x),x)
            count+=1
    print("count = ",count,"/",len(files[-1]))
    ind.sort()
    print(ind)

    data = datas["Amazon_home1"][0]
    #print_name(data)
    


