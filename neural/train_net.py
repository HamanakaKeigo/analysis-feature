import tensorflow as tensor
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Dense, Flatten
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
sys.path.append('../src')
import pic_feature as pic
import random


def save_fig(history):
    
    graph = plt.figure()
    ax1 = graph.add_subplot(211)
    ax2 = graph.add_subplot(212)

    ax1.plot(history.history['accuracy'])
    #ax1.title('Model accuracy')
    #ax1.ylabel('Accuracy')
    #ax1.xlabel('Epoch')
    ax1.legend(['Train_acc'], loc='upper left')
    #ax1.show()

    ax2.plot(history.history['loss'])
    #ax2.title('Model loss')
    #ax2.ylabel('Loss')
    #ax2.xlabel('Epoch')
    ax2.legend(['Train loss'], loc='upper right')
    #ax2.show()
    plt.savefig("../data/train_loss.png")

def normalization(inp):
    
    means = inp.mean()
    stds = np.std(inp)
    norm = (inp-means)/stds

    """
    f_means = feature.mean()
    f_stds = np.std(feature)
    norm_feature = (feature-f_means)/f_stds
    """

    norm_data = np.array([means,stds])
    #np.save("../data/norm_data",norm_data)
    print("norm =" + str(norm_data))

    return norm,norm_data


def build_cumul(data_set=[],page=0):
    ###build neural net###

    #x = np.array(data_set)
    print(data_set.shape)

    xshape = len(data_set[0][0])
    yshape = len(data_set[0][1])
    print(xshape)
    
    input1 = Input(shape=(xshape,)) 
    input2 = Input(shape=(yshape,))

    #x1 = Flatten(input_shape=(xshape,))
    x1 = Dense(64,activation='relu')(input1)
    x1 = Model(inputs=input1, outputs=x1)
    #x2 = Flatten(input_shape=(yshape,))
    x2 = Dense(64,activation='relu')(input2)
    x2 = Model(inputs=input2, outputs=x2)

    combined = concatenate([x1.output,x2.output])

    y = Dense(128,activation="relu")(combined)
    #y = Dense(128,activation="relu")(y)
    #y = Dense(128,activation="relu")(y)
    y = Dense(page,activation="softmax")(y)
    #z = Dense(64,activation="softmax")(y)


    model = Model(inputs=[input1,input2],outputs=y)
    model.compile(optimizer='adam', 
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
    model.summary()
    
    pos =[]
    neg=[]
    cl=[]
    for i in range(len(data_set)):
        pos.append(data_set[i][0])
        neg.append(data_set[i][1])
        cl.append(data_set[i][2])

    pos = np.array(pos)
    neg = np.array(neg)
    cl = np.array(cl)

    pos.reshape(-1,xshape)
    neg.reshape(-1,yshape)
    cl.reshape(-1,len(cl))

    pos,norm = normalization(pos)
    neg,norm2=normalization(neg)
    norm=norm.extend(norm2)


    print("ts : "+str(pos.shape)+" tf : "+str(neg.shape)+" tl : "+str(cl.shape))
    print("ts : "+str(len(pos))+" tf : "+str(len(neg))+" tl : "+str(len(cl)))
    history=model.fit([pos,neg], cl, epochs=5000)
    model.save("../data/trained_model")

    save_fig(history)
    return model,norm


def build_neural(data_set=[],page=0):
    ###build neural net###

    input_shape = len(data_set[0][0])
    #print(input_shape)
    inputs = Input(shape=(input_shape,)) 


    x1 = Dense(64,activation='relu')(inputs)
    x1 = Model(inputs=inputs, outputs=x1)

    y = Dense(128,activation="relu")(x1.output)
    #y = Dense(128,activation="relu")(y)
    #y = Dense(128,activation="relu")(y)
    y = Dense(page,activation="softmax")(y)
    #z = Dense(64,activation="softmax")(y)


    model = Model(inputs=inputs,outputs=y)
    model.compile(optimizer='adam', 
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
    model.summary()
    
    inp =[]
    cl=[]
    for i in range(len(data_set)):
        inp.append(data_set[i][0])
        cl.append(data_set[i][1])

    inp = np.array(inp)
    cl = np.array(cl)

    inp.reshape(-1,input_shape)
    cl.reshape(-1,len(cl))


    inp,norm = normalization(inp)


    #print("ts : "+str(pos.shape)+" tf : "+str(neg.shape)+" tl : "+str(cl.shape))
    #print("ts : "+str(len(pos))+" tf : "+str(len(neg))+" tl : "+str(len(cl)))
    history=model.fit(inp, cl, epochs=50000)
    model.save("../data/trained_model")

    save_fig(history)
    return model,norm


def evaluate(data_set=[],model=0,norm=0):

    ########evaluate########
    #model.summary()

    input_shape = len(data_set[0][0])

    inp = []
    cl=[]
    for i in range(len(data_set)):
        inp.append(data_set[i][0])
        cl.append(data_set[i][1])


    inp = np.array(inp)
    cl = np.array(cl)

    inp.reshape(-1,input_shape)
    cl.reshape(-1,len(cl))
    #pos,neg,norm = normalization(pos,neg)

    inp = (inp-norm[0])/norm[1]
    pre = model.evaluate(inp,cl)
    print(pre)

def make_feature(get=[]):
    fset=[]
    """
    cumul = get[10][4:]
    fset.append(np.array(cumul[0:50]))
    fset.append(np.array(cumul[50:100]))
    """
    features = []
    for g in get:
        features.extend(g)
    fset = features[2943:]
    return fset

def get_data(origin=True):

    #origin = True
    data_set = []

    if origin:
        train_size=100
        with open("../data/sites",'r') as f:
            sites = f.readlines()
            page=0
            for site in sites:

                s = site.split()
                if s[0] == "#":
                    continue
                
                features=[]
                for i in range(train_size):
                    if not os.path.isfile("../data/train/"+s[1]+"/"+str(i)+".pcap"):
                        break
                    get = pic.get_features("../data/train/"+s[1]+"/"+str(i))
                    fset = []
                    """
                    cumul = get[10][4:]
                    fset.append(np.array(cumul[0:50]))
                    fset.append(np.array(cumul[50:100]))
                    """
                    fset.append(make_feature(get))
                    fset.append(np.array(page))
                    data_set.append(np.array(fset))
                page+=1
    else:
        page=0
        while(True):
            if not os.path.isdir("../data/amazon/Amazonjp"+str(page)):
                break
            for i in range(100):
                if not os.path.isfile("../data/amazon/Amazonjp"+str(page)+"/amazonjp"+str(page)+f'{i:02}'+".csv"):
                    break
                get = pic.get_csv("../data/amazon/Amazonjp"+str(page)+"/amazonjp"+str(page)+f'{i:02}'+".csv")
                
                fset = []
                fset.append(make_feature(get))
                fset.append(np.array(page))
                data_set.append(np.array(fset))
                #print(len(feature))
            page += 1

    data_set=np.array(data_set)
    return data_set,page
    
        #print(data_set[0])

def split_data(data_set=[]):

    valid_size = 50

    val_set     =[]

    for i in range(valid_size):
        x = random.randrange(len(data_set))

        val_set.append(data_set[x])
        np.delete(data_set,x)

    val_set=np.array(val_set)

    return data_set,val_set

if __name__ == "__main__":

    data_set,page = get_data(False) #origin = True
    #print(data_set)
    train_set,val_set = split_data(data_set)

    model,norm = build_neural(train_set,page)
    evaluate(data_set,model,norm)
    