import tensorflow as tensor
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Dense, Flatten, BatchNormalization, Activation, Dropout, Conv1D
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
import random


def save_fig(history):
    
    graph = plt.figure()
    ax1 = graph.add_subplot(211)
    ax2 = graph.add_subplot(212)

    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    #ax1.title('Model accuracy')
    #ax1.ylabel('Accuracy')
    #ax1.xlabel('Epoch')
    ax1.legend(['Train_acc','Valid_acc'], loc='upper left')
    #ax1.show()

    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    #ax2.title('Model loss')
    #ax2.ylabel('Loss')
    #ax2.xlabel('Epoch')
    ax2.legend(['Train loss','valid loss'], loc='upper right')
    #ax2.show()
    plt.savefig("../data/train_loss.png")
    print(max(history.history['val_accuracy']))

def normalization(inp):
    
    means = inp.mean()
    stds = np.std(inp)
    norm = (inp-means)/stds

    norm_data = np.array([means,stds])
    #np.save("../data/norm_data",norm_data)
    print("norm =" + str(norm_data))

    return norm,norm_data

def standardization(inp):
    a = np.min(inp)
    b = np.max(inp)

    data = (inp-a)/(b-a)

    return data,a,b



def build_cumul(train_set,valid_set,page=0):
    ###build neural net###

    xshape = 50
    yshape = 50
    
    input1 = Input(shape=(xshape,)) 
    input2 = Input(shape=(yshape,))

    x1 = Dense(64,activation='relu')(input1)
    x1 = Model(inputs=input1, outputs=x1)
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
    
    pos,neg,tr_lab = make_cumul_input(train_set)
    pos.reshape(-1,xshape)
    neg.reshape(-1,yshape)
    tr_lab.reshape(-1,len(tr_lab))
    pos,p_norm = normalization(pos)
    neg,n_norm = normalization(neg)


    val_pos,val_neg,val_lab = make_cumul_input(valid_set)
    val_pos.reshape(-1,xshape)
    val_neg.reshape(-1,yshape)
    val_lab.reshape(-1,len(val_lab))
    val_pos = (val_pos-p_norm[0])/p_norm[1]
    val_neg = (val_neg-n_norm[0])/n_norm[1]


    print("ts : "+str(pos.shape)+" tf : "+str(neg.shape)+" tl : "+str(tr_lab.shape))
    print("ts : "+str(len(pos))+" tf : "+str(len(neg))+" tl : "+str(len(tr_lab)))
    history=model.fit([pos,neg], tr_lab, epochs=5000,validation_data=([val_pos,val_neg],val_lab))
    model.save("../data/trained_model")

    save_fig(history)
    return model,p_norm,n_norm


def build_neural(train_set,train_label,valid_set,valid_label,sites):
    ###build neural net###

    input_shape = len(train_set[0])
    print(input_shape)
    inputs = Input(shape=(input_shape,)) 


    #x = Dense(16,activation="relu")(inputs)
    #x = BatchNormalization()(x)
    #x = Activation("relu")(x)
    #x = Dropout(0.2)(x)

    y = Dense(64,activation="relu")(inputs)
    y = Dense(128,activation="relu")(y)
    #y = Dropout(0.2)(y)
    y = Dense(64,activation="relu")(y)

    y = Dense(len(sites),activation="softmax")(y)
    

    model = Model(inputs=inputs,outputs=y)
    model.compile(optimizer='adam', 
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
    model.summary()

    tr_in,norm = normalization(train_set)
    val_in = (valid_set-norm[0])/norm[1]

    #tr_in,a,b = standardization(train_set)
    #val_in = (valid_set-a)/(b-a)
    #print(tr_in[:5])

    #print("ts : "+str(pos.shape)+" tf : "+str(neg.shape)+" tl : "+str(cl.shape))
    #print("ts : "+str(len(pos))+" tf : "+str(len(neg))+" tl : "+str(len(cl)))
    history=model.fit(tr_in, train_label, epochs=1000,validation_data=(val_in,valid_label))
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



def get_data(loc,siteslist):

    train_data = []
    train_label= []
    valid_data = []
    valid_label= []
    sites = []
    page=0

    for site in siteslist:
        sites.append(site)
        #data for train
        with open("../data/features/"+loc+"/"+site,"rb") as feature_set:
            data = pickle.load(feature_set)
            for d in data:
                train_data.append(d)
                train_label.append(page)

        #data for valid
        with open("../data/features/"+loc+"/"+site+"_valid","rb") as feature_set:
            data = pickle.load(feature_set)
            for d in data:
                valid_data.append(d)
                valid_label.append(page)
        page+=1
    return train_data,train_label,valid_data,valid_label,sites
    
        #print(data_set[0])


def select_feature(train_data,valid_data):
    target = "outcumul"
    train_set=[]
    valid_set=[]

    if target=="cumul":
        for d in train_data:
            train_set.append(d[3093:3143])
        for d in valid_data:
            valid_set.append(d[3093:3143])
    if target=="timecumul":
        for d in train_data:
            train_set.append(d[3193:3243])
        for d in valid_data:
            valid_set.append(d[3193:3243])
    if target=="incumul":
        for d in train_data:
            train_set.append(d[2943:2993])
        for d in valid_data:
            valid_set.append(d[2943:2993])
    if target=="outcumul":
        for d in train_data:
            train_set.append(d[2993:3043])
        for d in valid_data:
            valid_set.append(d[2993:3043])
    if target=="cdnburst":
        for d in train_data:
            train_set.append(d[3043:3093])
        for d in valid_data:
            valid_set.append(d[3043:3093])


    return train_set,valid_set

if __name__ == "__main__":

    siteslist=[]

    with open("../data/sites",'r') as f1:
        site_list = f1.readlines()
        for i,site in enumerate(site_list):
            s = site.split()
            if s[0] == "#":
                continue
            siteslist.append(s[1])

    loc = "odins"
    #place = ["odins","wsfodins"]
    #for loc in place:
    train_data,train_label,valid_data,valid_label,sites = get_data(loc,siteslist)

    
    train_set,valid_set = select_feature(train_data,valid_data)
    train_set=np.array(train_set)
    valid_set=np.array(valid_set)
    train_label=np.array(train_label)
    valid_label=np.array(valid_label)
    #print(len(train_set[0]))
    model,norm = build_neural(train_set,train_label,valid_set,valid_label,sites)
    #model,norm = build_cumul(train_set,valid_set,page)
    #evaluate(valid_set,model,norm)
    