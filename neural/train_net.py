import tensorflow as tensor
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Dense, Flatten, BatchNormalization, Activation, Dropout, Conv1D
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

def make_feature(get=[]):
    
    fset=[]
    for g in get:
        fset.extend(g)

    """
    data = pickle.load(open("../data/plot/feature_info/mix/wpf","rb"))
    data = np.array(data)
    arg = np.argsort(data)
    fset = np.array(fset)[arg]
    fset = fset[::-1]
    feature = fset[:100]
    """

    feature = fset[50:100]
    #feature = get[-1]
    return feature

def make_input(data,input_shape):

    inp = []
    label = []
    for i in range(len(data)):
        inp.append(data[i][0])
        label.append(data[i][1])

    inp = np.array(inp)
    label = np.array(label)

    inp.reshape(-1,input_shape)
    label.reshape(-1,len(label))

    return inp,label

def make_cumul_input(data):
    pos = []
    neg = []
    label = []
    for i in range(len(data)):
        pos.append(data[i][0][:50])
        neg.append(data[i][0][50:])
        label.append(data[i][1])

    pos = np.array(pos)
    neg = np.array(neg)
    label = np.array(label)

    pos.reshape(-1,50)
    neg.reshape(-1,50)
    label.reshape(-1,len(label))

    return pos,neg,label

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


def build_neural(train_set,valid_set,page=0):
    ###build neural net###

    #print(train_set[0])
    input_shape = len(train_set[0][0])
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



    y = Dense(page,activation="softmax")(y)
    

    model = Model(inputs=inputs,outputs=y)
    model.compile(optimizer='adam', 
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
    model.summary()

    tr_in,tr_lab = make_input(train_set,input_shape)
    tr_in,norm = normalization(tr_in)

    val_in,val_lab = make_input(valid_set,input_shape)
    val_in = (val_in-norm[0])/norm[1]

    #print("ts : "+str(pos.shape)+" tf : "+str(neg.shape)+" tl : "+str(cl.shape))
    #print("ts : "+str(len(pos))+" tf : "+str(len(neg))+" tl : "+str(len(cl)))
    history=model.fit(tr_in, tr_lab, epochs=1000,validation_data=(val_in,val_lab))
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



def get_data(origin=True):

    #origin = True
    train_data = []
    valid_data = []


    train_size=100
    valid_size=40
    with open("../data/sites",'r') as f:
        sites = f.readlines()
        page=0
        for site in sites:

            s = site.split()
            if s[0] == "#":
                continue
            
            features=[]
            for i in range(train_size):
                if(i<25):
                    if not os.path.isfile("../data/train/yhome/"+s[1]+"/"+str(i)+".pcap"):
                        break
                    get = pic.get_features("../data/train/yhome/"+s[1]+"/"+str(i))
                elif(i<50):
                    if not os.path.isfile("../data/train/icn/"+s[1]+"/"+str(i-25)+".pcap"):
                        break
                    get = pic.get_features("../data/train/icn/"+s[1]+"/"+str(i-25))
                elif(i<75):
                    if not os.path.isfile("../data/train/odins/"+s[1]+"/"+str(i-50)+".pcap"):
                        break
                    get = pic.get_features("../data/train/odins/"+s[1]+"/"+str(i-50))
                else:
                    if not os.path.isfile("../data/train/lib/"+s[1]+"/"+str(i-75)+".pcap"):
                        break
                    get = pic.get_features("../data/train/lib/"+s[1]+"/"+str(i-75))
                train_set = []

                train_set.append(make_feature(get))
                train_set.append(np.array(page))
                train_data.append(np.array(train_set))

            for i in range(valid_size):
                if(i<10):
                    if not os.path.isfile("../data/train/yhome/"+s[1]+"/"+str(25+i)+".pcap"):
                        break
                    get = pic.get_features("../data/train/yhome/"+s[1]+"/"+str(25+i))
                elif(i<20):
                    if not os.path.isfile("../data/train/icn/"+s[1]+"/"+str(99-i)+".pcap"):
                        break
                    get = pic.get_features("../data/train/icn/"+s[1]+"/"+str(99-i))
                elif(i<30):
                    x = i-20
                    if not os.path.isfile("../data/train/odins/"+s[1]+"/"+str(99-x)+".pcap"):
                        break
                    get = pic.get_features("../data/train/odins/"+s[1]+"/"+str(99-x))
                else:
                    x=i-20
                    if not os.path.isfile("../data/train/lib/"+s[1]+"/"+str(39-x)+".pcap"):
                        break
                    get = pic.get_features("../data/train/lib/"+s[1]+"/"+str(39-x))
                valid_set = []

                valid_set.append(make_feature(get))
                valid_set.append(np.array(page))
                valid_data.append(np.array(valid_set))
            page+=1


    train_data=np.array(train_data)
    valid_data=np.array(valid_data)
    return train_data,valid_data,page
    
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

    
    #print(data_set)
    #train_set,valid_set = split_data(data_set)
    
    
    train_set,valid_set,page = get_data(True) #origin = True
    dataset=np.array([train_set,valid_set,page])
    np.save('../data/features/dataset',dataset)
    
    
    """
    data = np.load('../data/features/dataset.npy',allow_pickle=True)
    train_set = data[0]
    valid_set = data[1]
    page = data[2]
    """

    model,norm = build_neural(train_set,valid_set,page)
    #model,norm = build_cumul(train_set,valid_set,page)
    evaluate(valid_set,model,norm)
    