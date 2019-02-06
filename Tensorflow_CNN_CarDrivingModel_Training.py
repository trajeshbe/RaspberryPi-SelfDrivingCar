# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 08:01:02 2018

@author: Rajesh
"""

import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm

from collections import Counter
#from random import shuffle
import pandas as pd



TRAIN_DIR='D:\Data\\Deep Learning\\selfdriving\\train'
TEST_DIR='D:\Data\\Deep Learning\\selfdriving\\test\\test6000'
IMAGE_SIZE=50
LR=1e-3

MODEL_NAME='cardriving-{}-{}v15_balanced1.model'.format(LR,'2conv-basic')

a = [1,0,0,0,0]
w = [0,1,0,0,0]
d = [0,0,1,0,0]
e = [0,0,0,1,0]
s = [0,0,0,0,1]



WIDTH = 160
HEIGHT = 120
#LR = 1e-3
LR=0.0001
EPOCHS = 10

def label_img(img):
    #cat.1.png [-1=png,-2=1,-3=cat]
    print('TRAIN_DIR',TRAIN_DIR)
    print(img.split('.')[-3])
    word_label=img.split('.')[-3]
    #Convert to one hot arrary of [cat,dog]
    if word_label=='straight': return [0,1,0]
    elif word_label=='right': return [0,0,1]
    elif word_label=='left': return [1,0,0]
    
def create_train_data():
    training_data=[]
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label=label_img(img)
        path=os.path.join(TRAIN_DIR,img)
        print('Path -',path)
        img=cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(WIDTH,HEIGHT))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('training_data.npy',training_data)
    return training_data

def process_test_data():
    testing_data=[]
    for img in tqdm(os.listdir(TEST_DIR)):
        path=os.path.join(TEST_DIR,img)
        print('Path -',path)
        img_num=img.split('.')[0]
        img=cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(WIDTH,HEIGHT))
        testing_data.append([np.array(img),img_num])
    shuffle(testing_data)
    np.save('test_data.npy',testing_data)
    return testing_data

#train_data=create_train_data()
test_data=process_test_data()
#train_data13=np.load('training_data_keys_v13.npy')
train_data14=np.load('training_data_keys_v15.npy')
#train_data=np.concatenate((train_data13, train_data14), axis=0)
train_data=train_data14

#shuffle(train_data)
###############################################################################################################
#train_data=np.load('training_data_keys_v14.npy')
len(train_data)

df = pd.DataFrame(train_data)
print(df.head())
print("Original Driven Training data",Counter(df[1].apply(str)))
len(df)

for i in range(1,8):
    train_data=np.vstack((train_data,train_data))



df = pd.DataFrame(train_data)
print(df.head())
print("Stacked Training data",Counter(df[1].apply(str)))
len(df)

lefts = []
rights = []
forwards = []
home = []
reverse = []
#shuffle(train_data)

for data in train_data:
    img = data[0]
    choice = data[1]

    if (choice==np.array([1,0,0,0,0])).all():
        lefts.append([img,choice])
    elif (choice==np.array([0,1,0,0,0])).all():
        forwards.append([img,choice])
    elif (choice == np.array([0,0,1,0,0])).all():
        rights.append([img,choice])
    elif (choice == np.array([0,0,0,1,0])).all():
        home.append([img,choice])
    elif (choice == np.array([0,0,0,0,1])).all():
        reverse.append([img,choice])
    else:
        print('no matches')

		
		
#forwards = forwards[:len(lefts)][:len(rights)]
#lefts = lefts[:len(forwards)]
#rights = rights[:len(forwards)]

#final_data = forwards+forwards+ lefts[:round(len(lefts)/2)]+ rights+rights[:round(len(rights)/2)]+home[:round(len(home)/40)]+reverse 
final_data = forwards+lefts+rights

#shuffle(final_data)
len(final_data)
np.save('training_data_v15_balanced1.npy', final_data)

train_data=final_data
#shuffle(train_data)
np.random.shuffle(train_data)
df = pd.DataFrame(train_data)
print(df.head())
#print(df.tail(-1))
#df.iloc[1:]

print("Balanced Training data",Counter(df[1].apply(str)))
len(df)
#df.iloc[[0],[0]]
import PIL
 
type(train_data)
# Convert PIL Image to NumPy array
#img = PIL.Image.open("foo.jpg")
#arr = numpy.array(img)
 
# Convert array to Image
#train_data=np.load('training_data_v14_balanced5.npy')
#shuffle(train_data)
for num,data in enumerate(train_data[:10]):
  
    trg_img_data = data[0]
    trg_img_label = data[1]

img = PIL.Image.fromarray(trg_img_data)
width, height = img.size
###############################################################################################################
#import tensorflow
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization
import tensorflow as tf
tf.reset_default_graph()
# =============================================================================
# convnet = input_data(shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name='input')
# 
# convnet = conv_2d(convnet, 32, 5, activation='relu')
# convnet = max_pool_2d(convnet, 5)
# 
# convnet = conv_2d(convnet, 64, 5, activation='relu')
# convnet = max_pool_2d(convnet, 5)
# 
# convnet = conv_2d(convnet, 128, 5, activation='relu')
# convnet = max_pool_2d(convnet, 5)
# 
# convnet = conv_2d(convnet, 64, 5, activation='relu')
# convnet = max_pool_2d(convnet, 5)
# 
# convnet = conv_2d(convnet, 32, 5, activation='relu')
# convnet = max_pool_2d(convnet, 5)
# 
# convnet = fully_connected(convnet, 1024, activation='relu')
# convnet = dropout(convnet, 0.8)
# 
# convnet = fully_connected(convnet, 2, activation='softmax')
# convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
# 
# =============================================================================
#model = tflearn.DNN(convnet, tensorboard_dir='log')
network = input_data(shape=[None, WIDTH, HEIGHT, 1], name='input')
network = conv_2d(network, 96, 11, strides=4, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 5, activation='softmax')
network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=LR, name='targets')

model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                        max_checkpoints=1, tensorboard_verbose=3, tensorboard_dir='log')



#model = tflearn.DNN(network, checkpoint_path='model_alexnet')

#print('inputnode',bundle.Graph.Operation("))

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME,weights_only=True)
    print('MODEL_NAME',MODEL_NAME)
    #model.load('cardriving-0.001-2conv-basic.model')

    print('model loaded!')

train = train_data[:48000]
test = train_data[48001:]
print("Training length",len(train))
print("Testing length",len(test))

X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=1, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)

print(MODEL_NAME)
#saver = tf.train.Saver()
#saver.save(sess, 'my-test-model')

############## Test the Results #########


import matplotlib.pyplot as plt

# if you need to create the data:
#test_data = process_test_data()
# if you already have some saved:
test_data = np.load('test_data.npy')
df = pd.DataFrame(test_data)
print(df.head())
#print(Counter(df[1].apply(str)))


fig=plt.figure(figsize=(15,15))
#fig.figsize=(15,15)
for num,data in enumerate(test_data[33:64]):
    # cat: [1,0]
    # dog: [0,1]
    
    img_num = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(8,4,num+1)
    orig = img_data
    data = img_data.reshape(WIDTH,HEIGHT,1)
    #model_out = model.predict([data])[0]
    model_out = model.predict([data])[0]
    print(model_out)
    if np.argmax(model_out) == 1: str_label='Straight'
    elif np.argmax(model_out) == 0: str_label='Left'
    elif np.argmax(model_out) == 2: str_label='Right'
    elif np.argmax(model_out) == 3: str_label='Home'
    else: str_label='Reverse'
    

    y.imshow(orig,cmap='gray')
    plt.title(str_label,color="r")
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()