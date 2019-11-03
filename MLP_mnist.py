# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 13:23:44 2019

@author: user
"""
import pydot
import numpy as np
import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout

(x_train,y_train_),(x_test,y_test_)=mnist.load_data()
x_train.shape; x_test.shape

y_test_.shape
input_size=x_train.shape[1]
input_size=input_size**2
X_train=x_train.reshape(-1,input_size)/255.; X_train.shape
X_test=x_test.reshape(-1,input_size)/255.; X_test.shape

targets,count=np.unique(y_train_,return_counts=True); targets
count

print('Train Labels :',dict(zip(targets,count)))
labels,counts=np.unique(y_test_,return_counts=True)
print('Test Labels :' ,dict(zip(labels,counts)))


c=np.where(y_train_==7)
print(c)
c=np.array(c).reshape(-1)
plt.figure(figsize=(4,2))
for j,i in enumerate(c[-10:]):
#    print(i)
    plt.subplot(2,5,j+1)
    plt.imshow(x_train[i],cmap='gray')
    plt.axis('off')
plt.show()

fig,ax=plt.subplots(2,5,figsize=(20,10))
for i,a in zip(targets,ax.ravel()):
    print(i)
    c=np.where(y_train_==i)
    c=np.array(c).reshape(-1)
    print(c[0])
    a.imshow(x_train[c[0]],cmap='gray')
plt.show()
    

y_train=to_categorical(y_train_);y_train.shape
y_test=to_categorical(y_test_)    

batch_size=32
hidden_units_1=512
hidden_units_2=256
dropout=0.4
num_labels=len(targets);num_labels

model=Sequential()
model.add(Dense(hidden_units_1,input_dim=input_size))
model.add(Activation('relu'))
model.add(Dropout(rate=dropout))
model.add(Dense(units=hidden_units_2,input_dim=hidden_units_1))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(units=num_labels,input_dim=hidden_units_2))
model.add(Activation('softmax'))

model.summary()
plot_model(model,to_file='mnist_mlp.png',show_shapes=True)

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x=X_train,y=y_train,batch_size=batch_size,epochs=20)

loss,acc=model.evaluate(x=X_test,y=y_test,batch_size=batch_size)
print('/nTest Accuracy : %.3f' %(100*acc))

