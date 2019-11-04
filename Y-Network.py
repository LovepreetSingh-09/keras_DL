
import numpy as np
import keras
from keras.layers import Dense,Conv2D,Dropout,Activation,Input,MaxPool2D,Flatten
from keras.layers.merge import concatenate
from keras.models import Model
from keras.utils import to_categorical, plot_model
from keras.datasets import mnist
import matplotlib.pyplot as plt
import pydot

(x_train, y_train_), (x_test, y_test_) = mnist.load_data()
x_train.shape
x_test.shape

y_test_.shape

targets, count = np.unique(y_train_, return_counts=True)
targets
count

num_labels = len(np.unique(y_train_))
print('Train Labels :', dict(zip(targets, count)))
labels, counts = np.unique(y_test_, return_counts=True)
print('Test Labels :', dict(zip(labels, counts)))

input_size=x_train.shape[1]
input_shape=[input_size,input_size,1]
X_train=x_train.reshape(-1,input_size,input_size,1)/255.
X_test= x_test.reshape(-1, input_size, input_size,1)/255.

y_train=to_categorical(y_train_)
y_test=to_categorical(y_test_)

batch_size=128
n_filters=32
kernel_size=3
dropout=0.2

left_inputs=Input(shape=input_shape)
x=left_inputs
filters=n_filters
for i in range(3):
    x=Conv2D(filters=filters,kernel_size=kernel_size,padding='same',activation='relu')(x)
    x=Dropout(dropout)(x)
    x=MaxPool2D()(x)
    filters*=2

right_inputs=Input(shape=input_shape)
y=right_inputs
filters=n_filters
for i in range(3):
    y=Conv2D(filters=filters,padding='same',kernel_size=kernel_size,dilation_rate=2,activation='relu')(y)
    y=Dropout(dropout)(y)
    y=MaxPool2D()(y)
    filters*=2

y=concatenate([x,y])
y=Flatten()(y)
y=Dropout(dropout)(y)
outputs=Dense(num_labels,activation='softmax')(y)

model=Model([left_inputs,right_inputs],outputs)

model.summary()

plot_model(model,to_file='Y_Network.png',show_shapes=True)

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit([X_train,X_train],y_train,validation_data=([X_test,X_test],y_test),batch_size=batch_size,epochs=10)

loss,acc=model.evaluate([X_test,X_test],y_test,batch_size=batch_size)

print('Test Accuracy : %.3f' %(100*acc))
