import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense,Conv2D,Conv2DTranspose,Flatten,Activation,Input
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Reshape
from keras.datasets import cifar10
import keras.backend as k
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
import os

(x_train,_),(x_test,_)=cifar10.load_data()
x_train.shape

def rgb2grey(rgb):
    return np.dot(rgb[:,:,:,:3],[0.299,0.587,0.114])

img_rows=x_train.shape[1];img_rows
img_cols=x_train.shape[2];img_cols
img_channels=x_train.shape[3];img_channels

imgs=x_test[:100]
imgs.shape
imgs=imgs.reshape([10,10,img_rows,img_cols,img_channels])
imgs=np.vstack([np.hstack(i) for i in imgs]);imgs.shape
plt.figure(figsize=(7,7))
plt.axis('off')
plt.imshow(imgs,interpolation='none')
plt.show()

x_train_grey=rgb2grey(x_train)
x_test_grey=rgb2grey(x_test)
# Shape (50000, 32, 32)
x_train_grey.shape

imgs = x_test_grey[:100]
imgs = imgs.reshape([10, 10, img_rows, img_cols])
imgs = np.vstack([np.hstack(i) for i in imgs])
imgs.shape
plt.figure(figsize=(7,7))
plt.axis('off')
plt.imshow(imgs, interpolation='none',cmap='gray')
plt.show()

x_train=x_train.astype('float32')/255
x_test=x_test.astype('float32')/255
x_train=x_train.reshape([-1,img_rows,img_cols,img_channels])
x_test=x_test.reshape([-1,img_rows,img_cols,img_channels])

x_train_grey=x_train_grey.astype('float32')/255
x_test_grey = x_test_grey.astype('float32')/255
x_train_grey.shape
x_train_grey = x_train_grey.reshape([-1, img_rows, img_cols, 1])
x_test_grey = x_test_grey.reshape([-1, img_rows, img_cols, 1])
print(x_test_grey.shape)

input_shape=x_train_grey.shape[1:]
batch_size=128
latent_dim=256
filter_layers=[64,128,256]
kernel_size=3

inputs=Input(shape=input_shape)
x=inputs
for fil in filter_layers:
    x=Conv2D(fil,kernel_size=kernel_size,strides=2,padding='same',activation='relu')(x)
shape=k.int_shape(x)
x=Flatten()(x)
outputs=Dense(latent_dim)(x)

encoder=Model(inputs=inputs,outputs=outputs)
encoder.summary()

print(shape)

latent_inputs=Input(shape=(latent_dim,))
x=latent_inputs
x=Dense(shape[1]*shape[2]*shape[3])(x)
x=Reshape([shape[1],shape[2],shape[3]])(x)
for fil in filter_layers[::-1]:
    x=Conv2DTranspose(fil,kernel_size=kernel_size,strides=2,padding='same',activation='relu')(x)
outputs=Conv2DTranspose(3,kernel_size=kernel_size,activation='sigmoid',padding='same')(x)

decoder=Model(inputs=latent_inputs,outputs=outputs)
decoder.summary()

autoencoder=Model(inputs,decoder(encoder(inputs)))
autoencoder.summary()

autoencoder.compile(loss='mse',optimizer='adam')

os.getcwd()
save_dir=os.path.join(os.getcwd(),'keras_models')
model_name='cifar10_coloured_{epoch:03d}.h5'

if not os.path.isdir(save_dir):
    os.makedir(save_dir)
    
filepath=os.path.join(save_dir,model_name)

lr_reducer=ReduceLROnPlateau(factor=np.sqrt(0.1),cool_down=0,monitor='val_acc',patience=5,min_lr=0.5e-06,verbose=1)
checkpoint=ModelCheckpoint(filepath,monitor='val_acc',verbose=1,save_best_only=True)

callbacks=[lr_reducer,checkpoint]

autoencoder.fit(x_train_grey,x_train,validation_data=(x_test_grey,x_test),epochs=30,batch_size=128,callbacks=callbacks)

x_decoded=autoencoder.predict(x_test_grey)

imgs=x_decoded[:100]
imgs=imgs.reshape([10,10,img_rows,img_cols,img_channels])
imgs=np.vstack([np.hstack(i) for i in imgs])
imgs.shape
plt.figure(figsize=(7,7))
plt.axis('off')
plt.imshow(imgs,interpolation='none')
plt.show()
