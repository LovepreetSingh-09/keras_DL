
import keras.backend as k
from keras.layers import Dense,Conv2D,Flatten,Conv2DTranspose,Input
from keras.layers import Reshape
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

(x_train,_),(x_test,_)=mnist.load_data()

x_train=x_train.astype('float32')/255.
x_test=x_test.astype('float32')/255.

image_size = x_train.shape[1]
x_train = x_train.reshape([-1, image_size, image_size, 1])
x_test = x_test.reshape([-1, image_size, image_size, 1])

plt.imshow(x_test[0].reshape(image_size,image_size),cmap='gray')

noise=np.random.normal(loc=0.5,scale=0.5,size=x_train.shape)
x_train_noisy=x_train+noise
noise = np.random.normal(loc=0.5, scale=0.5, size=x_test.shape)
x_test_noisy=x_test+noise

print(x_test_noisy.shape)
x_train_noisy=np.clip(x_train_noisy,0.,1.)
x_test_noisy=np.clip(x_test_noisy,0.,1.)
plt.imshow(x_test_noisy[0].reshape(image_size, image_size), cmap='gray')

input_shape=x_train_noisy.shape[1:]
batch_size=128
kernel_size=3
filter_layers=[32,64]
latent_dim=16

inputs=Input(shape=input_shape)
x=inputs
for fil in filter_layers:
    x=Conv2D(fil,kernel_size=kernel_size,strides=2,padding='same')(x)
shape=k.int_shape(x)
x=Flatten()(x)
outputs=Dense(latent_dim)(x)

encoder=Model(inputs=inputs,outputs=outputs)
encoder.summary()

latent_input=Input(shape=(latent_dim,))
x=latent_input
x=Dense(shape[1]*shape[2]*shape[3])(x)
x=Reshape((shape[1],shape[2],shape[3]))(x)
for fil in filter_layers[::-1]:
    x=Conv2DTranspose(fil,kernel_size=kernel_size,strides=2,padding='same')(x)
outputs=Conv2D(1,kernel_size=kernel_size,activation='sigmoid',strides=1,padding='same')(x)

decoder=Model(inputs=latent_input,outputs=outputs)
decoder.summary()

autoencoder=Model(inputs,decoder(encoder(inputs)))
autoencoder.summary()

autoencoder.compile(loss='mse',optimizer='adam')

epochs=2
autoencoder.fit(x_train_noisy,x_train,batch_size=batch_size,validation_data=(x_test_noisy,x_test),epochs=epochs)

x_decoded=autoencoder.predict(x_test_noisy)
plt.imshow(x_decoded[0].reshape(image_size,image_size),cmap='gray')

rows=3
col=9
num=rows*col
imgs=np.concatenate((x_test[:num],x_test_noisy[:num],x_decoded[:num]))
print(imgs.shape)
imgs=imgs.reshape((3*rows,col,image_size,image_size))
print(imgs.shape)
imgs=np.split(imgs,rows,axis=1)
print(np.array(imgs).shape)

imgs=np.vstack(imgs);imgs.shape
imgs=np.vstack([np.hstack(i) for i in imgs]);imgs.shape
imgs=(imgs*255.).astype(np.uint8)

plt.figure(figsize=(10,20))
plt.axis('off')
plt.title('Original:Top rows Corrupted:middle-rows  Denoised:third-rows')
plt.imshow(imgs, interpolation='none', cmap='gray')
plt.show()
