
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense,Conv2D,Conv2DTranspose,Flatten,Activation,Input
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Reshape
from keras.datasets import mnist
import keras.backend as k

(x_train,y_train),(x_test,y_test)=mnist.load_data()

X_train = x_train.astype('float32')/255.
X_test = x_test.astype('float32')/255.
X_train=X_train.reshape([-1,28,28,1])
X_test=X_test.reshape([-1,28,28,1])
input_shape = X_train.shape[1:]


filters=[64,32]
kernel_size=3
batch_size=128
latent_dim=2

inputs=Input(shape=input_shape,name='encoder_input')
x=inputs
for fl in filters:
    x=Conv2D(fl,kernel_size=kernel_size,padding='same',activation='relu',kernel_initializer='he_normal',strides=2)(x)
k_shape=k.int_shape(x)
x=Flatten()(x)
outputs=Dense(latent_dim,kernel_initializer='he_normal',name='encoder_output')(x)

encoder=Model(inputs=inputs,outputs=outputs,name='encoder')
encoder.summary()
plot_model(encoder,to_file='encoder_mnist.png',show_shapes=True)

print(k_shape)
latent_inputs=Input(shape=(latent_dim,),name='decoder_input')
x=latent_inputs
x=Dense(k_shape[1]*k_shape[2]*k_shape[3])(x)
x=Reshape((k_shape[1],k_shape[2],k_shape[3]))(x)
for fl in filters[::-1]:
    x=Conv2DTranspose(fl,padding='same',strides=2,activation='relu',kernel_size=kernel_size)(x)
outputs=Conv2DTranspose(1,padding='same',activation='sigmoid',kernel_size=kernel_size,name='decoder_output')(x)

decoder=Model(inputs=latent_inputs,outputs=outputs,name='decoder_1')
decoder.summary()
plot_model(decoder,to_file='decoder_mnist.png',show_shapes=True)

autoencoder=Model(inputs,decoder(encoder(inputs)),name='autoencoder')
autoencoder.summary()
plot_model(autoencoder,to_file='autoencoder_mnist.png',show_shapes=True)

autoencoder.compile(loss='mse',optimizer='adam')

epochs=1
autoencoder.fit(X_train,X_train,validation_data=(X_test,X_test),epochs=epochs,shuffle=True,batch_size=batch_size)

x_decoded=autoencoder.predict(X_test)
x_decoded[1][:,:,-1].shape
plt.imshow(x_decoded[1][:,:,-1])

image_size=28
imgs=np.concatenate([X_test[:8],x_decoded[:8]])
imgs=imgs.reshape([4,4,image_size,image_size])
v=np.array([np.hstack(i) for i in imgs]);v.shape
imgs=np.vstack([np.hstack(i) for i in imgs])
print(imgs.shape)
plt.figure()
plt.axis('off')
plt.title('Input:1-2 rows  Output:3-4 rows')
plt.imshow(imgs,cmap='gray')
plt.savefig('input_decoded_mnist.png')
plt.show()

def plot_results(model,data,batch_size,model_name='autoencoder_2dim'):
    encoder,decoder=model
    x_test,y_test=data
    z=encoder.predict(x_test,batch_size=batch_size)
    plt.figure(figsize=(12,10))
    plt.scatter(z[:,0],z[:,1],c=y_test)
    plt.colorbar()
    plt.show()

    n=30
    digit_size=28
    figure=np.zeros([n*digit_size,n*digit_size])
    grid_x=np.linspace(-2,12,n)
    grid_y=np.linspace(2,12,n)[::-1]
    for i,xi in enumerate(grid_x):
        for j,yj in enumerate(grid_y):
            z=np.array([[xi,yj]])
            x_decoded=decoder.predict(z)
            print(x_decoded.shape)
            print(x_decoded[0].shape)
            digit=x_decoded[0].reshape(28,28)
            print(digit.shape)
            figure[i*digit_size:(i+1)*digit_size,j*digit_size:(j+1)*digit_size]=digit
    plt.figure(figsize=(6,10))
    start_range=0
    end_range=n*digit_size+1
    pixel_range=np.arange(start_range,end_range,digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure,cmap="Greys_r")
    plt.show()

plot_results((encoder,decoder),data=(X_test,y_test),batch_size=32)
