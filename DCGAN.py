
from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.optimizers import RMSprop
from keras.models import Model
from keras.datasets import mnist
from keras.models import load_model

import numpy as np
import math
import matplotlib.pyplot as plt
import os
import argparse

def build_generator(inputs,image_size):
    image_resize=image_size//4
    kernel_size=5
    layers_filter=[128,64,32,1]
    x=Dense(image_resize*image_resize*layers_filter[0])(inputs)
    x=Reshape((image_resize,image_resize,layers_filter[0]))(x)
    for fil in layers_filter:
        if fil<=layers_filter[-2]:
            strides=1
        else:
            strides=2
        x=BatchNormalization()(x)
        x=Activation('relu')(x)
        x=Conv2DTranspose(fil,kernel_size=kernel_size,strides=strides,padding='same')(x)
    x=Activation('sigmoid')(x)
    generator=Model(inputs=inputs,outputs=x,name='generator_output')
    return generator

def build_discriminator(inputs):
    kernel_size=5
    filter_layers=[32,64,128,256]
    x=inputs
    for fil in filter_layers:
        if fil==filter_layers[-1]:
            strides=1
        else:
            strides=2
        x=LeakyReLU(alpha=0.2)(x)
        x=Conv2D(fil,kernel_size=kernel_size,strides=strides,padding='same')(x)
    x=Flatten()(x)
    x=Dense(1)(x)
    x=Activation('sigmoid')(x)
    discriminator=Model(inputs,x,name='generator')
    return discriminator

def train(Model,x_train,params):
    generator,discriminator,adversarial=Model
    batch_size,latent_size,train_steps,model_name=params
    save_interval=500
    noise_input=np.random.uniform(-1.0,1.0,size=(16,latent_size))
    train_size=x_train.shape[0]
    for i in range(train_steps):
        red_idx=np.random.randint(0,train_size,size=batch_size)
        real_images=x_train[red_idx];real_images.shape
        noise=np.random.uniform(-1.0,1.0,size=(batch_size,latent_size))
        fake_images=generator.predict(noise);fake_images.shape
        x=np.concatenate((real_images,fake_images));x.shape
        y=np.ones([2*batch_size,1])
        y[batch_size:,:]=0.0
        loss,acc=discriminator.train_on_batch(x,y)
        log = "%d: [discriminator loss: %f, acc: %f]" % (i, loss, acc)

        noise=np.random.uniform(-1.0,1.0,size=[batch_size,latent_size])
        y=np.ones([batch_size,1])
        loss,acc=adversarial.train_on_batch(noise,y)
        log = "%s [adversarial loss: %f, acc: %f]" % (log, loss, acc)
        print(log)
        if (i + 1) % save_interval == 0:
            if (i + 1) == train_steps:
                show = True
            else:
                show = False
            plot_images(generator, noise_input=noise_input,
                        show=show, step=(i + 1),model_name=model_name)
    generator.save(model_name+'.h5')

def plot_images(generator,noise_input,show=False,step=0,model_name='gan'):
    os.makedirs(model_name,exist_ok=True)
    filename = os.path.join(model_name, "%05d.png" % step)
    images=generator.predict(noise_input)
    plt.figure(figsize=(2.2, 2.2))
    num_images = images.shape[0]
    image_size = images.shape[1]
    rows = int(math.sqrt(noise_input.shape[0]))
    for i in range(num_images):
        plt.subplot(rows, rows, i + 1)
        image = np.reshape(images[i], [image_size, image_size])
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.savefig(filename)
    if show:
        plt.show()
    else:
        plt.close('all')

def build_and_train_models():
    (x_train, _), (_, _) = mnist.load_data()
    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_train = x_train.astype('float32') / 255
    model_name = "dcgan_mnist"
    latent_dim=100
    train_steps=40000
    batch_size=64
    lr=2e-04
    decay=6e-08

    image_shape=(image_size,image_size,1)
    inputs=Input(shape=image_shape,name='discriminator_input')
    discriminator=build_discriminator(inputs)
    optimizer=RMSprop(lr=lr,decay=decay)
    discriminator.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    discriminator.summary()

    inputs=Input(shape=(latent_dim,))
    generator=build_generator(inputs,image_size)
    generator.summary()

    discriminator.trainable=False
    optimizer=RMSprop(lr=lr*0.5,decay=decay*0.5)
    adversarial=Model(inputs,discriminator(generator(inputs)),name=model_name)
    adversarial.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    adversarial.summary()

    models = (generator, discriminator, adversarial)
    params = (batch_size, latent_dim, train_steps, model_name)
    train(models, x_train, params)


def test_generator(generator):
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
    plot_images(generator, noise_input=noise_input,
                show=True,model_name="test_outputs")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load generator h5 model with trained weights"
    parser.add_argument("-g", "--generator", help=help_)
    args = parser.parse_args()
    if args.generator:
        generator = load_model(args.generator)
        test_generator(generator)
    else:
        build_and_train_models()

