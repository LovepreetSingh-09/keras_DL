from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.optimizers import RMSprop
from keras.models import Model
from keras.layers.merge import concatenate
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import load_model

import numpy as np
import math
import matplotlib.pyplot as plt
import os
import argparse

def generator(inputs,image_size,activation='sigmoid',codes=None,labels=None):
    im_res=image_size//4
    kernel_size=5
    filter_layers=[128,64,32,1]
    if labels is not None:
        if codes is None:
            inputs=[inputs,labels]
        else:
            inputs=[inputs,labels]+codes
        x=concatenate(inputs,axis=1)
    elif codes is not None:
        inputs=[inputs,codes]
        x=concatenate(inputs,codes)
    else:
        x=inputs
    x=Dense(im_res*im_res*filter_layers[0])(x)
    x=Reshape((im_res,im_res,filter_layers[0]))(x)
    for fil in filter_layers:
        if fil>filter_layers[-2]:
            strides=2
        else:
            strides=1
        x=BatchNormalization()(x)
        x=Activation('relu')(x)
        x=Conv2DTranspose(fil,padding='same',strides=strides,kernel_size=kernel_size)(x)
    if activation is not None:
        x=Activation(activation)(x)
    return Model(inputs,x)

def discriminator(inputs,image_size,activation='sigmoid',num_labels=None,num_codes=None):
    kernel_size=5
    filter_layers=[32,64,128,256]
    x=inputs
    for fil in filter_layers:
        if fil<filter_layers[-1]:
            strides=2
        else:
            strides=1
        x=LeakyReLU(alpha=0.2)(x)
        x=Conv2D(fil,padding='same',kernel_size=kernel_size,strides=strides)(x)
    x=Flatten()(x)
    outputs=Dense(1)(x)
    if activation is not None:
        print(activation)
        outputs=Activation(activation)(outputs)
    if num_labels is not None:
        layer=Dense(filter_layers[-2])(x)
        labels=Dense(num_labels)(layer)
        labels=Activation('sigmoid')(labels)
        if num_codes is None:
            outputs=[outputs,labels]
        else:
            code1=Dense(1)(layer)
            code1=Activation('sigmoid')(code1)
            code2=Dense(1)(layer)
            code2=Activation('sigmoid')(code2)
            outputs=[outputs,labels,code1,code2]
    elif num_codes is not None:
        z0_recon=Dense(num_codes)(x)
        z0_recon=Activation('tanh')(x)
        outputs=[outputs,z0_recon]
    return Model(inputs,outputs)


def train(Model, x_train, params):
    generator, discriminator, adversarial = Model
    batch_size, latent_size, train_steps, model_name = params
    save_interval = 500
    noise_input = np.random.uniform(-1.0, 1.0, size=(16, latent_size))
    train_size = x_train.shape[0]
    for i in range(train_steps):
        red_idx = np.random.randint(0, train_size, size=batch_size)
        real_images = x_train[red_idx]
        real_images.shape
        noise = np.random.uniform(-1.0, 1.0, size=(batch_size, latent_size))
        fake_images = generator.predict(noise)
        fake_images.shape
        x = np.concatenate((real_images, fake_images))
        x.shape
        y = np.ones([2*batch_size, 1])
        y[batch_size:, :] = 0.0
        loss, acc = discriminator.train_on_batch(x, y)
        log = "%d: [discriminator loss: %f, acc: %f]" % (i, loss, acc)

        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])
        y = np.ones([batch_size, 1])
        loss, acc = adversarial.train_on_batch(noise, y)
        log = "%s [adversarial loss: %f, acc: %f]" % (log, loss, acc)
        print(log)
        if (i + 1) % save_interval == 0:
            show=True
            plot_images(generator, noise_input=noise_input,
                        show=show, step=(i + 1), model_name=model_name)
    generator.save(model_name+'.h5')


def plot_images(generator,noise_input,noise_label=None, noise_codes=None,
                show=False,step=0,model_name="gan"):
    os.makedirs(model_name, exist_ok=True)
    filename = os.path.join(model_name, "%05d.png" % step)
    rows = int(math.sqrt(noise_input.shape[0]))
    if noise_label is not None:
        noise_input = [noise_input, noise_label]
        if noise_codes is not None:
            noise_input += noise_codes
    images = generator.predict(noise_input)
    plt.figure(figsize=(2.2, 2.2))
    num_images = images.shape[0]
    image_size = images.shape[1]
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


def test_generator(generator):
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
    plot_images(generator, noise_input=noise_input,show=True,model_name="test_outputs")
