
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
import keras.backend as k

import numpy as np
import math
import matplotlib.pyplot as plt
import os
import argparse
import gan

def build_and_train():
    (x_train,_),(x_test,_)=mnist.load_data()
    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_train = x_train.astype('float32') / 255
    input_shape = [image_size, image_size, 1]
    latent_size=100
    batch_size=64
    train_steps=40000
    lr=2e-04
    decay=6e-08
    optimizer=RMSprop(lr=lr,decay=decay)
    inputs=Input(shape=input_shape)
    discriminator=gan.discriminator(inputs,image_size,activation=None)
    discriminator.compile(loss='mse',optimizer=optimizer,metrics=['accuracy'])
    discriminator.summary()

    inputs=Input(shape=(latent_size,))
    generator=gan.generator(inputs,image_size)
    generator.summary()

    discriminator.trainable=False
    adversarial=Model(inputs,discriminator(generator(inputs)))
    adversarial.compile(loss='mse',optimizer=optimizer,metrics=['accuracy'])
    adversarial.summary()
    models=(generator,discriminator,adversarial)
    model_name='lsgan_mnist'
    params=(batch_size,latent_size,train_steps,model_name)
    gan.train(models,x_train,params)


def test_generator(generator):
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
    gan.plot_images(generator, noise_input=noise_input,
                show=True, model_name="test_outputs")

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    _help="Load h5 trained generator with optimized weights"
    parser.add_argument('-g','--generator',help=_help)
    args=parser.parse_args()
    if args.generator:
        generator=load_model(args.generator)
        test_generator(generator)
    else:
        build_and_train()


