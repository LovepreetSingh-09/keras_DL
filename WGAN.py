
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

def train(models,x_train,params):
    generator,discriminator,adversarial=models
    batch_size,latent_size,n_critic,clip_value,train_steps,model_name=params
    save_interval=200
    noise_input=np.random.uniform(-1.0,1.0,size=[16,latent_size])
    train_size=x_train.shape[0]
    real_labels=np.ones([batch_size,1])
    for i in range(train_steps):
        loss=0
        acc=0
        for _ in range(n_critic):
            rand_idx=np.random.randint(0,train_size,size=batch_size)
            real_images=x_train[rand_idx]
            noise=np.random.uniform(-1.0,1.0,size=[batch_size,latent_size])
            fake_images=generator.predict(noise)
            real_loss,real_acc=discriminator.train_on_batch(real_images,real_labels)
            fake_loss,fake_acc=discriminator.train_on_batch(fake_images,-real_labels)
            loss+=0.5*(real_loss+fake_loss)
            acc+=0.5*(real_acc+fake_acc)
            for layers in discriminator.layers:
                weights=layers.get_weights()
                weights=[np.clip(weight,-clip_value,clip_value) for weight in weights]
                layers.set_weights(weights)
        loss/=n_critic
        acc/=n_critic
        log = "%d: [discriminator loss: %f, acc: %f]" % (i, loss, acc)

        noise=np.random.uniform(-1.0,1.0,size=[batch_size,latent_size])
        loss,acc=adversarial.train_on_batch(noise,real_labels)
        log = "%s [adversarial loss: %f, acc: %f]" % (log, loss, acc)
        print(log)
        if (i + 1) % save_interval == 0:
            if (i + 1) == save_interval:
                show = True
            else:
                show = False
            gan.plot_images(generator, noise_input=noise_input,show=show, step=(i + 1),model_name=model_name)
    generator.save(model_name + ".h5")

def wesserstein_loss(y_labels,y_pred):
    return -k.mean(y_labels*y_pred)

def build_and_train():
    batch_size=64
    latent_size=100
    n_critic=5
    clip_value=0.01
    train_steps=40000
    lr=5e-05
    model_name='wgan.mnist'
    (x_train,_),(x_test,_)=mnist.load_data()
    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_train = x_train.astype('float32') / 255
    input_shape=[image_size,image_size,1]
    inputs=Input(shape=input_shape)
    discriminator=gan.discriminator(inputs,image_size,activation='linear')
    optimizer=RMSprop(lr=lr)
    discriminator.compile(loss=wesserstein_loss,optimizer=optimizer,metrics=['accuracy'])
    discriminator.summary()

    input_shape=(latent_size,)
    inputs=Input(shape=input_shape)
    generator=gan.generator(inputs,image_size)
    generator.summary()
    
    discriminator.trainable=False
    adversarial=Model(inputs,discriminator(generator(inputs)))
    adversarial.compile(loss=wesserstein_loss,optimizer=optimizer,metrics=['accuracy'])
    adversarial.summary()
    models = (generator, discriminator, adversarial)
    params = (batch_size, latent_size,n_critic,clip_value,train_steps, model_name)
    train(models, x_train, params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load generator h5 model with trained weights"
    parser.add_argument("-g", "--generator", help=help_)
    args = parser.parse_args()
    if args.generator:
        generator = load_model(args.generator)
        gan.test_generator(generator)
    else:
        build_and_train()


    
