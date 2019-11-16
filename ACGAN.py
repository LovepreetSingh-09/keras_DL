
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

def train(models,data,params):
    generator,discriminator,adversarial=models
    batch_size, latent_size, train_steps, num_labels, model_name = params
    x_train,y_train=data
    save_interval=20
    noise_input=np.random.uniform(-1.0,1.0,size=[16,latent_size])
    noise_labels=np.eye(num_labels)[np.arange(0,16) %num_labels ]
    print(model_name,
          "Labels for generated images: ",
          np.argmax(noise_labels, axis=1))
    train_size=x_train.shape[0]
    for i in range(train_steps):
        rand_idx=np.random.randint(0,train_size,size=batch_size)
        noise=np.random.uniform(-1.0,1.0,size=[batch_size,latent_size])
        fake_labels=np.eye(num_labels)[np.random.choice(num_labels,batch_size)]
        real_images=x_train[rand_idx]
        real_labels=y_train[rand_idx]
        y=np.ones([2*batch_size,1])
        y[batch_size:,:]=0
        fake_images=generator.predict([noise,fake_labels])
        x=np.concatenate((real_images,fake_images))
        labels=np.concatenate((real_labels,fake_labels))
        metrics=discriminator.train_on_batch(x,[y,labels])
        # ['loss', 'activation_1_loss', 'label_loss', 'activation_1_acc', 'label_acc']
        fmt = "%d: [disc loss: %f, srcloss: %f, lblloss: %f, srcacc: %f, lblacc: %f]"
        log = fmt % (i, metrics[0], metrics[1], metrics[2], metrics[3], metrics[4])

        noise=np.random.uniform(-1.0,1.0,size=[batch_size,latent_size])
        fake_labels=np.eye(num_labels)[np.random.choice(num_labels,batch_size)]
        y=np.ones([batch_size,1])
        metrics=adversarial.train_on_batch([noise,fake_labels],[y,fake_labels])
        fmt = "%s [advr loss: %f, srcloss: %f, lblloss: %f, srcacc: %f, lblacc: %f]"
        log = fmt % (log, metrics[0], metrics[1],
                     metrics[2], metrics[3], metrics[4])
        print(log)
        if (i + 1) % save_interval == 0:
            show=True
            gan.plot_images(generator, noise_input=noise_input,noise_label=noise_labels,
                            show=show, step=(i + 1), model_name=model_name)
    generator.save(model_name + ".h5")

def build_and_train():
    latent_size=100
    batch_size=64
    model_name='acgan_mnist'
    train_steps=40000
    lr=2e-4
    decay=6e-8
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_train = x_train.astype('float32') / 255
    input_shape = [image_size, image_size, 1]
    num_labels=len(np.unique(y_train))
    y_train=to_categorical(y_train)
    inputs=Input(shape=input_shape)
    discriminator=gan.discriminator(inputs=inputs,image_size=image_size,num_labels=num_labels)
    loss=['binary_crossentropy','categorical_crossentropy']
    optimizer=RMSprop(lr=lr,decay=decay)
    discriminator.compile(loss=loss,optimizer=optimizer,metrics=['accuracy'])
    discriminator.summary()
    inputs=Input(shape=(latent_size,))
    labels=Input(shape=(num_labels,))
    generator=gan.generator(inputs=inputs,image_size=image_size,labels=labels)
    generator.summary()
    discriminator.trainable=False
    adversarial=Model([inputs,labels],discriminator(generator([inputs,labels])))    
    optimizer=RMSprop(lr=lr*0.5,decay=decay*0.5)
    adversarial.compile(loss=loss,optimizer=optimizer,metrics=['accuracy'])
    adversarial.summary()
    models = (generator, discriminator, adversarial)
    data = (x_train, y_train)
    params = (batch_size, latent_size, train_steps, num_labels, model_name)
    train(models, data, params)

def test_generator(generator,class_label):
    noise_input=np.random.uniform(-1.0,1.0,size=[16,100])
    step=0
    if class_label is None:
        num_labels=10
        labels=np.eye(num_labels)[np.arange(0,16) %num_labels]
    else:
        labels=np.zeros((16,10))
        labels[:,class_label]=1
        step=class_label
    gan.plot_images(generator,noise_input=noise_input, noise_label=labels,
                    show=True,step=step,model_name="test_outputs")


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    help_="Loading model for ACGAN generator"
    parser.add_argument('-g','--generator',help=help_)
    help1_='Digit no. for generating images'
    parser.add_argument('-d','--digit',help=help1_)
    args=parser.parse_args()
    if args.generator:
        generator=load_model(args.generator)
        class_label=None
        if args.digit is not None:
            class_label=args.digit
        test_generator(generator,class_label)
    else:
        build_and_train()




