
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


def build_discriminator(inputs, y_labels, image_size):
     kernel_size = 5
     filter_layers = [32, 64, 128, 256]
     x = inputs
     y = Dense(image_size*image_size*1)(y_labels)
     y = Reshape((image_size, image_size, 1))(y)
     x = concatenate([x, y])
     for fil in filter_layers:
        if fil == filter_layers[-1]:
             strides = 1
        else:
             strides = 2
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(fil, strides=strides, padding='same',
                   kernel_size=kernel_size)(x)
     x=Flatten()(x)
     x=Dense(1)(x)
     x=Activation('sigmoid')(x)
     discriminator=Model([inputs,y_labels],x)
     return discriminator


def build_generator(inputs,y_labels,image_size):
    image_resize=image_size//4
    filter_layers=[128,64,32,1]
    kernel_size=5
    x=concatenate([inputs,y_labels])
    x=Dense(image_resize*image_resize*filter_layers[0])(x)
    x=Reshape((image_resize,image_resize,filter_layers[0]))(x)
    for fil in filter_layers:
        if fil>filter_layers[-2]:
            strides=2
        else:
            strides=1
        x=BatchNormalization()(x)
        x=Activation('relu')(x)
        x=Conv2DTranspose(fil,strides=strides,padding='same',kernel_size=kernel_size)(x)
    x=Activation('sigmoid')(x)
    generator=Model([inputs,y_labels],x)
    return generator


def train(Model,params,data):
    generator,discriminator,adversarial=Model
    x_train,y_train=data
    batch_size,latent_size,num_labels,train_steps,model_name=params
    save_interval=200
    noise_input=np.random.uniform(-1.0,1.0,size=[16,latent_size])
    noise_class=np.eye(num_labels)[np.arange(0,16) %num_labels]
    print(noise_class.shape,noise_class)
    train_size=x_train.shape[0]
    print(model_name,"Labels for generated images: ",
          np.argmax(noise_class, axis=1))
    for i in range(train_steps):
        rand_idx=np.random.randint(0,train_size,size=batch_size)
        real_images=x_train[rand_idx]
        real_labels=y_train[rand_idx]
        noise=np.random.uniform(-1.0,1.0,size=[batch_size,latent_size])
        fake_labels=np.eye(num_labels)[np.random.choice(num_labels,batch_size)]
        fake_images=generator.predict([noise,fake_labels])
        
        x=np.concatenate((real_images,fake_images))
        labels=np.concatenate((real_labels,fake_labels))
        y=np.ones((2*batch_size,1))
        y[batch_size:,:]=0
        loss,acc=discriminator.train_on_batch([x,labels],y)
        log = "%d: [discriminator loss: %f, acc: %f]" % (i, loss, acc)

        noise=np.random.uniform(-1.0,1.0,size=[batch_size,latent_size])
        fake_labels=np.eye(num_labels)[np.random.choice(num_labels,batch_size)]
        y=np.ones((batch_size,1))
        loss,acc=adversarial.train_on_batch([noise,fake_labels],y)
        log = "%s [adversarial loss: %f, acc: %f]" % (log, loss, acc)
        print(log)
        if (i + 1) % save_interval == 0:
            if (i + 1) == train_steps:
                show = True
            else:
                show = False
            plot_images(generator,noise_input=noise_input,noise_class=noise_class,
                        show=show,step=(i + 1),model_name=model_name)
    generator.save(model_name+'.h5')


def plot_images(generator, noise_input,noise_class,show=False,step=0, model_name="gan"):
    os.makedirs(model_name, exist_ok=True)
    filename = os.path.join(model_name, "%05d.png" % step)
    images = generator.predict([noise_input, noise_class])
    print(model_name , " labels for generated images: ", np.argmax(noise_class, axis=1))
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


def build_and_train():
    (x_train, y_train), (_, _) = mnist.load_data()
    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_train = x_train.astype('float32') / 255
    num_labels = np.amax(y_train) + 1
    y_train = to_categorical(y_train)
    model_name = "cgan_mnist"
    latent_size = 100
    batch_size = 64
    train_steps = 40000
    lr = 2e-4
    decay = 6e-8
    input_shape=x_train.shape[1:]
    labels_shape=(num_labels,)

    inputs=Input(shape=input_shape)
    labels=Input(shape=labels_shape)
    discriminator=build_discriminator(inputs,labels,image_size)
    discriminator.summary()
    optimizer=RMSprop(lr=lr,decay=decay)
    discriminator.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])

    inputs=Input(shape=(latent_size,))
    labels=Input(shape=labels_shape)
    generator=build_generator(inputs,labels,image_size)
    generator.summary()
    
    discriminator.trainable=False
    optimizer=RMSprop(lr=lr*0.5,decay=decay*0.5)
    adversarial=Model([inputs,labels],discriminator([generator([inputs,labels]),labels]),name=model_name)
    adversarial.summary()
    adversarial.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    models = (generator, discriminator, adversarial)
    data = (x_train, y_train)
    params = (batch_size, latent_size, num_labels,train_steps, model_name)
    train(models,params,data)

def test_generator(generator, class_label=None):
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
    step = 0
    if class_label is None:
        num_labels = 10
        noise_class = np.eye(num_labels)[np.random.choice(num_labels, 16)]
    else:
        noise_class = np.zeros((16, 10))
        noise_class[:,class_label] = 1
        step = class_label

    plot_images(generator, noise_input=noise_input,noise_class=noise_class,
                show=True,step=step,model_name="test_outputs")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load generator h5 model with trained weights"
    parser.add_argument("-g", "--generator", help=help_)
    help_ = "Specify a specific digit to generate"
    parser.add_argument("-d", "--digit", type=int, help=help_)
    args = parser.parse_args()
    if args.generator:
        generator = load_model(args.generator)
        class_label = None
        if args.digit is not None:
            class_label = args.digit
        test_generator(generator, class_label)
    else:
        build_and_train()

        




