
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


def train(models, params, data):
    generator, discriminator, adversarial = models
    x_train, y_train = data
    batch_size, latent_size, train_steps, num_labels, model_name = params
    save_interval = 50
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, latent_size])
    noise_labels = np.eye(num_labels)[np.arange(0, 16) % num_labels]
    noise_code1 = np.random.normal(scale=0.5, size=[16, 1])
    noise_code2 = np.random.normal(scale=0.5, size=[16, 1])
    for i in range(train_steps):
        rand_idx = np.random.randint(0, train_steps, size=batch_size)
        real_code1 = np.random.normal(scale=0.5, size=[batch_size, 1])
        real_code2 = np.random.normal(scale=0.5, size=[batch_size, 1])
        real_images = x_train[rand_idx]
        real_labels = y_train[rand_idx]
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])
        fake_labels = np.eye(num_labels)[
            np.random.choice(num_labels, batch_size)]
        fake_code1 = np.random.normal(scale=0.5, size=[batch_size, 1])
        fake_code2 = np.random.normal(scale=0.5, size=[batch_size, 1])
        inputs = [noise, fake_labels, fake_code1, fake_code2]
        fake_images = generator.predict(inputs)
        x = np.concatenate((real_images, fake_images))
        y = np.ones((2*batch_size, 1))
        y[batch_size:, :] = 0
        code1 = np.concatenate((real_code1, fake_code1))
        code2 = np.concatenate((real_code2, fake_code2))
        labels = np.concatenate((real_labels, fake_labels))
        outputs = [y, labels, code1, code2]
        metrics = discriminator.train_on_batch(x, outputs)
        # metrics = ['loss', 'activation_1_loss', 'label_loss',
        # 'code1_loss', 'code2_loss', 'activation_1_acc',
        # 'label_acc', 'code1_acc', 'code2_acc']
        # from discriminator.metrics_names
        fmt = "%d: [discriminator loss: %f, label_acc: %f]"
        log = fmt % (i, metrics[0], metrics[6])

        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])
        fake_code1 = np.random.normal(scale=0.5, size=[batch_size, 1])
        fake_code2 = np.random.normal(scale=0.5, size=[batch_size, 1])
        fake_labels = np.eye(num_labels)[
            np.random.choice(num_labels, batch_size)]
        y = np.ones((batch_size, 1))
        x = [noise, fake_labels, fake_code1, fake_code2]
        y = [y, fake_labels, fake_code1, fake_code2]
        metrics = adversarial.train_on_batch(x, y)
        fmt = "%s [adversarial loss: %f, label_acc: %f]"
        log = fmt % (log, metrics[0], metrics[6])
        print(log)
        if (i + 1) % save_interval == 0:
            show = True
            gan.plot_images(generator, noise_input=noise_input, noise_label=noise_labels, noise_codes=[noise_code1, noise_code2],
                            show=show, step=(i + 1), model_name=model_name)
    generator.save(model_name + ".h5")


def mi_loss(c, q_of_c_given_x):
    # mi_loss = -c * log(Q(c|x))
    return k.mean(-k.sum(k.log(q_of_c_given_x + k.epsilon())*c, axis=1))



def build_and_train(latent_size=100):
    batch_size = 64
    model_name = 'infogan_mnist'
    train_steps = 40000
    lr = 2e-4
    decay = 6e-8
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_train = x_train.astype('float32') / 255
    input_shape = [image_size, image_size, 1]
    num_labels = len(np.unique(y_train))
    y_train = to_categorical(y_train)
    codes_shape = (1,)

    inputs = Input(shape=input_shape)
    discriminator = gan.discriminator(
        inputs, image_size, num_labels=num_labels, num_codes=2)
    loss = ['binary_crossentropy', 'categorical_crossentropy', mi_loss, mi_loss]
    optimizer = RMSprop(lr=lr, decay=decay)
    loss_weights = [1.0, 1.0, 0.5, 0.5]
    discriminator.compile(loss=loss, loss_weights=loss_weights,
                          optimizer=optimizer, metrics=['accuracy'])
    discriminator.summary()

    inputs = Input(shape=(latent_size,))
    labels = Input(shape=(num_labels,))
    codes1 = Input(shape=codes_shape)
    codes2 = Input(shape=codes_shape)
    generator = gan.generator(
        inputs, image_size, labels=labels, codes=[codes1, codes2])
    generator.summary()

    discriminator.trainable = False
    optimizer = RMSprop(lr=lr*0.5, decay=decay*0.5)
    inputs = [inputs, labels, codes1, codes2]
    adversarial = Model(inputs, discriminator(generator(inputs)))
    adversarial.compile(loss=loss, loss_weights=loss_weights,
                        optimizer=optimizer, metrics=['accuracy'])
    adversarial.summary()

    models = (generator, discriminator, adversarial)
    data = (x_train, y_train)
    params = (batch_size, latent_size, train_steps, num_labels, model_name)
    train(models, params, data)


def test_generator(generator, params, latent_size=100):
    label, code1, code2, p1, p2 = params
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, latent_size])
    if label is None:
        num_labels = 10
        noise_labels = np.eye(num_labels)[np.random.choice(num_labels, 16)]
    else:
        noise_labels = np.zeros((16, 10))
        noise_labels[:, label] = 1
        step = label
    if code1 is None:
        code1 = np.random.normal(0.5, size=[16, 1])
    else:
        if p1:
            a = np.linspace(-2, 2, 16)
            a = a.reshape((16, 1))
            noise_code1 = np.ones((16, 1))*a
        else:
            noise_code1 = np.ones((16, 1))*code1
    print(noise_code1)
    if code2 is None:
        noise_code2 = np.random.normal(scale=0.5, size=[16, 1])
    else:
        if p2:
            a = np.linspace(-2, 2, 16)
            a = a.reshape((16, 1))
            noise_code2 = np.ones((16, 1))*a
        else:
            noise_code2 = np.ones((16, 1))*code2
    print(code2)
    gan.plot_images(generator, noise_input=noise_input, noise_label=noise_labels, noise_codes=[noise_code1, noise_code2],
                    show=True, step=step, model_name='test_ouputs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load generator h5 model with trained weights"
    parser.add_argument("-g", "--generator", help=help_)
    help_ = "Specify a specific digit to generate"
    parser.add_argument("-d", "--digit", type=int, help=help_)
    help_ = "Specify latent code 1"
    parser.add_argument("-a", "--code1", type=float, help=help_)
    help_ = "Specify latent code 2"
    parser.add_argument("-b", "--code2", type=float, help=help_)
    help_ = "Plot digits with code1 ranging fr -n1 to +n2"
    parser.add_argument("--p1", action='store_true', help=help_)
    help_ = "Plot digits with code2 ranging fr -n1 to +n2"
    parser.add_argument("--p2", action='store_true', help=help_)
    args = parser.parse_args()
    if args.generator:
        generator = load_model(args.generator)
        label = args.digit
        code1 = args.code1
        code2 = args.code2
        p1 = args.p1
        p2 = args.p2
        params = (label, code1, code2, p1, p2)
        test_generator(generator, params, latent_size=62)
    else:
        build_and_train(latent_size=62)
