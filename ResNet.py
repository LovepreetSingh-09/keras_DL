
import keras
import numpy as np
from keras.datasets import cifar10
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, Activation, Conv2D, Input, BatchNormalization
from keras.layers import AveragePooling2D, Flatten
from keras.utils import plot_model, to_categorical
from keras.layers.merge import add
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import os

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train.shape
y_test.shape

plt.imshow(x_train[1])
fig, ax = plt.subplots(2, 5, figsize=(20, 10))
for i, a in zip(range(10), ax.ravel()):
    a.imshow(x_train[i])
plt.show()

batch_size = 32
data_augmentation = True
num_classes = 10
epochs = 200

subtract_pixel_mean = True

n = 3
version = 1

if version == 1:
    depth = n*6+2
else:
    n=2
    depth = n*9+2

model_type = 'Resnet_%d_v%d' % (depth, version)

input_shape = x_train.shape[1:]
X_train = x_train.astype('float32')/255.
X_test = x_test.astype('float32')/255.

if subtract_pixel_mean:
    mean = X_train.mean(axis=0)
    X_train -= mean
    X_test -= mean

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print('No. of Training Samples : ', X_train.shape[0])
print('Shape : ', X_train.shape)
print('\nNo. of Test Samples : ', X_test.shape[0])
print('Shape : ', X_test.shape)


def lr_schedular(epoch):
    lr = 1e-03
    if epoch > 180:
        lr *= 0.5e-03
    elif epoch > 160:
        lr *= 1e-03
    elif epoch > 120:
        lr *= 1e-02
    elif epoch > 80:
        lr *= 1e-01
    print('Learning Rate : ', lr)
    return lr


def resnet_layer(inputs, num_filters=16, kernel_size=3, kernel_regularizer=l2(l=1e-04),
                 strides=1, activation='relu', batch_normalization=True, conv_first=True):
    conv = Conv2D(num_filters, kernel_size=kernel_size,
                  kernel_initializer='he_normal', strides=strides, padding='same')
    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=10):
    if (depth-2) % 6 != 0.0:
        raise ValueError('Invalid Depth for Resnet_v1 ')
    num_filters = 16
    num_res_blocks = int((depth-2)/6)
    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs)
    for stack in range(3):
        for block in range(num_res_blocks):
            strides = 1
            if stack > 0 and block == 0:
                strides = 2
            y = resnet_layer(x, num_filters=num_filters, strides=strides)
            y = resnet_layer(y, num_filters=num_filters, activation=None)
            if stack > 0 and block == 0:
                x = resnet_layer(x, strides=strides, num_filters=num_filters, activation=None,
                                 kernel_size=1, batch_normalization=False)
            x = add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2
    x = AveragePooling2D(pool_size=8)(x)
    x = Flatten()(x)
    outputs = Dense(units=num_classes, activation='softmax',
                    kernel_initializer='he_normal')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_v2(input_shape, depth, num_classes):
    if ((depth-2) % 9) != 0.0:
        raise ValueError('Invalid Depth for Resnet_v1 ')
    num_res_blocks = int((depth-2)/9)
    num_filters_in = 16
    inputs = Input(shape=input_shape)
    x = resnet_layer(
        inputs=inputs, num_filters=num_filters_in, conv_first=True)
    for stack in range(3):
        for block in range(num_res_blocks):
            strides = 1
            activation = 'relu'
            batch_normalization = True
            if stack == 0:
                num_filters_out = num_filters_in*4
                if block == 0:
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in*2
                if block == 0:
                    strides = 2
            y = resnet_layer(inputs=x, num_filters=num_filters_in, activation=activation,batch_normalization=batch_normalization,
                             strides=strides,kernel_size=1, conv_first=False)
            y = resnet_layer(
                inputs=y, num_filters=num_filters_in, conv_first=False)
            y = resnet_layer(inputs=y, kernel_size=1,
                             num_filters=num_filters_out, conv_first=False)
            if block == 0:
                x = resnet_layer(inputs=x, kernel_size=1, strides=strides, num_filters=num_filters_out,
                                 activation=None, batch_normalization=False)
            x = add([x, y])
        num_filters_in = num_filters_out
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = AveragePooling2D(pool_size=8)(x)
    x = Flatten()(x)
    outputs = Dense(units=num_classes, activation='softmax',
                    kernel_initializer='he_normal')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


if version == 2:
    model = resnet_v2(input_shape=input_shape,
                      depth=depth, num_classes=num_classes)
else:
    model = resnet_v1(input_shape=input_shape,
                      depth=depth, num_classes=num_classes)

model.compile(loss='categorical_crossentropy', optimizer=Adam(
    lr=lr_schedular(0)), metrics=['accuracy'])
model.summary()
plot_model(model, to_file='%s.png' % model_type, show_shapes=True)
print(model_type)

save_dir = os.path.join(os.getcwd(), 'keras_models')
model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

filepath = os.path.join(save_dir, model_name)

checkpoint = ModelCheckpoint(
    filepath, monitor='val_acc', verbose=1, save_best_only=True)

lr_sch = LearningRateScheduler(lr_schedular)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-06)

callbacks = [checkpoint, lr_sch, lr_reducer]

if not data_augmentation:
    print('Training without data augmentation........')
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, shuffle=True,
              callbacks=callbacks, batch_size=batch_size)
else:
    print('Using Data Augmentation.......')
    datagen = ImageDataGenerator(featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False,
                                 samplewise_std_normalization=False, zca_whitening=False, rotation_range=0,
                                 width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, vertical_flip=False)
    datagen.fit(X_train)
    model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), validation_data=(X_test, y_test), workers=5,
                        steps_per_epoch=len(X_train)//batch_size,verbose=1,epochs=epochs, callbacks=callbacks, shuffle=True)


scores = model.evaluate(X_test, y_test, batch_size=batch_size)
print('Test Loss : ', scores[0])
print('Test Accuracy : ', scores[1])
