
import keras
import numpy as np
from keras.datasets import cifar10
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, Activation, Conv2D, Input, BatchNormalization
from keras.layers import AveragePooling2D, Flatten,MaxPooling2D,Dropout
from keras.utils import plot_model, to_categorical
from keras.layers.merge import concatenate
from keras.optimizers import RMSprop
from keras.regularizers import l2
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import os

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train.shape
y_test.shape

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


plt.imshow(x_train[1])
fig, ax = plt.subplots(2, 5, figsize=(20, 10))
for i, a in zip(range(10), ax.ravel()):
    a.imshow(x_train[i])
plt.show()

input_shape = x_train.shape[1:]
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.


batch_size = 32
data_augmentation = True
num_classes = 10
epochs = 200


n_blocks=3
growth_size=12
use_max_pool=False
depth=100
n_layers = int((depth-4)//(2*n_blocks))
n_filters=2*growth_size
compression_factor=0.5

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

inputs=Input(shape=input_shape)
x=BatchNormalization()(inputs)
x=Activation('relu')(x)
x=Conv2D(n_filters,kernel_size=3,padding='same',kernel_initializer='he_normal')(x)
x=concatenate([inputs,x])
for block in range(n_blocks):
    for layer in range(n_layers):
        y=BatchNormalization()(x)
        y=Activation('relu')(x)
        y=Conv2D(4*growth_size,padding='same',kernel_size=1,kernel_initializer='he_normal')(y)
        if not data_augmentation:
            y=Dropout(0.2)(y)
        y=BatchNormalization()(y)
        y=Activation('relu')(y)
        y=Conv2D(growth_size,padding='same',kernel_size=3,kernel_initializer='he_normal')(y)
        if not data_augmentation:
            y=Dropout(0.2)(y)
        x=concatenate([x,y])
    if block==n_blocks-1:
        continue
    n_filters+=(n_layers)*growth_size
    n_filters=int(n_filters*compression_factor)
    x=BatchNormalization()(x)
    x=Conv2D(n_filters,kernel_size=1,padding='same',kernel_initializer='he_normal')(x)
    if not data_augmentation:
        x=Dropout(0.2)(x)
    x=AveragePooling2D()(x)

x=AveragePooling2D(pool_size=8)(x)
x=Flatten()(x)
outputs=Dense(num_classes,activation='softmax',kernel_initializer='he_normal')(x)

model=Model(inputs=inputs,outputs=outputs)
model.summary()
plot_model(model,to_file='DenseNet.png',show_shapes=True)

model.compile(loss='categorical_crossentropy',optimizer=RMSprop(1e-3),metrics=['accuracy'])

save_dir=os.path.join(os.getcwd(),'keras_models')
model_name='cifar10_densenet_model.{epoch:02d}.h5'

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

filepath=os.path.join(save_dir,model_name)

lr_sche=LearningRateScheduler(lr_schedular)
lr_reducer=ReduceLROnPlateau(factor=np.sqrt(0.1),patience=5,cooldown=0,min_lr=0.5e-6)
checkpoint=ModelCheckpoint(filepath,monitor='val_acc',save_best_only=True,verbose=1)

callbacks = [checkpoint, lr_reducer, lr_sche]

if not data_augmentation:
    print('Without Using Data Augmentation..........')
    model.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=callbacks,shuffle=True,epochs=epochs,batch_size=batch_size)
else:
    print('Using Data Augmentation..........')
    datagen = ImageDataGenerator(featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False,
                                 samplewise_std_normalization=False, zca_whitening=False, rotation_range=0,
                                 width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, vertical_flip=False)
    datagen.fit(x_train)
    model.fit_generator(datagen.flow(x_train,y_train,batch_size=batch_size),validation_data=(x_test,y_test),shuffle=True,
                        epochs=epochs,workers=6,steps_per_epoch=int(len(x_train)/batch_size),callbacks=callbacks)

scores=model.evaluate(x_test,y_test,batch_size=batch_size,verbose=1)
print('Test Loss : ',scores[0])
print('Test Acc : ',scores[1])
