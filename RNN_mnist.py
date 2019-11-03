
import pydot
import keras
from keras.utils import to_categorical,plot_model
from keras.layers import SimpleRNN,Dense,Activation
from keras.models import Sequential
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt

(x_train, y_train_), (x_test, y_test_) = mnist.load_data()
x_train.shape
x_test.shape

y_test_.shape

targets, count = np.unique(y_train_, return_counts=True)
targets
count

num_labels = len(np.unique(y_train_))
print('Train Labels :', dict(zip(targets, count)))
labels, counts = np.unique(y_test_, return_counts=True)
print('Test Labels :', dict(zip(labels, counts)))
c = np.where(y_train_ == 7)
print(c)
c = np.array(c).reshape(-1)
plt.figure(figsize=(4, 2))
for j, i in enumerate(c[-10:]):
    #    print(i)
    plt.subplot(2, 5, j+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.axis('off')
plt.show()

fig, ax = plt.subplots(2, 5, figsize=(20, 10))
for i, a in zip(targets, ax.ravel()):
    print(i)
    c = np.where(y_train_ == i)
    c = np.array(c).reshape(-1)
    print(c[0])
    a.imshow(x_train[c[0]], cmap='gray')
plt.show()


y_train = to_categorical(y_train_)
y_train.shape
y_test = to_categorical(y_test_)

input_size=x_train.shape[1]
X_train=x_train.reshape(-1,input_size,input_size)
X_test=x_test.reshape(-1,input_size,input_size)

input_shape=(input_size,input_size)
dropout=0.0
num_labels=len(np.unique(y_train_))
batch_size=128

model=Sequential()

model.add(SimpleRNN(units=256,input_shape=input_shape,dropout=dropout))
model.add(Dense(units=num_labels))
model.add(Activation('softmax'))

model.summary()
plot_model(model,to_file='mnist_RNN.png',show_shapes=True)

model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])

model.fit(X_train,y_train,batch_size=batch_size,epochs=10)

loss,acc=model.evaluate(X_test,y_test,batch_size=batch_size)
print('/nTest Accuracy : %.3f' % (100*acc))
