import os
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers

path_dir_train = '/home/ubuntu/Deep-Learning/plant-classification/load_data/train/'
# path_dir_validate = '/home/ubuntu/Deep-Learning/plant-classification/load_data/'
path_dir_test = '/home/ubuntu/Deep-Learning/plant-classification/load_data/test/'

img_width = 150
img_height = 150

generator = ImageDataGenerator(rescale=1./255)
train_generator = generator.flow_from_directory(path_dir_train,
                                                target_size=(img_width, img_height),
                                                batch_size=16,
                                                class_mode='categorical')


# validation_generator = generator.flow_from_directory(
#     path_dir_validate,
#     target_size=(img_width, img_height),
#     batch_size=32,
#     class_mode='binary')

#define model
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(img_width, img_height,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(20))
model.add(Activation('softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

epochs = 10
num_train = 9982
# num_validate = 0


# model.fit_generator(
#     train_generator,
#     samples_per_epoch=num_train,
#     nb_epoch=epochs,
#     validation_data=validation_generator,
#     nb_val_samples=num_validate)

model.fit_generator(
    train_generator,
    samples_per_epoch=num_train,
    nb_epoch=epochs)