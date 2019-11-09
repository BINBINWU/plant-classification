from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from IPython.display import SVG
from keras.utils import model_to_dot
import matplotlib.pyplot as plt
from keras import optimizers


path_dir_train = '/home/ubuntu/Deep-Learning/plant-classification/load_data/train/'
path_dir_validate = '/home/ubuntu/Deep-Learning/plant-classification/load_data/validation/'
path_dir_test = '/home/ubuntu/Deep-Learning/plant-classification/load_data/test/'

img_width = 128
img_height = 128

generator = ImageDataGenerator(rescale=1./255)

train_generator = generator.flow_from_directory(
    path_dir_train,
    target_size=(img_width, img_height),
    batch_size=256,
    interpolation="lanczos",
    class_mode='categorical')


validation_generator = generator.flow_from_directory(
    path_dir_validate,
    target_size=(img_width, img_height),
    batch_size=256,
    interpolation="lanczos",
    class_mode='categorical')

#print(train_generator.class_indices)
#print(train_generator.classes)

# #define model
model = Sequential()
model.add(Convolution2D(32, 5, 5, input_shape=(img_width, img_height,3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 5, 5))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.1))
#for 10 classes
#model.add(Dense(10))
#for 20 classes
model.add(Dense(20))
model.add(Activation('softmax'))

AdamOP=optimizers.adam(lr=0.001)

model.compile(optimizer=AdamOP,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

epochs = 60
num_train = 4994
num_validate = 1989

#visualizations
# plot_model(model, to_file='model.png')

# SVG(model_to_dot(model).create(prog='dot', format='svg'))

history = model.fit_generator(
    train_generator,
    #steps_per_epoch=16000,
    nb_epoch=epochs,
    validation_data=validation_generator,
    #validation_steps=800,
    callbacks=[ModelCheckpoint("/home/ubuntu/Deep-Learning/plant-classification/pc.hdf5", monitor="val_loss", save_best_only=True)]

)

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('basic_imgAG_acc_vs_val_acc.png')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.savefig('basic_imgAG_loss_vs_val_loss.png')
plt.show()