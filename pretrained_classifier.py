from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils import model_to_dot
import matplotlib.pyplot as plt

path_dir_train = '/home/ubuntu/Deep-Learning/plant-classification/load_data/train/'
path_dir_validate = '/home/ubuntu/Deep-Learning/plant-classification/load_data/validation/'
path_dir_test = '/home/ubuntu/Deep-Learning/plant-classification/load_data/test/'

img_width = 150
img_height = 150

generator = ImageDataGenerator(rescale=1./255)

train_generator = generator.flow_from_directory(
    path_dir_train,
    target_size=(img_width, img_height),
    batch_size=16,
    class_mode='categorical')


validation_generator = generator.flow_from_directory(
    path_dir_validate,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical')

base_model=MobileNet(weights='imagenet',include_top=False)

model=Model(inputs=base_model.input,outputs=preds)

for layer in model.layers[:19]:
    layer.trainable=False
for layer in model.layers[19:]:
    layer.trainable=True

epochs = 10
num_train = 4994
num_validate = 1989

# visualizations
plot_model(model, to_file='model.png')

SVG(model_to_dot(model).create(prog='dot', format='svg'))

history = model.fit_generator(
    train_generator,
    samples_per_epoch=num_train,
    nb_epoch=epochs,
    validation_data=validation_generator,
    nb_val_samples=num_validate
)

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
