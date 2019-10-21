from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNet
from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils import model_to_dot
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

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
    batch_size=16,
    class_mode='categorical')

test_generator = generator.flow_from_directory(
    path_dir_test,
    target_size=(img_width, img_height),
    batch_size=16,
    class_mode='categorical'
)

# base_model=MobileNet(weights='imagenet',include_top=False)
#
# x=base_model.output
# x=GlobalAveragePooling2D()(x)
# x=Dense(1024,activation='relu')(x)
# x=Dense(1024,activation='relu')(x)
# x=Dense(512,activation='relu')(x)
# preds=Dense(10,activation='softmax')(x)
#
# model=Model(inputs=base_model.input,outputs=preds)
#
# for layer in model.layers[:20]:
#     layer.trainable=False
# for layer in model.layers[20:]:
#     layer.trainable=True
#
# epochs = 30
# num_train = 4994
# num_validate = 1989
# num_test = 1990
#
# model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#
# # visualizations
# plot_model(model, to_file='model.png')
#
# SVG(model_to_dot(model).create(prog='dot', format='svg'))
#
# history = model.fit_generator(
#     train_generator,
#     samples_per_epoch=num_train,
#     nb_epoch=epochs,
#     validation_data=validation_generator,
#     nb_val_samples=num_validate
# )

# model.predict_generator(test_generator)
# history.evaluate_generator(test_generator)
# model.save('pretrained_model.h5')
# loss, acc = model.evaluate_generator(test_generator, verbose = 0)
# print(acc*100)
# probabilities = model.predict_generator(test_generator, nb_test_samples= num_test)
# pred= model.predict_generator(test_generator, num_test // 4)
# predicted_class_indices=np.argmax(pred,axis=1)
# labels = (test_generator.class_indices)
# labels2 = dict((v,k) for k,v in labels.items())
# predictions = [labels2[k] for k in predicted_class_indices]
# print(predicted_class_indices)
# print(labels)
# print(predictions)
# print(confusion_matrix(predicted_class_indices,labels))


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
