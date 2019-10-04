from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.preprocessing import image
import numpy as np

base_model=MobileNet(weights='imagenet',include_top=False)

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x)
x=Dense(1024,activation='relu')(x)
x=Dense(512,activation='relu')(x)
preds=Dense(3,activation='softmax')(x)

model=Model(inputs=base_model.input,outputs=preds)

for layer in model.layers[:18]:
    layer.trainable=False
for layer in model.layers[18:]:
    layer.trainable=True

train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator=train_datagen.flow_from_directory('/home/ubuntu/Deep-Learning/plant-classification/load_data/train',
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])


step_size_train=train_generator.n//train_generator.batch_size
model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   epochs=5)



# img_width, img_height = 320, 240
# img = image.load_img('/home/ubuntu/Deep-Learning/plant-classification/load_data/test/test1.jpeg', target_size=(img_width, img_height))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# 
# images = np.vstack([x])
# classes = model.predict_classes(images, batch_size=10)
# print(classes)