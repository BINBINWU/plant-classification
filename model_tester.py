"/home/ubuntu/Deep-Learning/plant-classification/pre_pc.h5"
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.metrics import confusion_matrix,classification_report
from keras.models import load_model
from keras.models import Model
from keras.applications import MobileNet

model = load_model('/home/ubuntu/Deep-Learning/plant-classification/pre_pc.h5')
base_model=MobileNet(weights='imagenet',include_top=False)


print(model.summary())

#model1=Model(inputs=base_model.input)
i=0
for layer in model.layers[:20]:
    print(layer)
    i += 1
    print(i)

for layer in model.layers[20:]:
    print(layer)
    i += 1
    print(i)