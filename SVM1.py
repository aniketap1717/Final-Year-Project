import numpy as np
import pandas as pd
import tensorflow as tf
from keras.activations import softmax
from keras.metrics import accuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D,Dense,MaxPool2D,Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras.losses import squared_hinge
from tensorflow.python.keras.optimizer_v1 import adam

# Augmentation
train_dir = 'Data2/train/'
test_dir = 'Data2/test/'

train_datagen = ImageDataGenerator(rescale=(1/255.),shear_range = 0.2,zoom_range=0.2,
                                   horizontal_flip=True)
training_set = train_datagen.flow_from_directory(directory = train_dir,target_size=(128,128),
                                                batch_size=32,
                                                class_mode = "binary")
test_datagen = ImageDataGenerator(rescale=(1/255.))
test_set = test_datagen.flow_from_directory(directory = test_dir,target_size=(128,128),
                                                batch_size=32,
                                                class_mode = "binary")

#Model Creation:

model = Sequential()
model.add(Conv2D(filters = 32, padding = "same",activation = "relu",kernel_size=3, strides = 2,input_shape=(128,128,3)))
model.add(MaxPool2D(pool_size=(2,2),strides = 2))

model.add(Conv2D(filters = 32, padding = "same",activation = "relu",kernel_size=3))
model.add(MaxPool2D(pool_size=(2,2),strides = 2))

model.add(Flatten())
model.add(Dense(128,activation="relu"))

#Output layer
model.add(Dense(1,kernel_regularizer=l2(0.01),activation = "softmax"))




# Compile model
number_of_classes=1
model.add(Dense(number_of_classes,kernel_regularizer = l2(0.01),activation= "softmax"))
model.compile(optimizer="adam",loss="squared_hinge", metrics = ['accuracy'])

# train our model

history = model.fit(x = training_set,  batch_size = 32,validation_data = test_set, epochs=25,verbose = 1)

#save model

model_dir = 'Model/model_SVM.h5'

model.save(model_dir)

from tensorflow import keras
model = keras.models.load_model(model_dir)
