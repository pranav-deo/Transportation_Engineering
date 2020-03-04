import tensorflow as tf
import cv2
import numpy as np
import time
from datetime import datetime
from keras.applications.resnet50 import ResNet50
from keras.models import Model
import keras
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import optimizers
from keras.callbacks import TensorBoard


""" Get data and preprocess it"""
datagen = ImageDataGenerator()


# inputs = tf.placeholder(tf.float32, [None, 416,416,3])
# model = nets.YOLOv3VOC(inputs)

IMG_WIDTH=300
IMG_HEIGHT=300
IMG_DIM = (IMG_WIDTH, IMG_HEIGHT)
NO_OF_CLASSES = 10


# classes={'0':'person','1':'bicycle','2':'car','3':'bike','5':'bus','7':'truck'}
# list_of_classes=[0,1,2,3,5,7]#to display other detected #objects,change the classes and list of classes to their respective #COCO indices available in their website. Here 0th index is for #people and 1 for bicycle and so on. If you want to detect all the #classes, add the indices to this list

""" Code for training """
restnet = ResNet50(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT,IMG_WIDTH,3))
output = restnet.layers[-1].output
output = keras.layers.Flatten()(output)
restnet = Model(restnet.input, output=output)
for layer in restnet.layers:
    layer.trainable = False
restnet.summary()

model = Sequential()
model.add(restnet)
model.add(Dense(512, activation='relu', input_dim=input_shape))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(NO_OF_CLASSES, activation='sigmoid'))
model.compile(loss='catergorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['accuracy'])
model.summary()

now = datetime.now()
save_name = 'idd_restnet50_{epoch:08d}' + now.strftime("%d/%m/%Y_%H-%M-%S") + '.h5' 

mc = keras.callbacks.ModelCheckpoint(save_name, save_weights_only=True, period=5)

tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))

history = model.fit_generator(train_generator, 
                              steps_per_epoch=100, 
                              epochs=100,
                              validation_data=val_generator, 
                              validation_steps=50, 
                              verbose=1,
                              callbacks = [mc, tensorboard])

now = datetime.now()
save_name = 'idd_restnet50_{epoch:08d}' + now.strftime("%d/%m/%Y_%H-%M-%S") + '.h5' 
model.save(save_name)