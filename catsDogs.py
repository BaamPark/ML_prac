import cv2
import os
import numpy as np
import matplotlib as plt
def load_image(file_path):
    return cv2.imread(file_path)
def preprocess_image(img, side=96):
    min_side = min(img.shape[0], img.shape[1]) #img.shape[0]가 row, image shape[1]가 coulmn.
    img = img[:min_side, :min_side] # setting aspect ratio as square
    img = cv2.resize(img, (side,side)) #resize 96 by 96
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #grayscalize
    return img / 255.0

import tensorflow as tf
print("Tensorflow:", tf.__version__)
from keras import backend as K
print(K.image_data_format())

layers = [
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), padding="same", activation=tf.nn.relu, input_shape=train_images.shape[1:]),
    tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same", activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=512, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=256, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=2, activation=tf.nn.softmax)
]

eval_images = [preprocess_image(load_image(file)) for file in "C:\\Users\\asb46\\Downloads\\dogs-vs-cats (1)\\test1\\test1"]
eval_model = tf.keras.Sequential(layers)
eval_model.load_weights("model.tf.index") #이걸로 학습된 모델을 사용
eval_predictions = eval_model.predict(np.expand_dims(eval_images, axis=-1))

cols = 4
rows = np.ceil(len(eval_images)/cols)
fig = plt.gcf()
fig.set_size_inches(cols * 4, rows * 4)
for i in range(len(eval_images)):
    plt.subplot(rows, cols, i+1)
    plt.imshow(eval_images[i], cmap="gray")
    plt.title("Dog" if np.argmax(eval_predictions[i])==1 else "Cat")
    plt.axis('off')
