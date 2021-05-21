import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras


def load_image(file_path):
    return cv2.imread(file_path)

def extract_label(file_name):     #train images의 이름에 따라 2,1,0으로 labeling 하는 function
    if "can" in file_name:
      return 2
    if "glass" in file_name:
      return 1
    else:
      return 0

train_path = "./obj/"
image_files = os.listdir(train_path) #디렉토리에 있는 이미지 파일들을 리스트화
train_images = [load_image(train_path + file) for file in image_files] #데이타 사진프레임의 개체를 opencv라이브러리를 사용하여 images를 read하고 data frame리스트화
train_labels = [extract_label(file) for file in image_files] #labelling set으로 리스트화
print(train_images)

def preprocess_image(img, side=96):            
    min_side = min(img.shape[0], img.shape[1]) 
    img = img[:min_side, :min_side] 
    img = cv2.resize(img, (side,side)) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    return img / 255.0


preview_index = 1
plt.subplot(1,2,1)
plt.imshow(train_images[preview_index])
plt.subplot(1,2,2)
plt.imshow(preprocess_image(train_images[preview_index]), cmap="gray")
plt.show()

for i in range(len(train_images)):
    train_images[i] = preprocess_image(train_images[i])

train_images = np.expand_dims(train_images, axis=-1) #expand the last dimension to be a single channel
#train_images = np.expand_dims(train_images, axis=1) #--이걸로 차이 확인 가능
#train_images = np.squeeze(train_images, axis=-1 #--squeeze는 axis를 없앰
train_labels = np.array(train_labels) #array화 시킴=열로 만듬
print(train_images.shape, train_labels.shape)



model = keras.models.load_model("model.h5")


test_path = "./test/"
test_files = os.listdir(test_path)
eval_images = [preprocess_image(test_path + file) for file in test_files]
eval_model = tf.keras.Sequential(layers)
eval_model.load_weights("model.h5") #이걸로 학습된 모델을 사용
eval_predictions = eval_model.predict(np.expand_dims(eval_images, axis=-1))

cols = 4
rows = np.ceil(len(eval_images)/cols)
fig = plt.gcf()
fig.set_size_inches(cols * 4, rows * 4)
for i in range(len(eval_images)):
    plt.subplot(rows, cols, i+1)
    plt.imshow(eval_images[i], cmap="gray")
    plt.title(np.argmax(eval_predictions[i]))
    plt.axis('off')
