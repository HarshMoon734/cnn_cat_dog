import tensorflow as tf
from zipfile import ZipFile

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from google.colab.patches import cv2_imshow

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import Adam

import cv2
import glob

dataset = '/content/dogs-vs-cats.zip'

with ZipFile(dataset, 'r') as zip:
  zip.extractall()
  print('The dataset is extracted')

import os
# counting the number of files in train folder
path, dirs, files = next(os.walk('/content/train'))
file_count = len(files)
print('Number of images: ', file_count)

file_names = os.listdir('/content/train/')

os.mkdir('/content/image resized')
original_folder = '/content/train/'
resized_folder = '/content/image resized/'

for i in range(2000):

  filename = os.listdir(original_folder)[i]
  img_path = original_folder+filename

  img = Image.open(img_path)
  img = img.resize((224, 224))
  img = img.convert('RGB')

  newImgPath = resized_folder+filename
  img.save(newImgPath)

  filenames = os.listdir('/content/image resized/')


labels = []

for i in range(2000):

  file_name = filenames[i]
  label = file_name[0:3]

  if label == 'dog':
    labels.append(1)

  else:
    labels.append(0)

image_directory = '/content/image resized/'
image_extension = ['png', 'jpg']

files = []

[files.extend(glob.glob(image_directory + '*.' + e)) for e in image_extension]

dog_cat_images = np.asarray([cv2.imread(file) for file in files])

X = dog_cat_images
Y = np.asarray(labels)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

X_train_scaled = X_train/255

X_test_scaled = X_test/255

total_outputs = len(set(Y_train).union(set(Y_test)))

model = Sequential([
    Conv2D(32,(3,3),activation = "relu",input_shape = (32,32,3)),
    MaxPooling2D(pool_size = (2,2)),
    Conv2D(32,(3,3),activation = "relu"),
    MaxPooling2D(pool_size = (2,2)),
    Conv2D(32,(3,3),activation = "relu"),
    MaxPooling2D(pool_size = (2,2)),
    Flatten(),
    Dense(300+total_outputs, activation="relu"),
    Dense(200+total_outputs,activation = "relu"),
    Dense(100+total_outputs,activation = "relu"),
    Dense(total_outputs, activation="softmax")
])

model.compile(optimizer=Adam(),
              loss = "sparse_categorical_crossentropy",
              metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=50, validation_data=(X_test, Y_test), callbacks=[early_stop])

print(model.evaluate(X_test,Y_test))