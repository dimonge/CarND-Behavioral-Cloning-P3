
import csv
import cv2
import numpy as np

lines = []
with open('./data/driving_log.csv') as csv_file: 
  reader = csv.reader(csv_file)
  for line in reader:
    lines.append(line)

images = []
measurements = []
for line in lines:
  source_path = line[0]
  filename = source_path.split('/')[-1]
  current_path = './data/IMG/' + filename
  image = cv2.imread(current_path)
  images.append(image)
  measurement = float(line[3])
  measurements.append(measurement)

  # adjust the measurement for the left and right cameras
  steering_center = measurement

X_train = np.array(images)
y_train = np.array(measurements)
"""  
  correction = 0.2
  steering_left = steering_center + correction
  steering_right = steering_center - correction
  
  # read the images
  img_center = process_image(np.asarray(Image.open(path + line[0])))
  img_left = process_image(np.asarray(Image.open(path + line[1])))
  img_right = process_image(np.asarray(Image.open(path + line[2])))

  # add image and angles to the data set
  images.extend(img_center, img_left, img_right)
  measurements.extend(steering_center, steering_left, steering_right)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D

model = Sequential()
#model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: (x / 255) - 0.5, input_shape=(160, 320, 3)))

model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)

model.save('model.h5')

# plot model history object error rate for each epoch

from keras.models import Model
import matplotlib.pyplot as plt

history_object = model.fit_generator(train_generator, samples_per_epoch = len(train_samples), validation_data = validation_generator, \
  nb_val_samples = len(validation_samples), \
  nb_epoch=5, verbose=1)

# print the keys in the history object
print(history_object.keys())

# plot the training and validation loss for each epoch

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title("model mean squared error loss")
plt.ylabel("mean squared error loss")
plt.xlabel("epoch")
plt.legend(["training set", "validation set"], loc="upper right")
plt.show()
"""

from keras.models import Sequential, Model
from keras.layers import Flatten, Lambda, Cropping2D, Dense
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255) - 0.5, input_shape=(160, 320, 3)))
model.add(Convolution2D(6,5,5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)
