import csv

def read_lines(file_name):
  lines = []
  with open(file_name) as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
      lines.append(line)
  return lines

from sklearn.model_selection import train_test_split

file_name = './data/driving_log.csv'
samples = read_lines(file_name)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
ch, row, col = 3, 80, 320  # Trimmed image format

def flip_image(image, measurement):
  image = cv2.flip(image, 1)
  measurement = -measurement
  return image, measurement

def recovery_argument(file_dir, image_center, image_left, image_right, measurement):
  correction = 0.2
  measurement_left = measurement + correction
  measurement_right = measurement - correction
  choice = np.random.choice(3)
  if choice == 0:
    return load_image(file_dir, image_left), measurement_left
  elif choice == 1:
    return load_image(file_dir, image_right), measurement_right
  return load_image(file_dir, image_center), measurement

def load_image(image_dir, image_type):
  image_dir = image_dir+image_type
  return cv2.imread(image_dir)

def generator(samples, training, batch_size=32):
  num_samples = len(samples)
  images = np.empty([batch_size, col, row, ch])
  angles = np.empty(batch_size)
  while 1: # Loop forever so the generator never terminates
    index = 0
    shuffle(samples)
    for offset in range(0, num_samples, batch_size):
      batch_samples = samples[offset:batch_size]
      #images = np.empty([batch_size, row, col, ch]) 
     # angles = np.empty(batch_size)
      for batch_sample in batch_samples:
        file_dir = './IMG/'
        left_image = batch_sample[1].split('/')[-1]
        right_image = batch_sample[2].split('/')[-1]
        center_image = batch_sample[0].split('/')[-1]
        measurement = float(batch_sample[3])
        if training and np.random.rand() < 0.6:
          image, measurement = recovery_argument(file_dir, center_image, left_image, right_image, measurement)
        else:

          image = load_image(file_dir, center_image)

 #       image, measurement = flip_image(image, measurement)

        images[index] = image
        angles[index] = measurement
        index+=1
      # trim image to only see section with road
      X_train = np.array(images)
      y_train = np.array(angles)
      yield sklearn.utils.shuffle(X_train, y_train)

from keras.models import Sequential, Model
from keras.layers import Flatten, Lambda, Cropping2D, Dense, Conv2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# compile and train the model using the generator function
train_generator = generator(train_samples, True, batch_size=32)
validation_generator = generator(validation_samples, False, batch_size=32)

#ch, row, col = 3, 80, 320  # Trimmed image format

def LeNet():
  model = Sequential()
  # Preprocess incoming data, centered around zero with small standard deviation 
  model.add(Lambda(lambda x: x/127.5 - 0.5, input_shape=(None, 320, 80, 3)))

  model.add(Convolution2D(6,5,5, activation='relu'))
  model.add(MaxPooling2D())
  model.add(Convolution2D(6,5,5, activation='relu'))
  model.add(MaxPooling2D())
  model.add(Flatten())
  model.add(Dense(120))
  model.add(Dense(84))
  model.add(Dense(1))

def E2ENet():
  model = Sequential()
  model.add(Lambda(lambda x: ((x/255.0) - 0.5), input_shape=(col, row, ch)))
  model.add(Cropping2D(cropping=((50, 20), (0,0))))
  model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
  model.add(Conv2D(36, (5, 5), strides=(2,2), activation='relu'))
  model.add(Conv2D(48, (5, 5), strides=(2,2), activation='relu'))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(Flatten())
  model.add(Dense(100))
  model.add(Dense(50))
  model.add(Dense(10))
  model.add(Dense(1))

#E2ENet()
  model.compile(loss='mse', optimizer='adam')
  model.fit_generator(train_generator, samples_per_epoch=
            len(train_samples), validation_data=validation_generator,
            nb_val_samples=len(validation_samples), nb_epoch=3)
  model.save('model2.h5')
E2ENet()
"""
def process_file(lines):
  data = {}
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
  data['images'] = images
  data['measurements'] = measurements

def augment_images_direction(images, angle):
  new_images = []
  pass

def augment_steering_angle(measurements):
  pass
"""
