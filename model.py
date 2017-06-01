
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
train_samples, validation_samples = train_test_split(samples, test_size=0.3)

import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
ch, row, col = 3, 160, 320  # Trimmed image format

def random_shadow(image):
  
  """
  Randomly apply shadow
  """
  # (x1, y1) and (x2, y2) forms a line 

  x1,y1 = col * np.random.rand(), 0
  x2, y2 = col * np.random.rand(), row
  xm, ym = np.mgrid[0:row, 0:col]

  mask = np.zeros_like(image[:, :, 1])
  mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

  # choose which side should have shadow and adjust saturation
  cond = mask == np.random.randint(2)
  s_ratio = np.random.uniform(low=0.2, high=0.5)

  # adjust Saturation in HLS(Hue, Light, Saturation)
  hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
  hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
  return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


def random_brightness(image):
  """
  Applying random brightness to the image.
  """
  # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
  hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
  ratio = 1.0 + 0.4 * (np.random.rand() - 0.6)
  hsv[:,:,2] =  hsv[:,:,2] * ratio
  return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def flip_image(image, measurement):
  #if np.random.rand() < 0.5:
  image = cv2.flip(image, 1)
  measurement = -measurement
  return image, measurement

def image_translate(image, steering_angle):
  x = 100
  y = 10
  shift_index = 0.5
  x_train = x * (np.random.rand() - shift_index)
  Y_train = y * (np.random.rand() - shift_index)
  steering_angle += x_train * 0.002
  m_train = np.float32([[1, 0, x_train], [0, 1, Y_train]])
  height, width = image.shape[:2]
  image = cv2.warpAffine(image, m_train, (width, height))
  return image, steering_angle

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


def convert_to_yuv(image):
  return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

def argument_image_data(file_dir, center_image, left_image, right_image, measurement):
  image, measurement = recovery_argument(file_dir, center_image, left_image, right_image, measurement)
  image, measurement = flip_image(image, measurement)
  image, measurement = image_translate(image, measurement)
  image = random_shadow(image)
  image = random_brightness(image)
  return image, measurement

def generator(samples, training, batch_size=32):
  num_samples = len(samples)
  images = np.empty([batch_size, row, col, ch])
  angles = np.empty(batch_size)
  while 1: # Loop forever so the generator never terminates
    index = 0
    shuffle(samples)
    for offset in range(0, num_samples, batch_size):
      batch_samples = samples[offset:batch_size]

      for batch_sample in batch_samples:
        file_dir = './data/IMG/'
        left_image = batch_sample[1].split('/')[-1]
        right_image = batch_sample[2].split('/')[-1]
        center_image = batch_sample[0].split('/')[-1]
        measurement = float(batch_sample[3])
       # if training and np.random.rand() < 0.6:
        image, measurement = argument_image_data(file_dir, center_image, left_image, right_image, measurement)
        #else:
          #image = load_image(file_dir, center_image)
        images[index] = image
        angles[index] = measurement
        index+=1

      X_train = np.array(images)
      y_train = np.array(angles)
      yield sklearn.utils.shuffle(X_train, y_train)

from keras.models import Sequential, Model
from keras.layers import Flatten, Lambda, Cropping2D, Dense, Conv2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

# compile and train the model using the generator function
train_generator = generator(train_samples, True, batch_size=32)
validation_generator = generator(validation_samples, False, batch_size=32)


def LeNet():
  model = Sequential()
  # Preprocess incoming data, centered around zero with small standard deviation 
  model.add(Lambda(lambda x: x/255 - 0.5, input_shape=(None, 320, 80, 3)))

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
  model.add(Lambda(lambda x: ((x/127.5) - 1.0), input_shape=(row, col, ch)))
  
# Crop the image at top 65px and bottom 20px
  model.add(Cropping2D(cropping=((65, 20), (0,0))))
  
  model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
  model.add(Conv2D(36, (5, 5), strides=(2,2), activation='relu'))
  model.add(Conv2D(48, (5, 5), strides=(2,2), activation='relu'))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(Dropout(0.5))
  model.add(Flatten())
  model.add(Dense(100))
  model.add(Dense(50))
  model.add(Dense(10))
  model.add(Dense(1))

  model.compile(loss='mse', optimizer=Adam(lr=0.00001))
  model.fit_generator(train_generator, samples_per_epoch=
            len(train_samples), validation_data=validation_generator,
            nb_val_samples=len(validation_samples), nb_epoch=10)
  model.save('model.h5')

print("Running the model...")
E2ENet()
