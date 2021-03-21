import csv
import cv2
import numpy as np
import os

# read all lines of the driving_log.csv
lines = []
hasFirst = False
with open('data/driving_log.csv') as csvFile:
    reader = csv.reader(csvFile)
    for line in reader:
        if not hasFirst:
            hasFirst = True
            continue
        lines.append(line)

# loads an image
def loadImage(filename):
    filename = filename.split(os.sep)[-1]
    current_path = 'data/IMG/' + filename
    image = cv2.cvtColor(cv2.imread(current_path), cv2.COLOR_BGR2RGB)
    return image

# flips image and steering
def getFlipped(image, steering):
    image_flipped = np.fliplr(image)
    measurement_flipped = -steering
    return [image_flipped, measurement_flipped]

# go through all the lines and create images and labels for our neural network
images = []
measurements = []
for line in lines:
    filename_center = line[0]
    filename_left = line[1]
    filename_right = line[2]
    steering_center = float(line[3])

    # create adjusted steering measurements for the side camera images
    correction = 0.15  # this is a parameter to tune
    steering_left = steering_center + correction
    steering_right = steering_center - correction

    image = loadImage(filename_center)
    images.append(image)
    measurements.append(steering_center)

    [image_flipped, measurement_flipped] = getFlipped(image, steering_center)
    images.append(image_flipped)
    measurements.append(measurement_flipped)

    image = loadImage(filename_left)
    images.append(image)
    measurements.append(steering_left)

    [image_flipped, measurement_flipped] = getFlipped(image, steering_left)
    images.append(image_flipped)
    measurements.append(measurement_flipped)

    image = loadImage(filename_right)
    images.append(image)
    measurements.append(steering_right)

    [image_flipped, measurement_flipped] = getFlipped(image, steering_right)
    images.append(image_flipped)
    measurements.append(measurement_flipped)

# for writeup
sample_image = images[0]
sample_image_flipped = images[1]
sample_image_left = images[2]
sample_image_right = images[4]
cv2.imwrite("output_images/img1.jpg", cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR))
cv2.imwrite("output_images/img1_flipped.jpg", cv2.cvtColor(sample_image_flipped, cv2.COLOR_RGB2BGR))
cv2.imwrite("output_images/img1_left.jpg", cv2.cvtColor(sample_image_left, cv2.COLOR_RGB2BGR))
cv2.imwrite("output_images/img1_right.jpg", cv2.cvtColor(sample_image_right, cv2.COLOR_RGB2BGR))

print("Number of data points: ", len(images))

X_train = np.array(images)
y_train = np.array(measurements)


from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Cropping2D

# build up the neural network
model = Sequential()
model.add(Cropping2D(cropping=((50,24), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Conv2D(24, (5, 5), strides=(2,2), activation='relu'))
model.add(Conv2D(36, (5, 5), strides=(2,2), activation='relu'))
model.add(Conv2D(48, (5, 5), strides=(2,2), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# compile and fit
model.compile(loss='mse', optimizer='adam')
callback = EarlyStopping(monitor='loss', patience=2)
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3, callbacks = [callback])

#save the trained model
model.save('model.h5')
