# coding: utf-8

#Import libraries
import csv
#For this to work on Amazon AMI, run the following (recommended: with the environment activated)
# $ conda install opencv
# $ conda install keras
import cv2
import random
import numpy as np
from keras.layers import Lambda, Dense, Convolution2D, Cropping2D, Flatten, Activation
from keras.models import Model, Sequential
from keras.utils import plot_model
import keras
print(keras.__version__)

#Dataset
#$ wget https://s3.amazonaws.com/video.udacity-data.com/topher/2016/December/584f6edd_data/data.zip
#$ unzip data.zip
# read CSV recording data
lines = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines[1:-1]:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    batch_sample = line
    center_name = '../data/IMG/'+batch_sample[0].split('/')[-1]
    center_image = cv2.imread(center_name)
    center_image = cv2.cvtColor(center_image, cv2.COLOR_RGB2YUV)
    left_name = '../data/IMG/'+batch_sample[1].split('/')[-1]
    left_image = cv2.imread(left_name)
    left_image = cv2.cvtColor(left_image, cv2.COLOR_RGB2YUV)
    right_name = '../data/IMG/'+batch_sample[2].split('/')[-1]
    right_image = cv2.imread(right_name)
    right_image = cv2.cvtColor(right_image, cv2.COLOR_RGB2YUV)
    center_angle = float(batch_sample[3])
    #Use combined 3-camera image, not useful for simulator autonomous mode
    #image = np.concatenate((left_image[70:135, 0:], center_image[70:135, 0:],right_image[70:135, 0:]), axis=1)
    #image = cv2.imread(current_path)
    images.append(center_image)
    images.append(left_image)
    images.append(right_image)
    measurement = float(line[3])
    measurements.append(measurement)
    measurements.append(measurement+random.uniform(0.01, 0.2))
    measurements.append(measurement-random.uniform(0.01, 0.2))

X_train = np.array(images)
y_train = np.array(measurements)

# Generator implementation
# NOT USED because it's very slow.

# samples = []
# with open('../data/driving_log.csv') as csvfile:
#     reader = csv.reader(csvfile)
#     for line in reader:
#         samples.append(line)
# samples = samples[1:]
#
# from sklearn.model_selection import train_test_split
# train_samples, validation_samples = train_test_split(samples, test_size=0.2)
#
# import cv2
# import numpy as np
# import sklearn
#
# def generator(samples, batch_size=32):
#     num_samples = len(samples)
#     while 1: # Loop forever so the generator never terminates
#         sklearn.utils.shuffle(samples)
#         for offset in range(0, num_samples, batch_size):
#             batch_samples = samples[offset:offset+batch_size]
#
#             images = []
#             angles = []
#             for batch_sample in batch_samples:
#                 center_name = '../data/IMG/'+batch_sample[0].split('/')[-1]
#                 center_image = cv2.imread(center_name)
#                 #left_name = '../data/IMG/'+batch_sample[1].split('/')[-1]
#                 #left_image = cv2.imread(left_name)
#                 #right_name = '../data/IMG/'+batch_sample[2].split('/')[-1]
#                 #right_image = cv2.imread(right_name)
#                 center_angle = float(batch_sample[3])
#                 #image = np.concatenate((left_image, center_image,right_image), axis=1)
#                 images.append(center_image)
#                 angles.append(center_angle)
#
#             # trim image to only see section with road
#             X_train = np.array(images)
#             y_train = np.array(angles)
#             yield sklearn.utils.shuffle(X_train, y_train)
#
# # compile and train the model using the generator function
# train_generator = generator(train_samples, batch_size=500)
# validation_generator = generator(validation_samples, batch_size=500)

#NVIDIA CNN
model = Sequential()
#Normalization
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))

model.add(Flatten())

model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))

model.add(Dense(10))

model.add(Dense(1))


#loss='mse'
#optimizer='adam'
model.compile(loss='mse', optimizer='adam')

print(model.summary())
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
input("Press Enter to continue...")

#validation_split=0.2
#shuffle=True
model.fit(X_train, y_train, batch_size=500, epochs=10, verbose=1, shuffle=True, validation_split=0.2)

#model.fit_generator(train_generator, samples_per_epoch=len(train_samples),max_queue_size=500,workers=10, use_multiprocessing=True,epochs=10,validation_data=validation_generator,validation_steps=len(validation_samples))
model.save('model_updated.h5')


