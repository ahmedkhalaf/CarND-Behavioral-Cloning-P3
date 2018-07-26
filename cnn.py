#NVIDIA CNN
model = Sequential()
#Normalization
model.add(Lamda(lambda x: x / 255.0 - 0.5 , input_shape=(160,320,3))
model.add(Cropping2D(cropping = ((70,25),(0,0)))

model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))

model.add(Flatten())

model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))

model.add(Dense(10))
