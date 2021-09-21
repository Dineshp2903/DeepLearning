import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np



train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)

train_generator = train_datagen.flow_from_directory(
    '../input/intel-image-classification/seg_train/seg_train',
    target_size = (64,64),
    batch_size = 32,
    class_mode = 'categorical'   
)
test_data = ImageDataGenerator(rescale = 1./255)
test_generator = test_data.flow_from_directory(
    "../input/intel-image-classification/seg_test/seg_test",
    target_size = (64,64),
    batch_size = 32,
    class_mode = 'categorical'
)
  
  
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D( 48, 3,activation='relu',input_shape = [64,64,3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2,strides = 2))
cnn.add(tf.keras.layers.Conv2D(48 ,  3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2,strides = 2))
cnn.add(tf.keras.layers.Conv2D( 48 ,  3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2,strides = 2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(128, activation='relu'))
cnn.add(tf.keras.layers.Dense(64, activation='relu'))
cnn.add(tf.keras.layers.Dense(6, activation='softmax'))
  
  
  
  
cnn.compile(optimizer = "adam",loss = "categorical_crossentropy",metrics=["accuracy"])
history = cnn.fit(x=train_generator,validation_data = test_generator,epochs = 15)
  
  
epochs = 15
import matplotlib.pyplot as plt
 
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
  
img = keras.preprocessing.image.load_img(
    "../input/intel-image-classification/seg_pred/seg_pred/10005.jpg", target_size=(64, 64)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch




plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
