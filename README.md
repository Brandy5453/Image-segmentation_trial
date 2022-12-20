# Image-segmentation_trial
# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate, Dropout

# connect google drive
from google.colab import drive
drive.mount("/content/gdrive")
# set the working directory
import os
os.chdir(r'/content/gdrive/MyDrive/LULC classification')
path = os.chdir(r'/content/gdrive/MyDrive/LULC classification/') 

# data is already randomized and split in to training / test sets. So we can go ahead and use them as it is.
x_train = np.load('/content/gdrive/MyDrive/LULC classification/x_train.npy/x_train.npy').astype('float32')
y_train= np.load('/content/gdrive/MyDrive/LULC classification/y_train.npy/y_train.npy').astype('float32')
x_test = np.load('/content/gdrive/MyDrive/LULC classification/x_test.npy/x_test.npy').astype('float32')
y_test = np.load('/content/gdrive/MyDrive/LULC classification/y_test.npy/y_test.npy').astype('float32') 

print("x_train shape", x_train.shape)
print("y_train shape", y_train.shape)
print("x_test shape", x_test.shape)
print("y_test shape", y_test.shape)

# Let's plot a sample input RGB image and output image with land cover

plt.imshow(x_test[17,:,:,:].astype('uint8'))
plt.show()

plt.imshow(y_test[17,:,:,0].astype('uint8'))
plt.show()
