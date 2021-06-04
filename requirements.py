from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
import tensorflow as tf
from glob import glob
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

# re-size all the images to this, default input size for VGG16
IMAGE_SIZE = [224, 224]

train_path = '/content/drive/MyDrive/nonaugmented_dataset/train'
valid_path = '/content/drive/MyDrive/nonaugmented_dataset/test'