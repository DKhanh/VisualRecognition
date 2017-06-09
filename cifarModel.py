import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from random import shuffle
from datetime import timedelta
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import pickle
import prettytensor as pt

import cifar10
if not os.path.exists("./cifar_10"):
  os.makedirs("./cifar_10")
cifar10.data_path = "./cifar_10"

# cifar10.maybe_download_and_extract()
# images_train, cls_number_train, labels_train =  cifar10.load_training_data()
# images_test,  cls_number_test,  labels_test  = cifar10.load_test_data()

from cifar10 import img_size, num_channels, num_classes
img_size_cropped = 24
train_batch_size = 100
class_names = cifar10.load_class_names()
#  CLASS_NAME
# [0: 'airplane',
#  1: 'automobile',
#  2: 'bird',
#  3: 'cat',
#  4: 'deer',
#  5: 'dog',
#  6: 'frog',
#  7: 'horse',
#  8: 'ship',
#  9: 'truck']


LOGDIR = './logs/cifar_cnn/tensorboard/'
if not os.path.exists(LOGDIR):
  os.makedirs(LOGDIR)
MODEL_NAME = "./saved_model/cifar_cnn/cnn_model.ckpt"
MODEL_DIR = "./saved_model/cifar_cnn"
if not os.path.exists(MODEL_DIR):
  os.makedirs(MODEL_DIR)

with tf.name_scope('input'):
    x = tf.placeholder('float', shape=[None, img_size, img_size, num_channels], name='x')
    y_true = tf.placeholder('float', shape=[None, num_classes], name='y')

global_step = tf.Variable(initial_value=0,
                          name='global_step', trainable=False)

def pre_process_image(image, training):
  if training:
    image = tf.random_crop(image, size=[img_size_cropped, img_size_cropped, num_channels])

    image = tf.image.random_flip_left_right(image)

    # Randomly adjust hue, contrast and saturation.
    image = tf.image.random_hue(image, max_delta=0.05)
    image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

    # Limit the image pixels between [0, 1] in case of overflow.
    image = tf.minimum(image, 1.0)
    image = tf.maximum(image, 0.0)
  else:
    image = tf.image.resize_image_with_crop_or_pad(image,
                                                       target_height=img_size_cropped,
                                                       target_width=img_size_cropped)
  return image


def pre_process(images, training):
    # Use TensorFlow to loop over all the input images and call
    # the function above which takes a single image as input.
    images = tf.map_fn(lambda image: pre_process_image(image, training), images)

    return images


def convolution_neural_network(images, training):
  x_pretty = pt.wrap(images)

  # Pretty Tensor uses special numbers to distinguish between
  # the training and testing phases.
  if training:
    phase = pt.Phase.train
  else:
    phase = pt.Phase.infer

  with pt.defaults_scope(activation_fn=tf.nn.relu, phase=phase):
    y_pred, loss = x_pretty.\
      conv2d(kernel=5, depth=64, name='conv_layer1', batch_normalize=True).\
      conv2d(kernel=5, depth=64, name='conv_layer2').\
      max_pool(kernel=2, stride=2).\
      conv2d(kernel=5, depth=128, name='conv_layer3').\
      conv2d(kernel=5, depth=128, name='conv_layer4').\
      max_pool(kernel=2, stride=2).\
      conv2d(kernel=5, depth=256, name='conv_layer5').\
      conv2d(kernel=5, depth=256, name='conv_layer6', batch_normalize=True).\
      max_pool(kernel=2, stride=2).\
      flatten().\
      fully_connected(size=2048, name='fc_layer1').\
      fully_connected(size=1024, name='fc_layer2').\
      softmax_classifier(num_classes=num_classes, labels=y_true)

  return y_pred, loss

def create_network(training):
  with tf.device("/cpu:0"):
    with tf.variable_scope('network', reuse=None):
      images = x
      images = pre_process(images=images, training=training)

      y_pred, loss = convolution_neural_network(images=images, training=training)

  return y_pred, loss