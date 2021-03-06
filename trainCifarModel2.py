# import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from random import shuffle
from tqdm import tqdm
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

cifar10.maybe_download_and_extract()
images_train, cls_number_train, labels_train =  cifar10.load_training_data()
images_test,  cls_number_test,  labels_test  = cifar10.load_test_data()

from cifar10 import img_size, num_channels, num_classes
img_size_cropped = 24
train_batch_size = 100
test_batch_size = 500
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


LOGDIR = './logs/CIFAR-10/tensorboard/'
if not os.path.exists(LOGDIR):
  os.makedirs(LOGDIR)
MODEL_NAME = "./saved_model/cifar_cnn2/cnn_model.ckpt"
MODEL_DIR = "./saved_model/cifar_cnn2"
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

def random_batch():
    # Number of images in the training-set.
    num_images = len(images_train)

    # Create a random index.
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)

    # Use the random index to select random images and labels.
    x_batch = images_train[idx, :, :, :]
    y_batch = labels_train[idx, :]

    return x_batch, y_batch

def random_test_batch():
    # Number of images in the training-set.
    num_images = len(images_test)

    # Create a random index.
    idx = np.random.choice(num_images,
                           size=test_batch_size,
                           replace=False)

    # Use the random index to select random images and labels.
    x_batch = images_test[idx, :, :, :]
    y_batch = labels_test[idx, :]

    return x_batch, y_batch

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
  with tf.device("/gpu:0"):
    with tf.variable_scope('network', reuse=not training):
      images = x
      images = pre_process(images=images, training=training)

      y_pred, loss = convolution_neural_network(images=images, training=training)

  return y_pred, loss

def train_network(training, num_iterations):
  sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))

  with tf.name_scope('loss'):
    _, loss = create_network(training=True)
    tf.summary.scalar('loss', loss)


  y_pred, _ = create_network(training=False)

  with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, 
                                  global_step=global_step)
  
  with tf.name_scope('accuracy'):
    with tf.name_scope('correct'):
      correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      tf.summary.scalar('accuracy', accuracy)

  merged = tf.summary.merge_all()
  writer = tf.summary.FileWriter(LOGDIR, sess.graph)
  saver = tf.train.Saver()

  try:
      print("============================================> Trying to restore last checkpoint ...")

      last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=MODEL_DIR)

      # Try and load the data in the checkpoint.
      saver.restore(sess, save_path=last_chk_path)

      # If we get to this point, the checkpoint was successfully loaded.
      print("============================================> Restored checkpoint from:", MODEL_DIR)
  except:
      # If the above failed for some reason, simply
      # initialize all the variables for the TensorFlow graph.
      print("============================================> Failed to restore checkpoint. Initializing variables instead.")
      tf.global_variables_initializer().run() 

  start_time = time.time()

  for i in range(num_iterations):
    x_batch, y_batch = random_batch()
    feed_dict_train = {x: x_batch,
                       y_true: y_batch}

    summary, i_global, _ = sess.run([merged, global_step, optimizer], feed_dict=feed_dict_train)
    writer.add_summary(summary, i_global)

    if (i_global%100 == 0) or (i == num_iterations - 1):
      x_test_batch, y_test_batch = random_test_batch()
      feed_dict_test = {x: x_test_batch,
                        y_true: y_test_batch}
      batch_accuracy = sess.run(accuracy, feed_dict_test)
      print(batch_accuracy)
      if batch_accuracy > 0.88:
        saver.save(sess, save_path=MODEL_NAME, global_step=global_step)
        break
 
      # Print status.
      # print('Global Step: ', i_global, '===> Accuracy: ', accuracy)
      msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
      print(msg.format(i_global, batch_accuracy))

    # Save a checkpoint to disk every 1000 iterations (and last).
    if (i_global%1000 == 0) or (i == num_iterations-1):
      saver.save(sess, save_path=MODEL_NAME, global_step=global_step)
      print("Saved checkpoint.")

  end_time = time.time()

  time_dif = end_time - start_time

  print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

train_network(training=True, num_iterations=20000)







