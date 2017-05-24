import cv2
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
from cifarModel import create_network, global_step, x, y_true, img_size, num_channels, class_names

import cifar10
if not os.path.exists("./cifar_10"):
  os.makedirs("./cifar_10")
cifar10.data_path = "./cifar_10"

# cifar10.maybe_download_and_extract()
# images_train, cls_number_train, labels_train =  cifar10.load_training_data()
# images_test,  cls_number_test,  labels_test  = cifar10.load_test_data()

# cls_number_train have type: [0], [3], ...
# labels_train have type: [1 0 0 0 0 0 0 0 0 0], ...

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

TEST_DIR = './testCifar'
LOGDIR = './logs/cifar_cnn/tensorboard/'
if not os.path.exists(LOGDIR):
  os.makedirs(LOGDIR)
MODEL_NAME = "./saved_model/cifar_cnn/cnn_model.ckpt"
MODEL_DIR = "./saved_model/cifar_cnn"
if not os.path.exists(MODEL_DIR):
  os.makedirs(MODEL_DIR)

def plot_images(images, cls_true=None, cls_pred=None, smooth=True):
  # assert len(images) == len(cls_true) == 9

  fig, axes = plt.subplots(3, 3)

  if cls_pred is None:
    hspace = 0.3
  else:
    hspace = 0.6
  fig.subplots_adjust(hspace=hspace, wspace=0.3)

  for i, ax in enumerate(axes.flat):
    if smooth:
      interpolation = 'spline16'
    else:
      interpolation = 'nearest'

    # Plot image
    ax.imshow(images[i, :, :, :], 
      interpolation=interpolation)

    # Name of the true class
    if cls_true is None:
      cls_true_name = 'None'
    else: 
      cls_true_name = class_names[cls_true[i]]
    # Show true and predicted classes
    if cls_pred is None:
      xlabel = "True: {0}".format(cls_true_name)
    else:
      # Name of predicted classes
      cls_pred_name = class_names[cls_pred[i]]
      xlabel = "True: {0}\n Pred: {1}".format(cls_true_name, cls_pred_name)

    # Show the classes as the label on the x-axis.
    ax.set_xlabel(xlabel)

    ax.set_xticks([])
    ax.set_yticks([])

  plt.show()

def get_test_batch(start):
  num_images = len(images_test)

  # Create a random index.
  idx = np.random.choice(num_images,
                         size=9,
                         replace=False)
  sample_images = images_test[idx, :]
  sample_true_label = labels_test[idx, :]
  sample_true_cls = cls_number_test[idx]
  return sample_images, sample_true_cls, sample_true_label

def label_img(img):
    word_label = img.split('_')[-2]
    if word_label == 'airplane': return 0
    elif word_label == 'automobile': return 1
    elif word_label == 'bird': return 2
    elif word_label == 'cat': return 3
    elif word_label == 'deer': return 4
    elif word_label == 'dog': return 5
    elif word_label == 'frog': return 6
    elif word_label == 'horse': return 7
    elif word_label == 'ship': return 8
    elif word_label == 'truck': return 9
    

def load_test_data():
  testing_data = []

  for img in tqdm(os.listdir(TEST_DIR)):
    path = os.path.join(TEST_DIR, img)
    img_label = label_img(img)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (img_size,img_size))
    testing_data.append([np.array(img), np.array(img_label)])

  shuffle(testing_data)
  np.save("./testCifar/test_data.npy", testing_data)
  return testing_data

def get_test_data():
  test_data = np.load('./testCifar/test_data.npy')
  sample_images = np.array([i[0] for i in test_data])
  sample_true_cls = np.array([i[1] for i in test_data])
  sample_images = np.reshape(sample_images, [-1, img_size, img_size, num_channels])
  num_images = len(sample_images)

  # Create a random index.
  idx = np.random.choice(num_images,
                         size=9,
                         replace=False)
  sample_images = sample_images[idx]
  sample_true_cls = sample_true_cls[idx]
  print(sample_true_cls)

  return sample_images, sample_true_cls

def test_network():
  # sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
  sess = tf.InteractiveSession()

  y_pred, _ = create_network(training=False)
  y_pred_cls = tf.argmax(y_pred, 1)
  
  correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  saver = tf.train.Saver()
  try:
      print("============================================>111 Trying to restore last checkpoint ...")

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

  # Number of images.
  # sample_images, sample_true_cls, sample_true_label = get_test_batch(1808)
  sample_images, sample_true_cls = get_test_data()
  num_images = len(sample_images)

  # print(sample_true_cls)
  sample_pred_cls = np.zeros(shape=num_images, dtype=np.int)

  feed_dict = {x: sample_images}

  # Calculate the predicted class using TensorFlow.
  sample_pred_cls = sess.run(y_pred_cls, feed_dict=feed_dict)

  plot_images(images=sample_images,
              cls_true=sample_true_cls,
              cls_pred=sample_pred_cls)


# load_test_data()
test_network()








