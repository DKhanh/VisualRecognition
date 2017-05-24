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

  fig, axes = plt.subplots(2, 2)

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
    ax.imshow(images[0, :, :, :], 
      interpolation=interpolation)

    # Name of the true class
    if cls_true is None:
      cls_true_name = 'None'
    else: 
      cls_true_name = class_names[cls_true[0]]
    # Show true and predicted classes
    if cls_pred is None:
      xlabel = "True: {0}".format(cls_true_name)
    else:
      # Name of predicted classes
      cls_pred_name = class_names[cls_pred[0]]
      xlabel = "True: {0}\n Pred: {1}".format(cls_true_name, cls_pred_name)

    # Show the classes as the label on the x-axis.
    ax.set_xlabel(xlabel)

    ax.set_xticks([])
    ax.set_yticks([])

  plt.show()


def predict(img):
  # sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
  sess = tf.InteractiveSession()

  _, loss = create_network(training=True)
  y_pred, _ = create_network(training=False)
  y_pred_cls = tf.argmax(y_pred, 1)

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
  tf.get_variable_scope().reuse == True
  feed_dict = {x:img}

  sample_pred_cls = sess.run(y_pred_cls, feed_dict=feed_dict)

  # plot_images(images=img,
  #             cls_true=None,
  #             cls_pred=sample_pred_cls)
  print(class_names[sample_pred_cls[0]])

img = cv2.imread("./testCifar/horse_car_2.jpg")
max_dimension = max(img.shape)
scale = 700/max_dimension
img = cv2.resize(img, None, fx=scale, fy=scale)

# img = img[66:350,107:350] #car
# img = img[150:550,7:450]
img = img[50:550,407:750]
# max_dimension = max(img.shape)
# scale = 700/max_dimension
# img = cv2.resize(img, None, fx=scale, fy=scale)
cv2.imshow('test', img)
# cv2.waitKey(0)
img = cv2.resize(img, (img_size,img_size))
img = np.reshape(img, [-1, img_size, img_size, num_channels])
#3,9 15,20
# cv2.imshow('test', img)
# cv2.waitKey(0)

predict(img)