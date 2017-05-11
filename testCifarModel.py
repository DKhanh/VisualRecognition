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

import cifar10
if not os.path.exists("./cifar_10"):
  os.makedirs("./cifar_10")
cifar10.data_path = "./cifar_10"

cifar10.maybe_download_and_extract()
images_train, cls_number_train, labels_train =  cifar10.load_training_data()
images_test,  cls_number_test,  labels_test  = cifar10.load_test_data()

# cls_number_train have type: [0], [3], ...
# labels_train have type: [1 0 0 0 0 0 0 0 0 0], ...

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
    with tf.variable_scope('network', reuse=not training):
      images = x
      images = pre_process(images=images, training=training)

      y_pred, loss = convolution_neural_network(images=images, training=training)

  return y_pred, loss


def plot_example_errors(cls_pred, correct):
    # This function is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)
    
    # Get the images from the test-set that have been
    # incorrectly classified.
    images = images_test[incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = cls_test[incorrect]
    
    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])

def predict_cls(images, labels, cls_true):
    batch_size = 100
    # Number of images.
    num_images = len(images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_images, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_images)

        # Create a feed-dict with the images and labels
        # between index i and j.
        feed_dict = {x: images[i:j, :],
                     y_true: labels[i:j, :]}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = sess.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    return correct, cls_pred

def predict_cls_test():
    return predict_cls(images = images_test,
                       labels = labels_test,
                       cls_true = cls_number_test)

def classification_accuracy(correct):
    # When averaging a boolean array, False means 0 and True means 1.
    # So we are calculating: number of True / len(correct) which is
    # the same as the classification accuracy.
    
    # Return the classification accuracy
    # and the number of correct classifications.
    return correct.mean(), correct.sum()

def print_test_accuracy(show_example_errors=False, show_confusion_matrix=False):

    # For all the images in the test-set,
    # calculate the predicted classes and whether they are correct.
    correct, cls_pred = predict_cls_test()
    
    # Classification accuracy and the number of correct classifications.
    acc, num_correct = classification_accuracy(correct)
    
    # Number of images being classified.
    num_images = len(correct)

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))


    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

def print_output_test():
  _, cls_pred = predict_cls_test()
  # for i in range(cls_pred[0:9]):
  #   print('True: ', cls_number_test[i])
  #   print('Predict: ', cls_pred[i])
  #   print('===================================================>')
  plot_images(images=images_test[0:9],
                cls_true=cls_number_test[0:9],
                cls_pred=cls_pred[0:9])

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

  _, loss = create_network(training=True)
  y_pred, _ = create_network(training=False)
  y_pred_cls = tf.argmax(y_pred, 1)
  optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, 
                                  global_step=global_step)
  
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








