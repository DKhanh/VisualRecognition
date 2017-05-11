# import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import time

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("./MNIST_data", one_hot = True)

TRAIN_DIR = './cat_dog_data/train'
TEST_DIR = './cat_dog_data/test'

LOGDIR = './logs/catVSdog_cnn/tensorboard/'
MODEL_NAME = "./saved_model/catVSdog_cnn/cnn_model.ckpt"

IMG_SIZE = 50
LR = 1e-3

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

num_input_channel = 1
n_classes = 2
batch_size = 250


# Convolutional Layer 1.
FILTER_SIZE1 = 5          # Convolution filters are 5 x 5 pixels.
NUM_FILTERS1 = 64         # There are 16 of these filters.

# Convolutional Layer 2.
FILTER_SIZE2 = 5          # Convolution filters are 5 x 5 pixels.
NUM_FILTERS2 = 64         # There are 36 of these filters.

# Convolutional Layer 3.
FILTER_SIZE3 = 5          # Convolution filters are 5 x 5 pixels.
NUM_FILTERS3 = 128         # There are 16 of these filters.

# Convolutional Layer 4.
FILTER_SIZE4 = 5          # Convolution filters are 5 x 5 pixels.
NUM_FILTERS4 = 128         # There are 36 of these filters.

# Convolutional Layer 5.
FILTER_SIZE5 = 5          # Convolution filters are 5 x 5 pixels.
NUM_FILTERS5 = 256         # There are 16 of these filters.

# Convolutional Layer 6.
FILTER_SIZE6 = 5          # Convolution filters are 5 x 5 pixels.
NUM_FILTERS6 = 256         # There are 36 of these filters.

# Convolutional Layer 7.
FILTER_SIZE7 = 5          # Convolution filters are 5 x 5 pixels.
NUM_FILTERS7 = 32         # There are 16 of these filters.

# Convolutional Layer 8.
FILTER_SIZE8 = 5          # Convolution filters are 5 x 5 pixels.
NUM_FILTERS8 = 36         # There are 36 of these filters.

# Fully-connected layer.
FC1_SIZE = 2048             # Number of neurons in fully-connected layer.

# Fully-connected layer.
FC2_SIZE = 1024             # Number of neurons in fully-connected layer.

def label_img(img):
    word_label = img.split('.')[-3]
    # conversion to one-hot array [cat,dog]
    #                            [much cat, no dog]
    if word_label == 'cat': return [1,0]
    #                             [no cat, very doggo]
    elif word_label == 'dog': return [0,1]

def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])

    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

def create_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_num])
        
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

# ##train_data = create_train_data()
# ##test_data = create_test_data()

# train_data = np.load('train_data.npy')
# # train = train_data[:-1000]
# test = train_data[-1000:]

# test_data = np.load('test_data.npy')

#######################################################################################
def createNewWeights(shape):
  return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def createNewBiases(length):
  return tf.Variable(tf.constant(0.1, shape=[length]))

def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def createNewConV(input,              # The previous layer
                  filter_size,        # Size of filter  
                  num_input_channel,  # Number of channel in previous layer
                  num_filter,         # Number of filter
                  use_pooling=True,
                  batch_normalize=False):
  shape = [filter_size, filter_size, num_input_channel, num_filter]

  weights = createNewWeights(shape=shape)
  biases = createNewBiases(length=num_filter)

  layer = conv2d(input, weights)

  if use_pooling:
    layer = maxpool2d(layer)

  layer = tf.nn.relu(tf.add(layer, biases))

  if batch_normalize:
    layer = tf.nn.lrn(layer, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
   
  return layer, weights, biases

def createNewFC(input,
                input_size, 
                output_size,
                use_relu=True,
                use_softmax=True):
  shape = [input_size, output_size]

  weights = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
  biases =  tf.Variable(tf.constant(0.1, shape=[output_size]))

  layer = tf.add(tf.matmul(input, weights), biases)

  if use_relu:
    layer = tf.nn.relu(layer)

  if use_softmax:
    layer = tf.nn.softmax(layer)

  return layer, weights, biases


def flattenLayer(layer):
  layer_shape = layer.get_shape()
  num_features = layer_shape[1:4].num_elements()
  layer_flat = tf.reshape(layer, [-1, num_features])
  return layer_flat, num_features

with tf.name_scope('input'):
    x = tf.placeholder('float', [None, IMG_SIZE*IMG_SIZE], name='x')
    y = tf.placeholder('float', name='y')

def convolution_neural_network(x):
  with tf.device("/gpu:0"):
  # with tf.device("/cpu:0"):
    x = tf.reshape(x, shape=[-1, IMG_SIZE, IMG_SIZE, 1])

    with tf.name_scope('pattern1'):
      with tf.name_scope('ConvLayer1'):
        conv_layer1, W_conv1, b_conv1 = createNewConV(x, FILTER_SIZE1, num_input_channel, NUM_FILTERS1, use_pooling=False, batch_normalize=False)
      with tf.name_scope('ConvLayer2'):
        conv_layer2, W_conv2, b_conv2 = createNewConV(conv_layer1, FILTER_SIZE2, NUM_FILTERS1, NUM_FILTERS2, use_pooling=True, batch_normalize=False)

    with tf.name_scope('pattern2'):
      with tf.name_scope('ConvLayer3'):
        conv_layer3, W_conv3, b_conv3 = createNewConV(conv_layer2, FILTER_SIZE3, NUM_FILTERS2, NUM_FILTERS3, use_pooling=False, batch_normalize=False)
      with tf.name_scope('ConvLayer4'):
        conv_layer4, W_conv4, b_conv4 = createNewConV(conv_layer3, FILTER_SIZE4, NUM_FILTERS3, NUM_FILTERS4, use_pooling=True, batch_normalize=False)

    with tf.name_scope('pattern3'):
      with tf.name_scope('ConvLayer5'):
        conv_layer5, W_conv5, b_conv5 = createNewConV(conv_layer4, FILTER_SIZE5, NUM_FILTERS4, NUM_FILTERS5, use_pooling=False, batch_normalize=False)
      with tf.name_scope('ConvLayer6'):
        conv_layer6, W_conv6, b_conv6 = createNewConV(conv_layer5, FILTER_SIZE6, NUM_FILTERS5, NUM_FILTERS6, use_pooling=True, batch_normalize=True)

    with tf.name_scope('fc_pattern'):
      with tf.name_scope('FC1'):
        layer_flat, num_features = flattenLayer(conv_layer6) 
        fc_layer1, W_fc1, b_fc1 = createNewFC(layer_flat, num_features, FC1_SIZE, use_relu=True, use_softmax=False)
        fc_layer1 = tf.nn.dropout(fc_layer1, keep_rate)
      with tf.name_scope('FC2'): 
        fc_layer2, W_fc2, b_fc2 = createNewFC(fc_layer1, FC1_SIZE, FC2_SIZE, use_relu=True, use_softmax=False)
        fc_layer2 = tf.nn.dropout(fc_layer2, keep_rate)

    with tf.name_scope('output'):
      output, W_out, b_out = createNewFC(fc_layer2, FC2_SIZE, n_classes, use_relu=False, use_softmax=False)

  var = [W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_conv4, b_conv4,
         W_conv5, b_conv5, W_conv6, b_conv6, 
         W_fc1, b_fc1, W_fc2, b_fc2, W_out, b_out]    
  global saver
  saver = tf.train.Saver(var)
  return output

# saver = tf.train.Saver()

tf_log = 'tf.log'

def test_model():
    # fig = plt.figure()

    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
    prediction_test = convolution_neural_network(x) 
    saver.restore(sess, MODEL_NAME) 
    prediction_test = tf.nn.softmax(prediction_test)
    test_data1 = np.load('test_data.npy')
    train_data = np.load('train_data.npy')
    test_data = train_data[-1000:]
	

    for num,data in enumerate(test_data[:12]):
        # cat: [1,0]
        # dog: [0,1]
        
        img_num = data[1]
        img_data = data[0]
        
        # y = fig.add_subplot(3,4,num+1)
        orig = img_data
        data = img_data.reshape(-1,IMG_SIZE*IMG_SIZE)
        
        model_out = sess.run(tf.argmax(prediction_test, 1), feed_dict = {x:data})
        print('===========>', sess.run(prediction_test, feed_dict={x:data}), 'model_out', model_out, 'true', img_num)
        print('-----> model out', model_out, 'number', num)

        if (model_out) == 1: str_label='Dog'
        else: str_label='Cat'

    #     y.imshow(orig,cmap='gray')
    #     plt.title(str_label +'_'+ str(num) + 'true', )
    #     y.axes.get_xaxis().set_visible(False)
    #     y.axes.get_yaxis().set_visible(False)
    # plt.show()


test_model() 