import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import time

TRAIN_DIR = './cat_dog_data/train'
TEST_DIR = './cat_dog_data/test'

LOGDIR = './logs/cat_dog/tensorboard/'
MODEL_NAME = "./saved_model/cat_dogV2/cnn_model.ckpt"

IMG_SIZE = 50
LR = 1e-3

n_classes = 2
batch_size = 10

def create_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_num])
        
    shuffle(testing_data)
    np.save('./cat_dog_data/test_data.npy', testing_data)
    return testing_data

# test_data = create_test_data()


with tf.name_scope('input'):
    x = tf.placeholder('float', [None, IMG_SIZE*IMG_SIZE], name='x')
    x = tf.reshape(x, shape=[-1, IMG_SIZE, IMG_SIZE, 1])
    y = tf.placeholder('float', name='y')

weights = { 'W_conv1' : tf.Variable(tf.random_normal([5,5,1,32])), 
            'W_conv2' : tf.Variable(tf.random_normal([5,5,32,64])),
            'W_conv3' : tf.Variable(tf.random_normal([5,5,64,128])), 
            'W_conv4' : tf.Variable(tf.random_normal([5,5,128,256])), 
            'W_conv5' : tf.Variable(tf.random_normal([5,5,256,128])),
            'W_conv6' : tf.Variable(tf.random_normal([5,5,128,64])), 
            'W_fc' : tf.Variable(tf.random_normal([64, 1024])), 
            'W_out' : tf.Variable(tf.random_normal([1024, n_classes]))}

biases = {'b_conv1' : tf.Variable(tf.random_normal([32])), 
          'b_conv2' : tf.Variable(tf.random_normal([64])),
          'b_conv3' : tf.Variable(tf.random_normal([128])),
          'b_conv4' : tf.Variable(tf.random_normal([256])),
          'b_conv5' : tf.Variable(tf.random_normal([128])),
          'b_conv6' : tf.Variable(tf.random_normal([64])), 
          'b_fc' : tf.Variable(tf.random_normal([1024])), 
          'b_out' : tf.Variable(tf.random_normal([n_classes]))}

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


def convolution_neural_network(x):
  # with tf.device("/gpu:0"):
  with tf.device("/cpu:0"):
    x = tf.reshape(x, shape=[-1, IMG_SIZE, IMG_SIZE, 1])

    with tf.name_scope('ConvLayer1'):
        W_conv1 = weights['W_conv1']
        b_conv1 = biases['b_conv1']
        conv1 = tf.nn.relu(tf.add(conv2d(x, W_conv1), b_conv1))
        conv1 = maxpool2d(conv1)        
            
    with tf.name_scope('ConvLayer2'):
        W_conv2 = weights['W_conv2']
        b_conv2 = biases['b_conv2']
        conv2 = tf.nn.relu(tf.add(conv2d(conv1, W_conv2), b_conv2))
        conv2 = maxpool2d(conv2)

    with tf.name_scope('ConvLayer3'):
        W_conv3 = weights['W_conv3']
        b_conv3 = biases['b_conv3']
        conv3 = tf.nn.relu(tf.add(conv2d(conv2, W_conv3), b_conv3))
        conv3 = maxpool2d(conv3)

    with tf.name_scope('ConvLayer4'):
        W_conv4 = weights['W_conv4']
        b_conv4 = biases['b_conv4']
        conv4 = tf.nn.relu(tf.add(conv2d(conv3, W_conv4), b_conv4))
        conv4 = maxpool2d(conv4)

    with tf.name_scope('ConvLayer5'):
        W_conv5 = weights['W_conv5']
        b_conv5 = biases['b_conv5']
        conv5 = tf.nn.relu(tf.add(conv2d(conv4, W_conv5), b_conv5))
        conv5 = maxpool2d(conv5)        

    with tf.name_scope('ConvLayer6'):
        W_conv6 = weights['W_conv6']
        b_conv6 = biases['b_conv6']
        conv6 = tf.nn.relu(tf.add(conv2d(conv5, W_conv6), b_conv6))
        conv6 = maxpool2d(conv6)

    with tf.name_scope('FC'):
        W_fc = weights['W_fc']
        b_fc = biases['b_fc']
        fc = tf.reshape(conv6, [-1, 64]) 
        fc = tf.nn.relu(tf.add(tf.matmul(fc, W_fc), b_fc))
        fc = tf.nn.dropout(fc, keep_rate)

    with tf.name_scope('output'):
        W_out = weights['W_out']
        b_out = biases['b_out'] 
        output = tf.add(tf.matmul(fc, W_out), b_out)

    return output

saver = tf.train.Saver()
tf_log = 'tf.log'

def test_model():
    sess = tf.InteractiveSession()
    saver.restore(sess, MODEL_NAME)
    prediction_test = convolution_neural_network(x)
    fig = plt.figure()  

    test_data = np.load('./cat_dog_data/test_data.npy')

    for num,data in enumerate(test_data[:12]):
        # cat: [1,0]
        # dog: [0,1]
        
        img_num = data[1]
        img_data = data[0]
        
        y = fig.add_subplot(3,4,num+1)
        orig = img_data
        data = img_data.reshape(-1,IMG_SIZE,IMG_SIZE,1)
        
        model_out = sess.run(tf.argmax(prediction_test, 1), feed_dict = {x:data})
        
        if np.argmax(model_out) == 1: str_label='Dog'
        else: str_label='Cat'
            
        y.imshow(orig,cmap='gray')
        plt.title(str_label)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
    plt.show()

test_model()