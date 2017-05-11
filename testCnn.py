import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt

IMG_SIZE = 28
LR = 1e-3

n_classes = 10
batch_size = 128

LR = 1e-3
TEST_DIR = "./Object/"
LOGDIR = './logs/tensorboard_cnn/'
MODEL_NAME = "./saved_model_cnn/cnn_model.ckpt"

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def test_data():
  testing_data = []
  for img in tqdm(os.listdir(TEST_DIR)):
    path = os.path.join(TEST_DIR, img)
    img_num = img.split('_')[0]
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
    testing_data.append([np.array(img)])

  shuffle(testing_data)
  np.save("./Test/test_data.npy", testing_data)
  return testing_data

# test_data()

with tf.name_scope('input'):
    x = tf.placeholder('float', [None, 784], name='x')
    y = tf.placeholder('float', name='y')

weights = {'W_conv1' : tf.Variable(tf.random_normal([5,5,1,32])), 
                      'W_conv2' : tf.Variable(tf.random_normal([5,5,32,64])), 
                      'W_fc' : tf.Variable(tf.random_normal([7*7*64, 1024])), 
                      'W_out' : tf.Variable(tf.random_normal([1024, n_classes]))}

biases = {'b_conv1' : tf.Variable(tf.random_normal([32])), 
                    'b_conv2' : tf.Variable(tf.random_normal([64])), 
                    'b_fc' : tf.Variable(tf.random_normal([1024])), 
                    'b_out' : tf.Variable(tf.random_normal([n_classes]))}

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


def convolution_neural_network(x):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

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

    with tf.name_scope('FC'):
        W_fc = weights['W_fc']
        b_fc = biases['b_fc']
        fc = tf.reshape(conv2, [-1, 7*7*64]) # It can be -1 instead of None
        fc = tf.nn.relu(tf.add(tf.matmul(fc, W_fc), b_fc))
        fc = tf.nn.dropout(fc, keep_rate)

    with tf.name_scope('output'):
        W_out = weights['W_out']
        b_out = biases['b_out'] 
        output = tf.add(tf.matmul(fc, W_out), b_out)

    return output

saver = tf.train.Saver()


def test_model():
    test_data = np.load('./Test/test_data.npy')
    fig = plt.figure()

    sess = tf.InteractiveSession()
    saver.restore(sess, MODEL_NAME)
    prediction_test = convolution_neural_network(x)  
    accuracy_test =  tf.argmax(convolution_neural_network(x), 1)

    for num, data in enumerate(test_data[:100]):
      img_data = data[0]

      sub_plot = fig.add_subplot(5, 20, num+1)
      orig = img_data
      data = img_data.reshape(1, IMG_SIZE*IMG_SIZE)
      model_out = sess.run(tf.argmax(prediction_test, 1), feed_dict = {x:data})
      # print(model_out, num)

      sub_plot.imshow(orig, cmap='gray')
      plt.title(model_out)
      sub_plot.get_xaxis().set_visible(False)
      sub_plot.get_yaxis().set_visible(False)

      # batch_x, batch_y = mnist.train.next_batch(num)
      # print('Accuracy:',accuracy_test.eval(feed_dict={x:batch_x}),batch_y)

    plt.show()


test_model() 
    