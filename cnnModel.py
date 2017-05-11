import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import time

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST_data", one_hot = True)


n_classes = 10
batch_size = 500

LOGDIR = './logs/tensorboard_cnn/'
MODEL_NAME = "./saved_model/cnn_model.ckpt"

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
  # with tf.device("/gpu:0"):
  with tf.device("/cpu:0"):
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
tf_log = 'tf.log'

def train_neural_network(x):
  
  hm_epochs = 10

  # sess=tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
  sess = tf.InteractiveSession()
  # saver.restore(sess, MODEL_NAME)
  prediction = convolution_neural_network(x)    
  
  with tf.name_scope('cost'):
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    tf.summary.scalar('cost', cost)
  with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    #tf.summary.scalar('optimizer', optimizer)

  with tf.name_scope('accuracy'):
    with tf.name_scope('correct'):  
      correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
      accuracy_test =  tf.argmax(prediction, 1)
  tf.summary.scalar('accuracy', accuracy) 

  merged = tf.summary.merge_all()
  writer = tf.summary.FileWriter(LOGDIR, sess.graph)


  tf.global_variables_initializer().run() 

  for epoch in range(hm_epochs):
      epoch_loss = 0
      for i in range(int(mnist.train.num_examples/batch_size)):
        # int(mnist.train.num_examples/batch_size)
          epoch_x, epoch_y = mnist.train.next_batch(batch_size)
          summary, _, c = sess.run([merged ,optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
          epoch_loss += c
          writer.add_summary(summary, i)
          time.sleep(0.01)
          print('Epoch', epoch, '======>',i, 'completed out of', hm_epochs, 'loss:', epoch_loss, 'coss', c)

      print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
      saver.save(sess, MODEL_NAME)


  # batch_x, batch_y = mnist.train.next_batch(1)
  print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)

		



