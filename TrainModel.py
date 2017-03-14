import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100


IMG_SIZE = 28
LR = 1e-3
TEST_DIR = "./Test/test2/"
LOGDIR = './logs/tensorboard_out_none/'
MODEL_NAME = "./saved_model/model.ckpt"



hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                  'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                  'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                'biases':tf.Variable(tf.random_normal([n_classes])),}

with tf.name_scope('input'):
  x = tf.placeholder('float', [None, 784], name='x')
  y = tf.placeholder('float', name='y')


def add_layer(input, in_size, out_size, activation_function=None):
    with tf.name_scope('weights'):
      w = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
      tf.add_to_collection('vars', w)
      tf.summary.histogram('W', w)

    with tf.name_scope('biases'):
      b = tf.Variable(tf.random_normal([out_size]), name='b')
      tf.add_to_collection('vars', b)      
      tf.summary.histogram('b', b)

    with tf.name_scope('out'):
      pre_l = tf.add(tf.matmul(input,w), b)
      if activation_function is None:
        output = pre_l
      else:
        output = activation_function(pre_l) 
      #tf.summary.histogram('out', output)
    return output



def neural_network_model(data):
    with tf.name_scope('layer1'):
      w1 = hidden_1_layer['weights']
      b1 = hidden_1_layer['biases']
      pre_l1 = tf.add(tf.matmul(data, w1), b1)
      l1 = tf.nn.relu(pre_l1)


    with tf.name_scope('layer2'):
      w2 = hidden_2_layer['weights']
      b2 = hidden_2_layer['biases']
      pre_l2 = tf.add(tf.matmul(l1, w2), b2)
      l2 = tf.nn.relu(pre_l2)

    with tf.name_scope('layer3'):
      w3 = hidden_3_layer['weights']
      b3 = hidden_3_layer['biases']
      pre_l3 = tf.add(tf.matmul(l2, w3), b3)
      l3 = tf.nn.relu(pre_l3)

    with tf.name_scope('output'):
      w_out = output_layer['weights']
      b_out = output_layer['biases']
      output = tf.add(tf.matmul(l3, w_out), b_out)

    return output

saver = tf.train.Saver()
tf_log = 'tf.log'

def train_neural_network(x):
    hm_epochs = 10
    sess = tf.InteractiveSession()
    # saver.restore(sess, MODEL_NAME)
    prediction = neural_network_model(x)    
    
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
    tf.global_variables_initializer().run() #

    for epoch in range(hm_epochs):
        epoch_loss = 0
        for i in range(int(mnist.train.num_examples/batch_size)):
          # int(mnist.train.num_examples/batch_size)
            epoch_x, epoch_y = mnist.train.next_batch(batch_size)
            summary, _, c = sess.run([merged ,optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
            epoch_loss += c
            writer.add_summary(summary, i)

        print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
        saver.save(sess, MODEL_NAME)


    batch_x, batch_y = mnist.train.next_batch(1)
    print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


train_neural_network(x)

    

