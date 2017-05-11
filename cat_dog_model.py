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
MODEL_NAME = "./saved_model/cat_dog/cnn_model.ckpt"

IMG_SIZE = 50
LR = 1e-3

n_classes = 2
batch_size = 10


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

# train_data = create_train_data()
# test_data = create_test_data()

train_data = np.load('train_data.npy')
test_data = np.load('test_data.npy')



#######################################################################################
with tf.name_scope('input'):
    x = tf.placeholder('float', [None, IMG_SIZE*IMG_SIZE], name='x')
    x = tf.reshape(x, shape=[-1, IMG_SIZE, IMG_SIZE, 1])
    y = tf.placeholder('float', name='y')

weights = { 'W_conv1' : tf.Variable(tf.random_normal([5,5,1,32])), 
            'W_conv2' : tf.Variable(tf.random_normal([5,5,32,64])), 
            'W_fc' : tf.Variable(tf.random_normal([13*13*64, 1024])), 
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

    with tf.name_scope('FC'):
        W_fc = weights['W_fc']
        b_fc = biases['b_fc']
        fc = tf.reshape(conv2, [-1, 13*13*64]) 
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

  prediction = convolution_neural_network(x)
  prediction = prediction[:24500,:]    
  
  with tf.name_scope('cost'):
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    tf.summary.scalar('cost', cost)
  with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer().minimize(cost)

  with tf.name_scope('accuracy'):
    with tf.name_scope('correct'):  
      correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
      accuracy_test =  tf.argmax(prediction, 1)
  tf.summary.scalar('accuracy', accuracy) 

  merged = tf.summary.merge_all()
  writer = tf.summary.FileWriter(LOGDIR, sess.graph)

  train = train_data[:-500]
  test = train_data[-500:]

  tf.global_variables_initializer().run() 

  for epoch in range(hm_epochs):
      epoch_loss = 0
      # for j in len(train):
        # int(mnist.train.num_examples/batch_size)

      epoch_x = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
      epoch_x = epoch_x[:24500, :]
      epoch_y = np.array([i[1] for i in train])

      summary, _, c = sess.run([merged ,optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
      epoch_loss += c
      writer.add_summary(summary, epoch)
      time.sleep(0.01)

      print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
      saver.save(sess, MODEL_NAME)


  test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
  test_y = [i[1] for i in test]
  print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))


train_neural_network(x)

      # epoch_loss = 0
      # i = 0

      # while i < len(train):
      #   start = i 
      #   end = i + batch_size

      #   # int(mnist.train.num_examples/batch_size)
      #   epoch_x = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
      #   epoch_x = epoch_x[start:end,:]
      #   epoch_y = np.array([i[1] for i in train])
      #   epoch_y = epoch_y[start:end,:]

      #   summary, _, c = sess.run([merged ,optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
      #   epoch_loss += c
      #   writer.add_summary(summary, epoch)
      #   time.sleep(0.01)

      #   print('Epoch', epoch, '======>',i, 'completed out of', hm_epochs, 'loss:', epoch_loss)
      #   saver.save(sess, MODEL_NAME)
      #   i+= batch_size

      # while i < len(train)/10:
      #   start = int(epoch * len(train)/10) + i
      #   end = int(epoch * len(train)/10) + batch_size + i 

      #   # int(mnist.train.num_examples/batch_size)
      #   epoch_x = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE*IMG_SIZE)
      #   epoch_x = epoch_x[start:end,:]
      #   epoch_y = np.array([i[1] for i in train])
      #   epoch_y = epoch_y[start:end,:]

      #   summary, _, c = sess.run([merged ,optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
      #   epoch_loss += c
      #   writer.add_summary(summary, i)
      #   time.sleep(0.01)

      #   print('Epoch', epoch, '======>',i, 'completed out of', hm_epochs, 'loss:', epoch_loss)
      #   saver.save(sess, MODEL_NAME)
      #   i+= batch_size


      #       train = train_data[epoch*batch_size:epoch*batch_size+batch_size]
      # # int(mnist.train.num_examples/batch_size)
      # epoch_x = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE*IMG_SIZE)
      # epoch_y = np.array([i[1] for i in train])

      # summary, _, c = sess.run([merged ,optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
      # epoch_loss += c
      # writer.add_summary(summary, epoch)
      # time.sleep(0.01)

      # print('Epoch', epoch, '======>', 'completed out of', hm_epochs, 'loss:', epoch_loss)
      # saver.save(sess, MODEL_NAME)