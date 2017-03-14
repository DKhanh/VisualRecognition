import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt

IMG_SIZE = 28
LR = 1e-3
TEST_DIR = "./Test/test2/"
LOGDIR = './logs/tensorboard_out_none/'
MODEL_NAME = "./saved_model/model.ckpt"


n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100


hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                  'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                  'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                'biases':tf.Variable(tf.random_normal([n_classes])),}

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

test_data()

def neural_network_model(data):
	w1 = hidden_1_layer['weights']
	b1 = hidden_1_layer['biases']
	pre_l1 = tf.add(tf.matmul(data, w1), b1)
	l1 = tf.nn.relu(pre_l1)


	w2 = hidden_2_layer['weights']
	b2 = hidden_2_layer['biases']
	pre_l2 = tf.add(tf.matmul(l1, w2), b2)
	l2 = tf.nn.relu(pre_l2)

	w3 = hidden_3_layer['weights']
	b3 = hidden_3_layer['biases']
	pre_l3 = tf.add(tf.matmul(l2, w3), b3)
	l3 = tf.nn.relu(pre_l3)

	w_out = output_layer['weights']
	b_out = output_layer['biases']
	output = tf.add(tf.matmul(l3, w_out), b_out)

	return output

saver = tf.train.Saver()


def test_model():
    test_data = np.load('./Test/test_data.npy')
    fig = plt.figure()

    sess = tf.InteractiveSession()
    saver.restore(sess, MODEL_NAME)
    prediction_test = neural_network_model(x)  
    accuracy_test =  tf.argmax(neural_network_model(x), 1)

    for num, data in enumerate(test_data[:24]):
      img_data = data[0]

      sub_plot = fig.add_subplot(4, 6, num+1)
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
    