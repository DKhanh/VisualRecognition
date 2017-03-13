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


def label_image(img):
	word_labels = img.split('_')[0]
	if word_labels   == '0': return [1,0,0,0,0,0,0,0,0,0]
	elif word_labels == '1': return [0,1,0,0,0,0,0,0,0,0]
	elif word_labels == '2': return [0,0,1,0,0,0,0,0,0,0]
	elif word_labels == '3': return [0,0,0,1,0,0,0,0,0,0]
	elif word_labels == '4': return [0,0,0,0,1,0,0,0,0,0]
	elif word_labels == '5': return [0,0,0,0,0,1,0,0,0,0]
	elif word_labels == '6': return [0,0,0,0,0,0,1,0,0,0]
	elif word_labels == '7': return [0,0,0,0,0,0,0,1,0,0]
	elif word_labels == '8': return [0,0,0,0,0,0,0,0,1,0]
	elif word_labels == '9': return [0,0,0,0,0,0,0,0,0,1]


def test_data():
	testing_data = []
	for img in tqdm(os.listdir(TEST_DIR)):
		path = os.path.join(TEST_DIR, img)
		img_num = img.split('_')[0]
		img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
		img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
		testing_data.append([np.array(img), img_num])

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

# def use_neural_network(input_data):
# 	prediction = neural_network_model(x)

# 	sess = tf.InteractiveSession()
# 	sess.run(tf.global_variables_initializer())
# 	saver.restore(sess, MODEL_NAME)

# 	result = (sess.run(tf.argmax(prediction.eval(feed_dict={x=input_data}), 1)))

# 	return result


def visualize_graph():
	test_data = np.load('./Test/test_data.npy')
	prediction = neural_network_model(x)
	fig = plt.figure()

	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())

	saver.restore(sess, MODEL_NAME)

	for num,data in enumerate(test_data[:12]):
		img_num = data[1]
		img_data = data[0]

		y = fig.add_subplot(3,4,num+1)
		orig = img_data
		data = img_data.reshape(1,IMG_SIZE*IMG_SIZE)
		model_out = (sess.run(tf.convert_to_tensor(prediction.eval(feed_dict={x:data}), dtype='float')))
		# model_out = output_data[0]
		print((model_out), num)

		# if 	  model_out[0] == 1: str_label='Num_0'
		# elif  model_out[1] == 1: str_label='Num_1'
		# elif  model_out[2] == 1: str_label='Num_2'
		# elif  model_out[3] == 1: str_label='Num_3'
		# elif  model_out[4] == 1: str_label='Num_4'
		# elif  model_out[5] == 1: str_label='Num_5'
		# elif  model_out[6] == 1: str_label='Num_6'
		# elif  model_out[7] == 1: str_label='Num_7'
		# elif  model_out[8] == 1: str_label='Num_8'
		# elif  model_out[9] == 1: str_label='Num_9'

		# print(model_out, num)

		y.imshow(orig, cmap='gray')
		plt.title(num)
		y.axes.get_xaxis().set_visible(False)
		y.axes.get_yaxis().set_visible(False)

	plt.show()

visualize_graph()