import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100


with tf.name_scope('input'):
  x = tf.placeholder('float', [None, 784], name='x')
  y = tf.placeholder('float', name='y')

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def add_layer(input, in_size, out_size, activation_function=None):
    with tf.name_scope('weights'):
      w = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
      # variable_summaries(w)

    with tf.name_scope('biases'):
      b = tf.Variable(tf.random_normal([out_size]), name='b')
      # variable_summaries(b)

    with tf.name_scope('out'):
      pre_l = tf.add(tf.matmul(input,w), b)
      if activation_function is None:
        output = pre_l
      else:
        output = activation_function(pre_l) 
      tf.summary.histogram('out', output)
    return output



def neural_network_model(data):
    with tf.name_scope('layer1'):
      l1 = add_layer(data, 784, n_nodes_hl1, activation_function=tf.nn.relu)

    with tf.name_scope('layer2'):
      l2 = add_layer(l1, n_nodes_hl1, n_nodes_hl2, activation_function=tf.nn.relu)

    with tf.name_scope('layer3'):
      l3 = add_layer(l2, n_nodes_hl2, n_nodes_hl3, activation_function=tf.nn.relu)

    with tf.name_scope('output'):
      output = add_layer(l3,n_nodes_hl3, n_classes, activation_function=None)

    return output

def train_neural_network(x):
    hm_epochs = 10
    sess = tf.InteractiveSession()
    prediction = neural_network_model(x)
    # OLD VERSION:
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:

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
    tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(LOGDIR, sess.graph)
    #sess.run(tf.global_variables_initializer())
    tf.global_variables_initializer().run()

    for epoch in range(hm_epochs):
        epoch_loss = 0
        for i in range(int(mnist.train.num_examples/batch_size)):
            epoch_x, epoch_y = mnist.train.next_batch(batch_size)
            summary, _, c = sess.run([merged ,optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
            epoch_loss += c
            writer.add_summary(summary, i)

        print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

    
    print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


train_neural_network(x)