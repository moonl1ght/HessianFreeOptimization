""" Test Hessian Free Optimizer on XOR and MNIST datasets """
""" Author: MoonLight, 2018 """


import numpy as np
import tensorflow as tf
from hfoptimizer import HFOptimizer
from tensorflow.examples.tutorials.mnist import input_data


np.set_printoptions(suppress=True)


""" Run example on MNIST or XOR """
DATASET = 'MNIST'


def example_XOR():
  x = tf.placeholder(tf.float64, shape=[4,2], name='input')
  y = tf.placeholder(tf.float64, shape=[4,1], name='output')

  with tf.name_scope('ffn'):
    W_1 = tf.Variable([[3.0, 5.0],[4.0, 7.0]], dtype=tf.float64, name='weights_1')
    b_1 = tf.Variable(tf.zeros([2], dtype=tf.float64), name='bias_1')
    y_1 = tf.sigmoid(tf.matmul(x, W_1) + b_1)

    W_2 = tf.Variable([[-8.0], [7.0]], dtype=tf.float64, name='weights_2')
    b_2 = tf.Variable(tf.zeros([1], dtype=tf.float64), name='bias_2')
    y_out = tf.matmul(y_1, W_2) + b_2

    out = tf.nn.sigmoid(y_out)

  """ Log-loss cost function """
  loss = tf.reduce_mean(( (y * tf.log(out)) + 
    ((1 - y) * tf.log(1.0 - out)) ) * -1, name='log_loss')

  # train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

  XOR_X = [[0,0],[0,1],[1,0],[1,1]]
  XOR_Y = [[0],[1],[1],[0]]

  sess = tf.Session()
  hf_optimizer = HFOptimizer(sess, loss, y_out, dtype=tf.float64)

  init = tf.initialize_all_variables()
  sess.run(init)

  max_epoches = 100
  print('Begin Training')
  for i in range(max_epoches):
    feed_dict = {x: XOR_X, y: XOR_Y}
    hf_optimizer.minimize(feed_dict=feed_dict)
    if i % 10 == 0:
      print('Epoch:', i, 'cost:', sess.run(loss, feed_dict=feed_dict))
      print('Hypothesis ', sess.run(out, feed_dict=feed_dict))


def example_MNIST():
  mnist = input_data.read_data_sets("./data/")

  n_inputs = 28*28
  n_hidden1 = 300
  n_hidden2 = 100
  n_outputs = 10
  np.set_printoptions(suppress=True)
  x = tf.placeholder(tf.float64, shape=(None, n_inputs), name='input')
  t = tf.placeholder(tf.int64, shape=(None), name='target')

  """ Constructing simple neural network """
  with tf.name_scope('dnn'):
    with tf.name_scope('layer_1'):
      init_1 = tf.random_uniform((n_inputs, n_hidden1), -1.0, 1.0, dtype=tf.float64)
      w_1 = tf.Variable(init_1, name='weights_layer_1', dtype=tf.float64)
      b_1 = tf.Variable(tf.zeros([n_hidden1], dtype=tf.float64), name='bias_1', dtype=tf.float64)
      y_1 = tf.nn.sigmoid(tf.matmul(x, w_1) + b_1)

    # with tf.name_scope('layer_2'):
    #   n_inputs_2 = int(y_1.get_shape()[1])
    #   init_2 = tf.random_uniform((n_inputs_2, n_hidden2), -1.0, 1.0, dtype=tf.float64)
    #   w_2 = tf.Variable(init_2, name='weights_layer_2', dtype=tf.float64)
    #   b_2 = tf.Variable(tf.zeros([n_hidden2], dtype=tf.float64), name='bias_2', dtype=tf.float64)
    #   y_2 = tf.nn.relu(tf.matmul(y_1, w_2) + b_2)

    with tf.name_scope('out_layer'):
      n_inputs_out = int(y_1.get_shape()[1])
      init_out = tf.random_uniform((n_inputs_out, n_outputs), -1.0, 1.0, dtype=tf.float64)
      w_out = tf.Variable(init_out, name='weights_layer_out', dtype=tf.float64)
      b_out = tf.Variable(tf.zeros([n_outputs], dtype=tf.float64), name='bias_out', dtype=tf.float64)
      y_out = tf.matmul(y_1, w_out) + b_out
      y_out_sm = tf.nn.softmax(y_out)
      # y_out = tf.nn.sigmoid(y_out1)

  with tf.name_scope('loss'):
    # error = y_out - tf.one_hot(t, 10, dtype=tf.float64)
    # loss = tf.reduce_mean(tf.square(error), name="mse")
    # loss = tf.reduce_mean( tf.matmul(tf.one_hot(t, 10, dtype=tf.float64), tf.transpose(tf.log(y_out))) + tf.matmul((1-tf.one_hot(t, 10, dtype=tf.float64)) , tf.transpose(tf.log(1 - y_out)) ))
    one_hot = tf.one_hot(t, n_outputs, dtype=tf.float64)
    xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot, logits=y_out)
    loss = tf.reduce_mean(xentropy, name="loss")

  with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(tf.cast(y_out, tf.float32), t, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float64))

  n_epochs = 2
  batch_size = 50

  with tf.Session() as sess:
    """ Initializing hessian free optimizer """
    hf_optimizer = HFOptimizer(sess, loss, y_out, dtype=tf.float64, batch_size=batch_size, prec_loss=None)
    hf_optimizer.info()
    exit()
    init = tf.global_variables_initializer()
    init.run()

    for epoch in range(n_epochs):
      n_batches = mnist.train.num_examples // batch_size
      for iteration in range(n_batches):
        x_batch, t_batch = mnist.train.next_batch(batch_size)
        print(x_batch.shape)
        hf_optimizer.minimize({x: x_batch, t: t_batch})
        # if iteration == 10:
        #   exit()
        if iteration%1==0:
          print('Batch:', iteration, '/', n_batches)
          acc_train = accuracy.eval(feed_dict={x: x_batch, t: t_batch})
          acc_test = accuracy.eval(feed_dict={x: mnist.test.images,
                        t: mnist.test.labels})
          print('Loss:', sess.run(loss, {x: x_batch, t: t_batch}))
          print('T', t_batch[0])
          print('Out:', sess.run(y_out_sm, {x: x_batch, t: t_batch})[0])
          print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

      acc_train = accuracy.eval(feed_dict={x: x_batch, t: t_batch})
      acc_test = accuracy.eval(feed_dict={x: mnist.test.images,
                        t: mnist.test.labels})
      print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

class bcolors:
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'

if __name__ == '__main__':
  print('Runinng Hessian Free optimizer test on:', DATASET)
  if DATASET == 'MNIST':
    example_MNIST()
  elif DATASET == 'XOR':
    example_XOR()
  else:
    print(bcolors.FAIL + 
      'Unknown DATASET parameter, use only XOR or MNIST' + bcolors.ENDC)