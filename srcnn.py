from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import h5py
import numpy as np

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('num_iter', 100,
                            """Number of iterations to run.""")
tf.app.flags.DEFINE_integer('num_channels', 1,
                            """Number of channels of the input.""")
tf.app.flags.DEFINE_integer('image_size', 33,
                            """Size of the images.""")
tf.app.flags.DEFINE_integer('label_size', 21,
                            """Size of the labels.""")

def read_data(file):
  with h5py.File(file, 'r') as hf:
    data = hf.get('data')
    label = hf.get('label')
    return np.array(data), np.array(label)

def inference(images):
  weight_parameters = []
  bias_parameters = []
  # conv1
  with tf.name_scope('conv1') as scope:
    kernel = tf.Variable(tf.random_normal([9, 9, FLAGS.num_channels, 64], dtype=tf.float32, stddev=1e-3), name='weights')
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='VALID')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope)
    weight_parameters += [kernel]
    bias_parameters += [biases]


  # conv2
  with tf.name_scope('conv2') as scope:
    kernel = tf.Variable(tf.random_normal([1, 1, 64, 32], dtype=tf.float32, stddev=1e-3), name='weights')
    conv = tf.nn.conv2d(conv1, kernel, [1, 1, 1, 1], padding='VALID')
    biases = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32), trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope)
    # parameters += [kernel, biases]
    weight_parameters += [kernel]
    bias_parameters += [biases]

  # conv3
  with tf.name_scope('conv3') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 32, 1], dtype=tf.float32, stddev=1e-3), name='weights')
    conv = tf.nn.conv2d(conv2, kernel, [1, 1, 1, 1], padding='VALID')
    biases = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32), trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(bias, name=scope)
    # parameters += [kernel, biases]
    weight_parameters += [kernel]
    bias_parameters += [biases]

  return conv3, weight_parameters, bias_parameters


def run_benchmark():

  with tf.Session() as sess:
    images = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size, FLAGS.image_size, FLAGS.num_channels))
    labels = tf.placeholder(tf.float32, shape=(None, FLAGS.label_size, FLAGS.label_size, FLAGS.num_channels))

    outputs, weight_parameters, bias_parameters = inference(images)

    global_step = tf.Variable(0)

    loss = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(labels, outputs))))
    # print (outputs.get_shape())
    # train_op1 = tf.train.GradientDescentOptimizer(0.0001).minimize(loss, global_step = global_step)
    op1 = tf.train.GradientDescentOptimizer(0.0001)
    op2 = tf.train.GradientDescentOptimizer(0.0001*0.1)
    grads = tf.gradients(loss, weight_parameters + bias_parameters)
    grads1 = grads[:len(weight_parameters)]
    grads2 = grads[len(weight_parameters):]
    train_op1 = op1.apply_gradients(zip(grads1, weight_parameters), global_step = global_step)
    train_op2 = op2.apply_gradients(zip(grads2, bias_parameters), global_step = global_step)
    train_op = tf.group(train_op1, train_op2)

    saver = tf.train.Saver(weight_parameters + bias_parameters)

    init = tf.initialize_all_variables().run()

    train_data, train_label = read_data('train.h5')
    train_data = np.transpose(train_data, (0,2,3,1))
    train_label = np.transpose(train_label, (0,2,3,1))
    data_size = int(train_data.shape[0] / FLAGS.batch_size)
    # print (train_data.shape)
    num_steps_burn_in = 10
    step = 0
    for i in xrange(FLAGS.num_iter):
      start_time = time.time()
      batch_data = train_data[(i % data_size) * FLAGS.batch_size : ((i+1) % data_size) * FLAGS.batch_size, :,:,:]
      batch_label = train_label[(i % data_size) * FLAGS.batch_size : ((i+1) % data_size) * FLAGS.batch_size, :,:,:]
      # print (batch_label.shape)
      _,step = sess.run([train_op, global_step], feed_dict={images:batch_data, labels:batch_label})
      duration = time.time() - start_time
      # if i > num_steps_burn_in:
      if not i % num_steps_burn_in:
        saver.save(sess, 'my-model', global_step=i)
        print ('%s: step %d, duration = %.3f' %
               (datetime.now(), i, duration))


def main(_):
  run_benchmark()


if __name__ == '__main__':
  tf.app.run()
