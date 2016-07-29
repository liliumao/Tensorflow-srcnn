from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import h5py
import numpy as np


tf.app.flags.DEFINE_integer('num_channels', 1,
                            """Number of channels of the input.""")
tf.app.flags.DEFINE_integer('image_size', 33,
                            """Size of the images.""")
tf.app.flags.DEFINE_integer('label_size', 21,
                            """Size of the labels.""")

FLAGS = tf.app.flags.FLAGS


def read_training_data(file):
  with h5py.File(file, 'r') as hf:
    data = np.array(hf.get('data'))
    label = np.array(hf.get('label'))
    train_data = np.transpose(data, (0,2,3,1))
    train_label = np.transpose(label, (0,2,3,1))
    return train_data, train_label

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


def train(loss, weight_parameters, bias_parameters, global_step):
  op1 = tf.train.GradientDescentOptimizer(0.0001)
  op2 = tf.train.GradientDescentOptimizer(0.0001*0.1)
  grads = tf.gradients(loss, weight_parameters + bias_parameters)
  grads1 = grads[:len(weight_parameters)]
  grads2 = grads[len(weight_parameters):]
  train_op1 = op1.apply_gradients(zip(grads1, weight_parameters), global_step = global_step)
  train_op2 = op2.apply_gradients(zip(grads2, bias_parameters))
  train_op = tf.group(train_op1, train_op2)

  return train_op
