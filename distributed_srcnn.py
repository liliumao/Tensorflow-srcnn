from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import tensorflow as tf
import h5py
import numpy as np


# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

tf.app.flags.DEFINE_string('train_data_dir', 'train.h5', 'Directory for storing data')

tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('num_iter', 2*100,
                            """Number of iterations to run.""")
tf.app.flags.DEFINE_integer('num_channels', 1,
                            """Number of channels of the input.""")
tf.app.flags.DEFINE_integer('image_size', 33,
                            """Size of the images.""")
tf.app.flags.DEFINE_integer('label_size', 21,
                            """Size of the labels.""")

FLAGS = tf.app.flags.FLAGS

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

def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")

  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":

    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

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

      saver = tf.train.Saver()
      summary_op = tf.merge_all_summaries()
      init_op = tf.initialize_all_variables()

    # Create a "supervisor", which oversees the training process.
    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                             logdir="/tmp/train_logs3",
                             init_op=init_op,
                             summary_op=summary_op,
                             saver=saver,
                             global_step=global_step,
                             save_model_secs=600)


    train_data, train_label = read_data(FLAGS.train_data_dir)
    train_data = np.transpose(train_data, (0,2,3,1))
    train_label = np.transpose(train_label, (0,2,3,1))
    data_size = int(train_data.shape[0] / FLAGS.batch_size)

    # The supervisor takes care of session initialization, restoring from
    # a checkpoint, and closing when done or an error occurs.
    with sv.managed_session(server.target) as sess:
      # Loop until the supervisor shuts down or 1000000 steps have completed.
      step = 0
      start_time = time.time()
      while not sv.should_stop() and step < FLAGS.num_iter:
        # Run a training step asynchronously.
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.
        i = step / 2

        if not i % data_size:
          duration = time.time() - start_time
          print ('%s: step %d, duration = %.3f' %
                 (datetime.now(), i, duration))
          start_time = time.time()

        batch_data = train_data[(i % data_size) * FLAGS.batch_size : ((i+1) % data_size) * FLAGS.batch_size, :,:,:]
        batch_label = train_label[(i % data_size) * FLAGS.batch_size : ((i+1) % data_size) * FLAGS.batch_size, :,:,:]

        # _, step = sess.run([train_op, global_step], feed_dict=train_feed)
        _,step = sess.run([train_op, global_step], feed_dict={images:batch_data, labels:batch_label})

    # Ask for all the services to stop.
    sv.stop()

if __name__ == "__main__":
  tf.app.run()