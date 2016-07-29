from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import os

import tensor_srcnn
import tensorflow as tf

# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

tf.app.flags.DEFINE_string('log_dir', '/tmp/train_logs3', 'Directory for storing variables')
tf.app.flags.DEFINE_string('train_data_dir', 'train.h5', 'Directory for storing data')

tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('num_iter', 6000,
                            """Number of iterations to run.""")

FLAGS = tf.app.flags.FLAGS

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
    is_chief = (FLAGS.task_index == 0)
    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

      images = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size, FLAGS.image_size, FLAGS.num_channels))
      labels = tf.placeholder(tf.float32, shape=(None, FLAGS.label_size, FLAGS.label_size, FLAGS.num_channels))

      outputs, weight_parameters, bias_parameters = tensor_srcnn.inference(images)

      global_step = tf.Variable(0)

      loss = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(labels, outputs))))
      # print (outputs.get_shape())
      # train_op1 = tf.train.GradientDescentOptimizer(0.0001).minimize(loss, global_step = global_step)
      train_op = tensor_srcnn.train(loss, weight_parameters, bias_parameters, global_step)

      saver = tf.train.Saver()
      summary_op = tf.merge_all_summaries()
      init_op = tf.initialize_all_variables()

    # Create a "supervisor", which oversees the training process.
    sv = tf.train.Supervisor(is_chief=is_chief,
                             logdir=FLAGS.log_dir,
                             init_op=init_op,
                             summary_op=summary_op,
                             saver=saver,
                             global_step=global_step,
                             save_model_secs=600)


    train_data, train_label = tensor_srcnn.read_training_data(FLAGS.train_data_dir)
    data_size = int(train_data.shape[0] / FLAGS.batch_size)
    # print ("total data %d" % (train_data.shape[0]))
    # print ("total data size %d" % (data_size))
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
        i = step

        if not i % 100:
          duration = time.time() - start_time
          print ('%s: step %d, duration = %.3f' %
                 (datetime.now(), i, duration))
          start_time = time.time()

        # print (i)
        if i % data_size == data_size - 1:
            batch_data = train_data[(i % data_size) * FLAGS.batch_size : , :,:,:]
            batch_label = train_label[(i % data_size) * FLAGS.batch_size : , :,:,:]
        else:
            batch_data = train_data[(i % data_size) * FLAGS.batch_size : ((i+1) % data_size) * FLAGS.batch_size, :,:,:]
            batch_label = train_label[(i % data_size) * FLAGS.batch_size : ((i+1) % data_size) * FLAGS.batch_size, :,:,:]

        # _, step = sess.run([train_op, global_step], feed_dict=train_feed)
        _,step = sess.run([train_op, global_step], feed_dict={images:batch_data, labels:batch_label})

      if is_chief:
          saver.save(sess,
                   os.path.join(FLAGS.log_dir, 'model.ckpt'),
                   global_step=global_step)
    # Ask for all the services to stop.
    sv.stop()

if __name__ == "__main__":
  tf.app.run()
