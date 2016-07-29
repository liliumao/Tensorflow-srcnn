from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import os

import tensor_srcnn
import tensorflow as tf
from PIL import Image as im
import matplotlib.pyplot as plt
import numpy

import argparse

tf.app.flags.DEFINE_string('log_dir', '/tmp/train_logs3', 'Directory for storing variables')


FLAGS = tf.app.flags.FLAGS


def main(argv):
    file = "1.bmp"

    if file.endswith(".bmp") or file.endswith(".jpg") or file.endswith(".png"):
        image = im.open(file)
        B = numpy.asarray(image.convert('L'))
        x = numpy.expand_dims(B[:,:], axis = 2)
        x = numpy.expand_dims(x[:,:,:], axis = 0)
    else:
        print ("bad test image file")
        exit()

    images = tf.placeholder(tf.float32, shape=(1, x.shape[1], x.shape[2], FLAGS.num_channels))

    conv, weight_parameters, bias_parameters = tensor_srcnn.inference(images)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.latest_checkpoint(FLAGS.log_dir)
        if ckpt:
            saver.restore(sess, ckpt)
        else:
            print ("no checkpoint found")
            return

        out = sess.run([conv], feed_dict={images:x})


    out = out[0][0,:,:,0]


    # save the out image
    grey_image = im.fromarray(out, 'L')
    grey_image.save("out.bmp")

if __name__ == "__main__":
  tf.app.run()
