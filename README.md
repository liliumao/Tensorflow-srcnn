# Tensorflow-srcnn
A tensorflow based super-resolution convolutional neural network model. Distributed learning supported.

# Files
srcnn.py: Single computer running file.

d_srcnn.py: Distributed training file.

tensor_srcnn.py: SRCNN model file.

predict.py: Put the test image into the network and get the gray(light) image as result.

generate_train.m: Caffe model original training data generation file.

generate_test.m: Caffe model original training data generation file.

generate_train_im.m: Training data generation file for ImageNet images with control of the output size.

# Details
Executing command follows the guide here: https://www.tensorflow.org/versions/r0.9/how_tos/distributed/index.html#putting-it-all-together-example-trainer-program. Also, tensorflow should be installed.

The input data file is a single hdf5 file called train.h5 by default. Changes is needed when large training data is used.

For testing, one can compute psnr of the output of predict.py and the Y channel of the original picture.
