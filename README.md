# Tensorflow-srcnn
A tensorflow based super-resolution convolutional neural network model. Distributed learning supported.

# Files
srcnn.py: Single computer running file.

distributed_srcnn.py: Distributed training file.

generate_train.m: Caffe model original training data generation file.

generate_test.m: Caffe model original training data generation file.

generate_train_im.m: Training data generation file for ImageNet images with control of the output size.

# Details
Executing command follows the guide here: https://www.tensorflow.org/versions/r0.9/how_tos/distributed/index.html#putting-it-all-together-example-trainer-program. Also, tensorflow should be installed.

The model is used for training. Testing part is coming soon.

The input data file is a single hdf5 file called train.h5 by default. Changes is needed when large training data is used.
