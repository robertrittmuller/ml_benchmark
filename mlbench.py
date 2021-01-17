import tensorflow as tf
import os
import platform
import timeit
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime
from tensorflow.python.compiler.mlcompute import mlcompute
from tests import test1, test2, test3, test4, display_output, how_long

# TODO: Add NLP tests!

# Apple M1 force CPU/GPU device.
# mlcompute.set_mlc_device(device_name='gpu') # Available options are 'cpu', 'gpu', and ‘any'.
# tf.compat.v1.disable_eager_execution()
os.environ["TFHUB_CACHE_DIR"] = "./.model_cache"

# get and save some environment data
envData = {}
envData['outputFileName'] = "test_results.csv"
envData['machineType'] = platform.uname().machine
envData['runDate'] = datetime.date(datetime.now())
envData['runTime'] = datetime.time(datetime.now())

# start of MNIST test -----------------------------------------------------------------
print('Starting machine learning performance tests on ' + str(envData['machineType']) + '...')

start = timeit.default_timer()
display_output(envData,'Simple Neural Network Test', test1(5))              # accepts batch size
display_output(envData,'RELU Activation Test', test2(5, 'relu'))            # accepts batch size and activation function type; relu, tanh, sigmoid
display_output(envData,'TANH Activation Test', test2(5, 'tanh'))
display_output(envData,'SIGMOID Activation Test', test2(5, 'sigmoid'))
# display_output(envData,'Resnet50 Fine Tuning Test', test3(5, feature_extractor_model="https://tfhub.dev/tensorflow/resnet_50/feature_vector/1"))
display_output(envData,'MobilenetV2 Fine Tuning Test', test3(5, feature_extractor_model="https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4")) 
display_output(envData,'Inception V3 Inference Test (Batch size = 1)', test4(1, classifier_model="https://tfhub.dev/google/imagenet/inception_v3/classification/4"))         
display_output(envData,'Inception V3 Inference Test (Batch size = 4)', test4(4, classifier_model="https://tfhub.dev/google/imagenet/inception_v3/classification/4")) 
# display_output(envData,'Resnet 50 Inference Test (Batch size = 1)', test4(1, classifier_model="https://tfhub.dev/tensorflow/resnet_50/classification/1"))
# display_output(envData,'Resnet 50 Inference Test (Batch size = 4)', test4(4, classifier_model="https://tfhub.dev/tensorflow/resnet_50/classification/1")) 
display_output(envData,'EfficientNet-B0 Inference Test (Batch size = 1)', test4(1, classifier_model="https://tfhub.dev/tensorflow/efficientnet/b0/classification/1"))
display_output(envData,'EfficientNet-B0 Inference Test (Batch size = 4)', test4(4, classifier_model="https://tfhub.dev/tensorflow/efficientnet/b0/classification/1"))
display_output(envData,'MobilenetV2 Inference Test (Batch size = 1)', test4(1, classifier_model="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"))
display_output(envData,'MobilenetV2 Inference Test (Batch size = 4)', test4(4, classifier_model="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"))

stop = timeit.default_timer()

# Final time
display_output(envData, 'Total time for tests', how_long(start, stop))