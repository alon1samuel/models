# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Edited by: Alon Samuel - Razor LTD.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import os

from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

import tensorflow.contrib.slim as slim

tf.app.flags.DEFINE_integer(
    'batch_size', 100, 'The number of samples in each batch.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'mobilenet_v2', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
                                'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

tf.app.flags.DEFINE_string(
    'save_frozen_folder_path', "/tmp/tfmodel/", 'Path for saving frozen graph.')

tf.app.flags.DEFINE_integer(
    'num_classes', 2, 'Number of classes that the model predict from.')

tf.app.flags.DEFINE_integer(
    'image_height', None, 'Image height in pixels.')

tf.app.flags.DEFINE_integer(
    'image_width', None, 'Image width in pixels.')

tf.app.flags.DEFINE_integer(
    'num_channels', 3, 'Number of channels for image last dimension.')

tf.app.flags.DEFINE_string(
    'output_layer_name', 'MobilenetV2/Predictions/Reshape', 'The name of the last layer of the model.')

tf.app.flags.DEFINE_string(
    'model_name_nodes', 'MobilenetV2', 'Prefix for model variables.')


FLAGS = tf.app.flags.FLAGS


def main(_):

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():

        ####################
        # Select the model #
        ####################
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(FLAGS.num_classes - FLAGS.labels_offset),
            is_training=False)

        # Input placeholder instead of dataset
        image = tf.placeholder(tf.uint8, shape=[FLAGS.image_height, FLAGS.image_width, FLAGS.num_channels], name='image')

        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=False)

        eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

        image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

        image = tf.expand_dims(image, 0)

        ####################
        # Define the model #
        ####################
        logits, _ = network_fn(image)

        ###################
        #  Restore model  #
        ###################

        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

        vars = [var for var in tf.global_variables() if FLAGS.model_name_nodes in var.name]
        tf.train.Saver(vars).restore(sess, FLAGS.checkpoint_path)

        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph_def,
            [FLAGS.output_layer_name])

        frozen_folder_path = FLAGS.save_frozen_folder_path
        if not os.path.exists(frozen_folder_path):
            os.makedirs(frozen_folder_path)
        pb_name = "frozen_graph.pb"  # TODO

        with open(os.path.join(frozen_folder_path, pb_name), 'wb') as f:
            f.write(frozen_graph_def.SerializeToString())
        print("Save frozen graph to {}".format(os.path.join(frozen_folder_path, pb_name)))


if __name__ == '__main__':
    tf.app.run()
