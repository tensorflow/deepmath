# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Stash of custom layers for premise selection."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
import tensorflow as tf

K = keras.backend
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool('legacy_conv1d', False,
                  'Whether or not to use the legacy Conv1D layer.')


class Convolution1D(keras.layers.Convolution1D):
  """Emulates the behavior of the legacy Keras Conv1D layer.

  Useful to load old checkpoints.
  """
  def build(self, input_shape):
    if not FLAGS.legacy_conv1d:
      return super(Convolution1D, self).build(input_shape)

    # pylint: skip-file
    input_dim = input_shape[2]
    self.W_shape = (self.nb_filter, input_dim, self.filter_length, 1)
    self.W = self.init(self.W_shape, name='{}_W'.format(self.name))
    if self.bias:
      self.b = K.zeros((self.nb_filter,), name='{}_b'.format(self.name))
      self.trainable_weights = [self.W, self.b]
    else:
      self.trainable_weights = [self.W]
    self.regularizers = []
    if self.W_regularizer:
      self.W_regularizer.set_param(self.W)
      self.regularizers.append(self.W_regularizer)
    if self.bias and self.b_regularizer:
      self.b_regularizer.set_param(self.b)
      self.regularizers.append(self.b_regularizer)
    if self.activity_regularizer:
      self.activity_regularizer.set_layer(self)
      self.regularizers.append(self.activity_regularizer)
    self.constraints = {}
    if self.W_constraint:
      self.constraints[self.W] = self.W_constraint
    if self.bias and self.b_constraint:
      self.constraints[self.b] = self.b_constraint
    if self.initial_weights is not None:
      self.set_weights(self.initial_weights)
      del self.initial_weights

  def call(self, x, mask=None):
    if not FLAGS.legacy_conv1d:
      if keras.__version__.startswith('1.'):
        return super(Convolution1D, self).call(x, mask)
      else:
        assert mask is None, 'Keras 2 does not support mask'
        return super(Convolution1D, self).call(x)
    x = K.expand_dims(x, -1)  # add a dimension of the right
    x = K.permute_dimensions(x, (0, 2, 1, 3))
    output = K.conv2d(x, self.W, strides=self.subsample,
                      border_mode=self.border_mode,
                      dim_ordering='th')
    if self.bias:
      output += K.reshape(self.b, (1, self.nb_filter, 1, 1))
    output = K.squeeze(output, 3)  # remove the dummy 3rd dimension
    output = K.permute_dimensions(output, (0, 2, 1))
    output = self.activation(output)
    return output
