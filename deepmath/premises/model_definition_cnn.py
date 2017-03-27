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
"""Convolutional model with word embedded input."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import layers
import tensorflow as tf
from deepmath.premises import model


MOVING_AVERAGE_DECAY = .9999


class Model(model.Model):
  """Convolutional model for word embedded input."""

  def __init__(self, graph=None, embedding_size=256):
    self.graph = graph or tf.get_default_graph()
    self.embedding_size = embedding_size

  def make_embedding(self, x):
    with self.graph.as_default():
      x = layers.Convolution1D(256, 1,
                               border_mode='valid',
                               subsample_length=1,
                               activation='relu')(x)
      x = layers.Convolution1D(256, 5,
                               border_mode='valid',
                               subsample_length=2,
                               activation='relu')(x)
      x = layers.Convolution1D(256, 5,
                               border_mode='valid',
                               subsample_length=2,
                               activation='relu')(x)
      x = layers.Convolution1D(self.embedding_size, 5,
                               border_mode='valid',
                               activation='relu')(x)
      x = tf.reduce_max(x, reduction_indices=1)
    return x

  def conjecture_embedding(self, conjectures):
    """Compute the embedding for each of the conjectures."""
    return self.make_embedding(conjectures)

  def axiom_embedding(self, axioms):
    """Compute the embedding for each of the axioms."""
    return self.make_embedding(axioms)
