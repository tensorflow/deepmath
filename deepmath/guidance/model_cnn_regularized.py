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
"""Convolutional model with word embedded input -- larger version.

Based on premises/model_definition_cnn_flat3.py, but with more regularization
(dropout or batch normalization).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
import tensorflow as tf
from deepmath.premises import model
from deepmath.util import model_utils

layers = tf.contrib.layers


def default_hparams():
  """Default values for model hyperparameters."""
  return tf.contrib.training.HParams(
      keep_prob=1.0,  # Keep probability for dropout (1.0 means no dropout)
      batch_norm=False,  # Whether to use batch normalization
      hidden_size=1024,  # Size of the conjecture and clause embeddings
      filter_width=5,  # Filter width
      weight_decay=0.0,  # Weight decay
  )


class Model(model.Model):
  """Convolutional model for word embedded input."""

  def __init__(self, mode, hparams):
    super(Model, self).__init__(graph=tf.get_default_graph(),
                                embedding_size=None)
    self.mode = mode
    self.hparams = hparams
    if hparams.weight_decay:
      self.regularizer = layers.l2_regularizer(hparams.weight_decay)
    else:
      self.regularizer = None

  def dropout(self, x):
    return layers.dropout(x, keep_prob=self.hparams.keep_prob,
                          is_training=self.mode == 'train')

  def make_keras_regularizer(self):
    if self.hparams.weight_decay and self.mode == 'train':
      return keras.regularizers.l2(self.hparams.weight_decay)

  def apply_keras_regularizer(self, reg):
    if reg is not None:
      tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, reg(0.0))

  def batch_norm(self, x):
    if self.hparams.batch_norm:
      return layers.batch_norm(x, is_training=self.mode == 'train')
    else:
      return x

  def make_embedding(self, x):
    hparams = self.hparams
    for size in 1024, 1024, hparams.hidden_size:
      x = model_utils.pad_up_to(x, size=13, axis=1)
      x = self.dropout(x)
      reg = self.make_keras_regularizer()
      x = keras.layers.Convolution1D(size, hparams.filter_width,
                                     border_mode='valid', subsample_length=1,
                                     W_regularizer=reg, activation=None)(x)
      self.apply_keras_regularizer(reg)
      x = self.batch_norm(x)
      x = tf.nn.relu(x)
    x = tf.reduce_max(x, reduction_indices=1)
    return x

  def conjecture_embedding(self, conjectures):
    """Compute the embedding for each of the conjectures."""
    return self.make_embedding(conjectures)

  def axiom_embedding(self, axioms):
    """Compute the embedding for each of the axioms."""
    return self.make_embedding(axioms)

  def classifier(self, conjecture_embedding, axiom_embedding):
    """Compute the logits from conjecture and axiom embeddings."""
    net = tf.concat((conjecture_embedding, axiom_embedding), 1)
    net = self.dropout(net)
    net = layers.relu(net, 1024, weights_regularizer=self.regularizer)
    logits = tf.squeeze(layers.linear(net, 1), [-1])
    return logits
