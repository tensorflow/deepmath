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
"""Unconditional convolutional model with word embedded input.

A control version of the CNN model that ignores the negated
conjecture.  We'll see if it performs worse than the conditional
model!
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
      hidden_size=1024,  # Size of the clause embeddings
      inner_size=1024,  # Size of intermediate convolutional layers
      classifier_size=1024,  # Size of classifier hidden layer
      filter_width=5,  # Filter width
  )


class Model(model.Model):
  """Convolutional model for word embedded input."""

  def __init__(self, mode, hparams):
    super(Model, self).__init__(graph=tf.get_default_graph(),
                                embedding_size=None)
    self.mode = mode
    self.hparams = hparams

  def make_embedding(self, x):
    hparams = self.hparams
    for size in hparams.inner_size, hparams.inner_size, hparams.hidden_size:
      x = model_utils.pad_up_to(x, size=13, axis=1)
      x = keras.layers.Convolution1D(size, hparams.filter_width,
                                     border_mode='valid', subsample_length=1,
                                     activation=None)(x)
      x = tf.nn.relu(x)
    x = tf.reduce_max(x, reduction_indices=1)
    return x

  def conjecture_embedding(self, conjectures):
    """Ignore the conjecture by outputing an empty tensor."""
    return tf.zeros(tf.stack([tf.shape(conjectures)[0], 0]))

  def axiom_embedding(self, axioms):
    """Compute the embedding for each of the axioms."""
    return self.make_embedding(axioms)

  def classifier(self, unused_conjecture_embedding, axiom_embedding):
    """Compute the logits from conjecture and axiom embeddings."""
    net = axiom_embedding  # Ignore the conjecture
    net = layers.relu(net, self.hparams.classifier_size)
    logits = tf.squeeze(layers.linear(net, 1), [-1])
    return logits
