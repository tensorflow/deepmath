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
"""Wavenet model for clause selection."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from deepmath.guidance import wavenet
from deepmath.premises import model
from deepmath.util import model_utils

layers = tf.contrib.layers


def default_hparams():
  """Default values for model hyperparameters."""
  return tf.contrib.training.HParams(
      keep_prob=1.0,  # Keep probability for dropout (1.0 means no dropout)
      weight_decay=0.0,  # Weight decay TODO(alemi): add weight decay back
      wavenet_layers=6,  # Number of layers in each block
      wavenet_blocks=3,  # number of wavenet blocks.
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

  def make_embedding(self, x):
    hparams = self.hparams
    # Ensure that text length is divisible by the total dilation
    # the wavenet blocks will impose.  This should be 2**(wavenet_layers)
    x = model_utils.pad_to_multiple(x, 2**hparams.wavenet_layers, axis=1)

    # Build a wavenet
    net = tf.expand_dims(x, 2)
    for _ in range(hparams.wavenet_blocks):
      net = wavenet.wavenet_block(net, hparams.wavenet_layers,
                                  hparams.embedding_size)

    # Do a global max pool
    net = tf.reduce_max(net, [1, 2])
    return net

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
