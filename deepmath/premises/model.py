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
"""Simple example and API for pairwise models.

This is the base class of premise selection models.  Derived classes should
implement their own conjecture_embedding, axiom_embedding and classifier
methods.  Other scripts should use this api to manipulate saved models for
evaluation and caching purposes.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

layers = tf.contrib.layers

MOVING_AVERAGE_DECAY = 0.9999


class Model(object):
  """The pairwise model class."""

  def __init__(self, graph=None, embedding_size=256):
    self.graph = graph or tf.get_default_graph()
    self.embedding_size = embedding_size

  def conjecture_embedding(self, conjectures):
    """Compute the embedding for each of the conjectures.

    When creating a new model, this should be overwritten.

    Args:
      conjectures: a bool tensor of shape [batch_size, text_length, num_chars]

    Returns:
      conjecture_embedding: the tensor representing the embedding of each
        conjecture. Recommended to be a float32 tensor of shape
        [batch_size, conjecture_embedding_size]
    """
    raise NotImplementedError('Use a derived model')

  def axiom_embedding(self, axioms):
    """Compute the embedding for each of the axioms.

    When creating a new model, this should be overwritten.

    Args:
      axioms: a bool tensor of shape [batch_size, text_length, num_chars]

    Returns:
      axiom_embeddings: the tensor representing the embedding of each
        axiom. Recommended to be a float32 tensor of shape
        [batch_size, axiom_embedding_size]
    """
    raise NotImplementedError('Use a derived model')

  def classifier(self, conjecture_embedding, axiom_embedding):
    """Given an axiom_embedding and conjecture_embedding, compute the logits.

    When creating a new model, this should be overwritten.

    Args:
      conjecture_embedding: the output of the model's conjecture_embedding
        method, which is recommended to be a float32 tensor of shape
        [batch_size, conjecture_embedding_size]
      axiom_embedding: the output of the model's axiom embedding
        method, which is recommended to be a float32 tensor of shape
        [batch_size, axiom_embedding_size]

    Returns:
      logits: the tensor representing logits of the pair being a true or
        false dependency. Must be a float32 tensor of shape
        [batch_size, 2].
    """
    with self.graph.as_default():
      if tf.flags.FLAGS.fully_connected:
        concat_rep = tf.concat((conjecture_embedding, axiom_embedding), 1)
        net = layers.relu(concat_rep, 256)
        net = layers.relu(net, 256)
        logits = layers.linear(net, 2)
      else:
        logit = tf.reduce_sum(
            tf.multiply(conjecture_embedding, axiom_embedding),
            1,
            name='dot_of_emb')
        logits = tf.concat([logit[:, None], -logit[:, None]], 1)
    return logits

  def model(self, conjectures, axioms):
    """Convenience method to compute the model end to end."""
    with self.graph.as_default():
      conjecture_embeddings = self.conjecture_embedding(conjectures)
      axiom_embeddings = self.axiom_embedding(axioms)
      predictions = self.classifier(conjecture_embeddings, axiom_embeddings)
    return predictions
