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
"""CNN model without padding."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tensorflow as tf
from deepmath.guidance import inputs
from deepmath.guidance import jagged
from deepmath.util import model_utils

layers = tf.contrib.layers


def default_hparams():
  """Default values for model hyperparameters."""
  return tf.contrib.training.HParams(
      conv_layers=3,
      mid_layers=1,
      final_layers=1,
      filter_width=5,
      hidden_size=1024)


def fast_model(conjectures, clauses, vocab, hparams, mode):
  """Classify conjectures and clauses.

  Args:
    conjectures: Negated conjectures as a Jagged of serialized FastClauses.
    clauses: Clauses as serialized FastClauses.
    vocab: Path to vocabulary file.
    hparams: Hyperparameters.
    mode: Either 'train' or 'eval'.  Unused.

  Returns:
    Logits.
  """
  _ = mode  # Mode is unused
  hidden_size = hparams.hidden_size
  conv_layers = hparams.conv_layers

  # Convert all FastClauses to sequences of ids
  conjectures = inputs.fast_clauses_as_sequence_jagged(conjectures)
  clauses = inputs.fast_clauses_as_sequence_jagged(clauses)

  # Embed ids
  vocab_size, _ = inputs.read_vocab(vocab)
  params = model_utils.embedding_weights(dim=hparams.embedding_size,
                                         size=vocab_size)
  conjectures = jagged.jagged(
      conjectures.sizes,
      tf.nn.embedding_lookup(params, conjectures.flat))
  clauses = jagged.jagged(clauses.sizes,
                          tf.nn.embedding_lookup(params, clauses.flat))

  def bias_relu(x, bias):
    return tf.nn.relu(x + bias)

  def embed_clauses(clauses, name):
    with tf.variable_scope(name):
      filters, activations = [], []
      dim = hparams.embedding_size
      for i in range(conv_layers):
        filters.append(
            tf.get_variable(
                'filter%d' % i,
                shape=(hparams.filter_width, dim, hidden_size),
                initializer=layers.xavier_initializer()))
        bias = tf.get_variable(
            'bias%d' % i,
            shape=(hidden_size,),
            initializer=tf.constant_initializer(0))
        activations.append(functools.partial(bias_relu, bias=bias))
        dim = hidden_size
      clauses = jagged.conv1d_stack(clauses, filters, activations)
      return jagged.reduce_max(clauses)

  # Embed conjectures
  conjectures = embed_clauses(conjectures, 'conjectures')
  for _ in range(hparams.mid_layers):
    conjectures = jagged.Jagged(conjectures.sizes,
                                layers.relu(conjectures.flat, hidden_size))
  conjectures = jagged.reduce_max(conjectures, name='conjecture_embeddings')

  # Embed clauses
  clauses = embed_clauses(clauses, 'clauses')

  # Repeat each conjecture enough times to match clauses
  expansion = tf.size(clauses) // tf.maximum(1, tf.size(conjectures))
  conjectures = tf.reshape(tf.tile(conjectures[:, None], [1, expansion, 1]),
                           [-1, hidden_size])

  # Classify
  net = tf.concat((conjectures, clauses), 1)
  net = layers.relu(net, hparams.hidden_size)
  logits = tf.squeeze(layers.linear(net, 1), [-1])
  return logits
