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
"""Tree RNN for clause search."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow_fold.public import loom
from deepmath.guidance import clause_loom
from deepmath.guidance import inputs
from deepmath.util import model_utils

layers = tf.contrib.layers


def default_hparams():
  return tf.contrib.training.HParams(
      cell='rnn-relu',  # Type of RNN cell to use
      layers=1,  # Number of layers
      hidden_size=1024,  # Size of intermediate RNN embeddings
      keep_prob=1.0,  # Optional recurrent dropout
      forget_bias=1.0,  # Bias towards not forgetting
      output_mode='cell',  # Which LSTM part to use as the final embedding
  )


def loom_model(weave, vocab, hparams, mode):
  """Tree RNN model to compute logits from from ProverClauseExamples.

  Args:
    weave: Called with the loom op keyword arguments described in
        clause_loom.weave_clauses.
    vocab: Path to vocabulary file.
    hparams: Hyperparameters.
    mode: Either 'train' or 'eval'.

  Returns:
    The results of the call to `weave`.
  """
  hidden_size = hparams.hidden_size
  embedding_size = hparams.embedding_size
  vocab_size, _ = inputs.read_vocab(vocab)
  per_layer = 2 if hparams.cell == 'lstm' else 1

  # TypeShapes
  vocab_id = clause_loom.VOCAB_ID
  logit = loom.TypeShape(tf.float32, (), 'logit')
  # Use separate embedding type shapes for separate layers to avoid confusion.
  # TODO(geoffreyi): Allow different sizes for different layers?
  embeddings = tuple(
      loom.TypeShape(tf.float32, (hidden_size,), 'embedding%d' % i)
      for i in range(hparams.layers * per_layer))

  @model_utils.as_loom_op([vocab_id], embeddings)
  def embed(ids):
    """Embed tokens and use a linear layer to get the right size."""
    values = model_utils.embedding_layer(ids, dim=embedding_size,
                                         size=vocab_size)
    if embedding_size < hidden_size:
      values = layers.linear(values, hidden_size)
    elif embedding_size > hidden_size:
      raise ValueError('embedding_size = %d > hidden_size = %d' %
                       (embedding_size, hidden_size))

    # Use relu layers to give one value per layer
    values = [values]
    for _ in range(hparams.layers - 1):
      # TODO(geoffreyi): Should these be relu or some other activation?
      values.append(layers.relu(values[-1], hidden_size))

    # If we're using LSTMs, initialize the memory cells to zero.
    if hparams.cell == 'lstm':
      memory = tf.zeros_like(values[0])
      values = [v for hidden in values for v in (memory, hidden)]
    return values

  def merge(arity, name):
    """Merge arity inputs with rule according to hparams.cell."""
    @model_utils.as_loom_op(embeddings * arity, embeddings, name=name)
    def merge(*args):
      """Process one batch of RNN inputs."""
      # shape = (arity, layers) for RNNs, (arity, layers, 2) for LSTMs,
      # where the 2 dimension is (memory, hidden).
      shape = (arity, hparams.layers) + (per_layer,) * (per_layer > 1)
      args = np.asarray(args).reshape(shape)
      below = ()  # Information flowing up from the previous layer
      outputs = []  # Results of each layer
      if hparams.cell == 'rnn-relu':
        # Vanilla RNN with relu nonlinearities
        if hparams.keep_prob != 1:
          raise ValueError('No dropout allowed for vanilla RNNs')
        for layer in range(hparams.layers):
          output = layers.relu(
              tf.concat(below + tuple(args[:, layer]), 1), hidden_size)
          outputs.append(output)
          below = output,
      elif hparams.cell == 'lstm':
        # Tree LSTM with separate forget gates per input and optional recurrent
        # dropout.  For details, see
        # 1. Improved semantic representations from tree-structured LSTM
        #    networks, http://arxiv.org/abs/1503.00075.
        # 2. Recurrent dropout without memory loss,
        #    http://arxiv.org/abs/1603.05118.
        # 3. http://colab/v2/notebook#fileId=0B2ewRpEjJXEFYjhtaExiZVBXbUk.
        memory, hidden = np.rollaxis(args, axis=-1)
        for layer in range(hparams.layers):
          raw = layers.linear(
              tf.concat(below + tuple(hidden[:, layer]), 1),
              (3 + arity) * hidden_size)
          raw = tf.split(value=raw, num_or_size_splits=3 + arity, axis=1)
          (i, j, o), fs = raw[:3], raw[3:]
          j = tf.tanh(j)
          j = layers.dropout(j, keep_prob=hparams.keep_prob,
                             is_training=mode == 'train')
          new_c = tf.add_n([tf.sigmoid(i) * j] +
                           [c * tf.sigmoid(f + hparams.forget_bias)
                            for c, f in zip(memory[:, layer], fs)])
          new_h = tf.tanh(new_c) * tf.sigmoid(o)
          outputs.extend((new_c, new_h))
          below = new_h,
      else:
        # TODO(geoffreyi): Implement tree GRU?
        raise ValueError('Unknown rnn cell type %r' % hparams.cell)
      return outputs
    return merge

  @model_utils.as_loom_op(embeddings * 2, logit)
  def classify(*args):
    """Compute logits from conjecture and clause embeddings."""
    # Use the top layer, and either cell state, hidden state, or both
    which = {'cell': 0, 'hidden': 1, 'both': (0, 1)}[hparams.output_mode]
    args = np.asarray(args).reshape(2, hparams.layers, per_layer)
    args = args[:, -1, which]
    value = layers.relu(tf.concat(tuple(args.flat), 1), hidden_size)
    return tf.squeeze(layers.linear(value, 1), [1])

  return weave(
      embed=embed,
      conjecture_apply=merge(
          2, name='conjecture/apply'),
      conjecture_not=merge(
          1, name='conjecture/not'),
      conjecture_or=merge(
          2, name='conjecture/or'),
      conjecture_and=merge(
          2, name='conjecture/and'),
      clause_apply=merge(
          2, name='clause/apply'),
      clause_not=merge(
          1, name='clause/not'),
      clause_or=merge(
          2, name='clause/or'),
      combine=classify)
