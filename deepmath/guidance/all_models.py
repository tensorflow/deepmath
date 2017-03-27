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
"""Indexed list of all modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tensorflow as tf
from deepmath.guidance import cnn_unconditional
from deepmath.guidance import fast_cnn
from deepmath.guidance import model_cnn_regularized
from deepmath.guidance import model_wavenet
from deepmath.guidance import tree_rnn
from deepmath.premises import model_definition_cnn_flat3
from deepmath.premises import model_final_cnn_3x_lstm


ALL_MODELS = {
    m.__name__.split('.')[-1]: m
    for m in [
        model_cnn_regularized,
        cnn_unconditional,
        fast_cnn,
        tree_rnn,
        model_wavenet,
    ]
}

PREMISE_MODELS = {
    m.__name__.split('.')[-1]: m
    for m in [
        model_definition_cnn_flat3,
        model_final_cnn_3x_lstm,
    ]
}


def model_module(name):
  """Gets the model module of given name.

  Args:
    name: Model name (same as the module that defines it).

  Returns:
    A model module.

  Raises:
    ValueError: If the model name doesn't exist.
  """
  if name in ALL_MODELS:
    return ALL_MODELS[name]
  elif name in PREMISE_MODELS:
    return PREMISE_MODELS[name]
  else:
    raise ValueError("Unknown model '%s'" % name)


def make_model(name, mode, hparams, vocab=None):
  """Constructs a Model from the given module.

  Args:
    name: Model name (same as the module that defines it).
    mode: Either 'train' or 'eval'.
    hparams: Hyperparameters.
    vocab: Path to vocabulary file (required for tree models).

  Returns:
    kind: Either 'sequence', 'tree', or 'fast'.
    model: If kind=='sequence', a Model class with conjecture embedding,
        axiom_embedding, and classifier functions.  conjecture_embedding and
        axiom_embedding both take pre-embedded sequences.  If kind=='tree', a
        function mapping a batch of ProverClauseExamples to logits, labels.
        If kind=='fast', a function mapping a jagged tensor of conjecture
        clauses and clause to logits.

  Raises:
    ValueError: If the model name doesn't exist.
  """
  if name in ALL_MODELS:
    module = model_module(name)
    tf.logging.info('Creating model %s', name)
    if hasattr(module, 'Model'):
      return 'sequence', module.Model(mode=mode, hparams=hparams)
    elif hasattr(module, 'loom_model'):
      return 'tree', functools.partial(module.loom_model, vocab=vocab,
                                       hparams=hparams, mode=mode)
    elif hasattr(module, 'fast_model'):
      return 'fast', functools.partial(
          module.fast_model, vocab=vocab, hparams=hparams, mode=mode)
    else:
      raise ValueError("Can't determine kind of model '%s'" % name)
  elif name in PREMISE_MODELS:
    return 'sequence', PREMISE_MODELS[name].Model(tf.get_default_graph())
  else:
    raise ValueError("Unknown model '%s'" % name)


def model_hparams(name):
  """Gets the hparams for the given model.

  Args:
    name: Model name (same as the module that defines it).

  Returns:
    An HParams object.
  """
  if name in ALL_MODELS:
    module = model_module(name)
    return module.default_hparams()
  # Old style models have no hyperparameters
  return tf.contrib.training.HParams()
