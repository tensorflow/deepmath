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
"""Clause selection driver utilities for training and eval."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf
from deepmath.guidance import all_models
from deepmath.guidance import clause_loom
from deepmath.guidance import gen_clause_ops
from deepmath.guidance import inputs
from deepmath.guidance import jagged
from deepmath.guidance import train
from deepmath.util import model_utils

FLAGS = tf.flags.FLAGS


def parse_hparams(hparam_str):
  """Parse hyperparameters from the given flag value."""
  m = re.search(r'(?:^|,)model=(\w+)(?:,|$)', hparam_str)
  if not m:
    raise ValueError('No model specified in --hparams=%r' % hparam_str)
  input_hparams = tf.contrib.training.HParams(
      model='',  # Name of the model to run
      embedding_size=256,  # Word embedding dimension
  )
  hparams = model_utils.merge_hparams(train.default_hparams(),
                                      all_models.model_hparams(m.group(1)),
                                      input_hparams)
  hparams.parse(hparam_str)
  return hparams


def mode_batch_size(mode, hparams):
  """Returns the batch size for a given mode (train or eval).

  Args:
    mode: Either 'train' or 'eval'.
    hparams: Hyperparameters.

  Returns:
    Integer batch size.

  Raises:
    ValueError: If mode is not 'train' or 'eval'.
  """
  if mode == 'train':
    return hparams.batch_size
  elif mode == 'eval':
    return hparams.eval_batch_size
  else:
    raise ValueError('Invalid --mode=%r' % mode)


def fix_logits(kind, logits):
  """Fix logits to be scalar (True / False) rather than two class.

  Args:
    kind: Either 'sequence' or 'tree'.
    logits: Logits tensor of shape (?, 2) or (?,).

  Returns:
    Logits tensor of shape (?,)

  Raises:
    ValueError: If logits has an invalid shape.
  """
  logits_rank = logits.get_shape().ndims
  if kind == 'sequence' and logits_rank == 2:
    logits.get_shape().merge_with((None, 2))
    logits = logits[:, 0]
  elif logits_rank != 1:
    raise ValueError('logits has bad rank %r' % logits_rank)
  return logits


def full_model(mode, hparams):
  """Make a clause search model including input pipeline.

  Args:
    mode: Either 'train' or 'eval'.
    hparams: Hyperparameters.  See default_hparams for details.

  Returns:
    logits, labels

  Raises:
    ValueError: If the model returns badly shaped tensors.
  """
  if hparams.use_averages:
    raise NotImplementedError('Figure out how to eval with Polyak averaging')
  kind, model = all_models.make_model(name=hparams.model, mode=mode,
                                      hparams=hparams, vocab=FLAGS.vocab)
  batch_size = mode_batch_size(mode, hparams)

  if kind == 'sequence':
    # Read
    _, conjectures, clauses, labels = inputs.sequence_example_batch(
        mode=mode, batch_size=batch_size, shuffle=True)
    clauses = tf.reshape(clauses, [2 * batch_size, -1])
    labels = tf.reshape(labels, [2 * batch_size])

    # Embed
    vocab_size, _ = inputs.read_vocab(FLAGS.vocab)
    conjectures, clauses = model_utils.shared_embedding_layer(
        (conjectures, clauses), dim=hparams.embedding_size, size=vocab_size)

    # Classify
    conjectures = model.conjecture_embedding(conjectures)
    conjectures = tf.reshape(
        tf.tile(tf.reshape(conjectures, [batch_size, 1, -1]), [1, 2, 1]),
        [2 * batch_size, -1])
    clauses = model.axiom_embedding(clauses)
    logits = model.classifier(conjectures, clauses)
  elif kind == 'tree':
    examples = inputs.proto_batch(mode=mode, batch_size=batch_size)
    def weave(**ops):
      return clause_loom.weave_clauses(
          examples=examples, vocab=FLAGS.vocab, **ops)

    logits, labels = model(weave)
  elif kind == 'fast':
    examples = inputs.proto_batch(mode=mode, batch_size=batch_size)
    conjecture_sizes, conjecture_flat, clauses, labels = (
        gen_clause_ops.random_clauses_as_fast_clause(
            examples, vocab=FLAGS.vocab))
    conjectures = jagged.Jagged(conjecture_sizes, conjecture_flat)
    logits = model(conjectures, clauses)

  # Done!
  return fix_logits(kind, logits), labels


def with_name(value, name):
  """Returns a tensor with value value and name name.

  Args:
    value: A tensor.
    name: A name.

  Returns:
    A tensor with the given value and name.

  Raises:
    TypeError: If the input isn't a Tensor or Operation.
    ValueError: If the name is taken.
  """
  if isinstance(value, tf.Tensor):
    nop, suffix = tf.identity, ':0'
  elif isinstance(value, tf.Operation):
    nop, suffix = tf.group, ''
  else:
    raise TypeError('Expected Tensor or Operation, got %r' % type(value))
  if value.name != name + suffix:
    value = nop(value, name=name)
    if value.name != name + suffix:
      raise ValueError('Tried to ensure name %r, but got %r' %
                       (name + suffix, value.name))
  return value


def inference(hparams):
  """Make a clause search graph suitable for inference at proof time.

  Each described node has the correct name, for purposes of C++ lookup:

      conjecture, clauses: string, shape (?,), placeholders of serialized
          FastClause protos.
      conjecture_embeddings: float32, shape (dim,).
      logits: float32, shape (?,) output logits.
      initialize: Initialization op.

  Args:
    hparams: Hyperparameters.  See default_hparams for details.

  Returns:
    The tf.Saver object.

  Raises:
    ValueError: If the model kind is not 'tree' or 'sequence'.
  """
  if hparams.use_averages:
    raise NotImplementedError('Figure out how to eval with Polyak averaging')
  kind, model = all_models.make_model(name=hparams.model, mode='eval',
                                      hparams=hparams, vocab=FLAGS.vocab)

  # Input placeholders, which will hold FastClause protos.
  conjecture = tf.placeholder(
      name='conjecture', shape=(None,), dtype=tf.string)
  clauses = tf.placeholder(name='clauses', shape=(None,), dtype=tf.string)

  def expand(embedding):
    """Tile the one conjecture to match clauses."""
    embeddings = tf.tile(embedding, tf.stack([tf.size(clauses), 1]))
    embeddings.set_shape([None, embedding.get_shape()[-1]])
    return embeddings

  if kind == 'sequence':
    # Embedding weights
    vocab_size, _ = inputs.read_vocab(FLAGS.vocab)
    params = model_utils.embedding_weights(dim=hparams.embedding_size,
                                           size=vocab_size)

    # Embed conjecture
    ids = gen_clause_ops.fast_clauses_as_sequence(
        conjecture, conjunction=True)
    ids = tf.nn.embedding_lookup(params, ids)
    ids = ids[None]  # Singleton batch since many clauses are one ~conjecture
    conjecture_embedding = with_name(
        model.conjecture_embedding(ids), name='conjecture_embeddings')

    # Embed clauses
    ids = gen_clause_ops.fast_clauses_as_sequence(clauses)
    ids = tf.nn.embedding_lookup(params, ids)
    clause_embeddings = model.axiom_embedding(ids)

    # Classify
    logits = model.classifier(expand(conjecture_embedding), clause_embeddings)
  elif kind == 'tree':
    def weave(embed, conjecture_apply, conjecture_not, conjecture_or,
              conjecture_and, clause_apply, clause_not, clause_or, combine):
      """Weave conjecture and clauses separately, then combine."""
      # Embed conjecture, naming a concatenated version for simplicity
      parts = clause_loom.weave_fast_clauses(
          clauses=conjecture,
          embed=embed,
          apply_=conjecture_apply,
          not_=conjecture_not,
          or_=conjecture_or,
          and_=conjecture_and,
          shuffle=False)
      concat = tf.concat(parts, 1, name='conjecture_embeddings')
      splits = tf.split(
          concat, [p.get_shape()[1].value for p in parts], axis=1)
      splits = [expand(s) for s in splits]

      # Embed clauses
      clause_embeddings = clause_loom.weave_fast_clauses(
          clauses=clauses,
          embed=embed,
          apply_=clause_apply,
          not_=clause_not,
          or_=clause_or,
          shuffle=False)

      # Combine into logits
      return combine.instantiate_batch(splits + list(clause_embeddings))

    logits, = model(weave)
  elif kind == 'fast':
    logits = model(jagged.pack([conjecture]), clauses)
  else:
    raise ValueError('Unknown kind %r' % kind)

  # Fix and name logits
  with_name(fix_logits(kind, logits), name='logits')

  # Add init op for testing purposes
  with_name(tf.global_variables_initializer(), name='initialize')

  # Add saver and init ops (the latter only for test purposes)
  return tf.train.Saver()


def export_inference_meta_graph(hparams, filename=None, as_text=False):
  """Export the inference graph to a file.

  See `inference` above for details about graph structure.

  Args:
    hparams: Hyperparameters.  See default_hparams for details.
    filename: Optional filename to write the MetaGraphDef to.
    as_text: If true, use ASCII format.

  Returns:
    A MetaGraphDef proto.
  """
  with tf.Graph().as_default():
    saver = inference(hparams)
    return saver.export_meta_graph(filename=filename, as_text=as_text)


def run_mode(mode, hparams):
  """Either train or evaluate a clause search model.

  Args:
    mode: Either 'train' or 'test'.
    hparams: Hyperparameters.

  Raises:
    ValueError: If mode is not 'train' or 'eval'.
  """
  if hparams.seed:
    tf.set_random_seed(hparams.seed)
  model = lambda: full_model(mode, hparams)
  if mode == 'train':
    train.sigmoid_train(model, hparams=hparams, joint_safe=True)
  elif mode == 'eval':
    train.sigmoid_eval(model, hparams=hparams, joint_safe=True)
  else:
    raise ValueError('Invalid --mode=%r' % mode)
