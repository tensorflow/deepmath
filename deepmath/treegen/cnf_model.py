# coding=utf-8
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
"""A generative model for CNF expressions.

CNF grammar in TPTP-3:
cnf
  CNF(literal+)
literal
  Literal(sign, atom)
sign
  positive
  negative
atom
  Equals(term, term)
  Pred(pred_name, term*)
pred_name
  pred1, pred2, ..., predN
term
  Variable(var_name)
  Function(func_name, term*)
  Number(number)
number
  0, 1, ...
func_name
  func1, ...funcN
Variable
  var1, ..., varN
  or sample a new variable/keep an existing one (like Dirichlet process?)

Simplified CNF grammar, with merged predicates and functions, and Equals
treated as a function (implemented here):

cnf
  CNF(literal+)
literal
  Literal(sign, atom)
sign
  positive
  negative
atom
  Function(func_name, term*)
term
  Function(func_name, term*)
  Variable(var_name)
  Number(number)
number
  0, 1, ...
func_name
  func1, ...funcN
Variable
  var1, ..., varN
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import functools
import itertools
import re
import threading
import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow_fold.public import loom
from deepmath.treegen import loom_ops
from deepmath.util import model_utils
from tensorflow.python.util import nest

framework = tf.contrib.framework
layers = tf.contrib.layers
losses = tf.contrib.losses
LSTMStateTuple = tf.contrib.rnn.LSTMStateTuple  # pylint: disable=invalid-name
flags = tf.flags
logging = tf.logging

FLAGS = flags.FLAGS

flags.DEFINE_integer('max_literals_in_clause', 30,
                     'Maximum number of literals that will be sampled '
                     'from one clause.')
flags.DEFINE_integer('max_function_depth', 10,
                     'Maximum amount of nesting of functions allowed '
                     'while sampling.')
flags.DEFINE_integer('max_sequential_tokens', 500,
                     'Maximum number of tokens that will be sampled '
                     'in the sequence model.')


class LoomInputWithExtras(object):
  """Wrapper around LoomInput to conveniently collect logits and labels."""

  def __init__(self, loom_input):
    self.loom_input = loom_input
    self.xent_costs = []

  def __getattr__(self, name):
    return getattr(self.loom_input, name)

  def supervise_choice(self, type_name, which_op_output, true_label):
    true_label = self.constant(np.int64(true_label))
    xent_cost, = self.loom_input.op('xent_' + type_name, [
        which_op_output, true_label
    ])
    self.loom_input.add_output(xent_cost)
    self.xent_costs.append(xent_cost)

  def reset_xent_costs(self):
    self.xent_costs = []


SamplingPlaceholder = collections.namedtuple('SamplingPlaceholder',
                                             ['type_shape', 'index', 'apply_fn',
                                              'ctx'])


class SamplingContext(object):
  """Maps rows in Loom-generated outputs to placeholders."""

  def __init__(self, the_loom, loom_input):
    self.loom = the_loom
    self.loom_input = loom_input

    self.outputs = None
    self.type_shape_counts = collections.defaultdict(int)

  def register(self, loom_outputs, apply_fn=lambda x: x):
    """Get `SamplingPlaceholder`s for each `loom_output`."""
    if not isinstance(loom_outputs, (list, tuple)):
      loom_outputs = [loom_outputs]
      was_singleton = True
    else:
      was_singleton = False

    results = []
    for loom_output in loom_outputs:
      self.loom_input.add_output(loom_output)
      output_type_shape = self.loom_input.get_type_shape(loom_output)
      result = SamplingPlaceholder(output_type_shape,
                                   self.type_shape_counts[output_type_shape],
                                   apply_fn, self)
      self.type_shape_counts[output_type_shape] += 1
      results.append(result)

    if was_singleton:
      return results[0]
    return results

  def run_and_insert(self, sess, obj):
    """Replace `SamplingPlaceholder`s in `obj` with actual output."""
    if self.outputs is None:
      output_tensors = {ts: self.loom.output_tensor(ts)
                        for ts in self.type_shape_counts.keys()}
      self.outputs = sess.run(output_tensors, self.loom_input.build_feed_dict())

    if isinstance(obj, SamplingPlaceholder):
      obj = self.retrieve(obj, self.outputs)
    else:
      self.insert(obj, self.outputs)
    return obj

  def insert(self, obj, outputs):
    if isinstance(obj, list):
      iterator = enumerate(obj)
    elif isinstance(obj, dict):
      iterator = obj.items()
    else:
      raise TypeError('%s must be list, dict' % repr(obj))

    for key, value in iterator:
      if isinstance(value, SamplingPlaceholder):
        obj[key] = self.retrieve(value, outputs)
      elif isinstance(value, (list, dict)):
        self.insert(value, outputs)

  def retrieve(self, value, outputs):
    if value.ctx is not self:
      raise ValueError('Mismatch between placeholder and context.')
    return value.apply_fn(outputs[value.type_shape][value.index])


def common_model_hparams():
  return tf.contrib.training.HParams(embedding_length=32, l2_weight=1e-6,
                                     batch_size=64)


class SerializedCNFStateMachine(object):
  """Keep track of which tokens can be next while generating serialized CNF."""

  # These are expected to start at 0 and be contiguous.
  PRED_START = 0
  FUNC_CALL = 1
  FUNC_ARG = 2
  PRED_END = 3
  ALL_STATES = (PRED_START, FUNC_CALL, FUNC_ARG, PRED_END)

  def __init__(self, func_arities):
    self.func_arities = func_arities
    self.last_token = None
    # Keeps track of how many arguments remain in the currently-active function
    # calls. All elements are positive integers (> 0).
    self.num_remaining_args = []
    self.state = self.PRED_START

  def add_token(self, token):
    """Add a new token and update the internal state.

    Args:
      token: The new token.
    Raises:
      ValueError: `token` results in an invalid state.
    """

    self.last_token = token
    if token.startswith('FUN_'):
      if self.num_remaining_args:
        # If there is currently an unclosed function call, mark that we are
        # adding an argument to it.
        self.num_remaining_args[-1] -= 1
      self.num_remaining_args.append(self.func_arities[token[4:]])
    elif re.match(r'^(VAR_|NUM_)', token):
      assert self.num_remaining_args
      self.num_remaining_args[-1] -= 1

    while self.num_remaining_args and self.num_remaining_args[-1] == 0:
      del self.num_remaining_args[-1]

    if self.last_token is None or self.last_token == '|':
      self.state = self.PRED_START
    elif self.last_token == '~':
      self.state = self.FUNC_CALL
    elif self.num_remaining_args:
      self.state = self.FUNC_ARG
    elif not self.num_remaining_args:
      self.state = self.PRED_END
    else:
      raise ValueError('%s caused transition to invalid state.' % token)


class CNFSequenceModel(object):
  """Builds a TF graph for a LSTM model.

  The model operates on tokens, and does not use tokens like | or (. Instead
  all groupings are inferred from knowing the arities of each function and
  predicate (like reverse Polish notation for arithmetic).
  """

  @staticmethod
  def default_hparams():
    return model_utils.merge_hparams(
        common_model_hparams(),
        tf.contrib.training.HParams(
            # TODO(ricshin): Add parameter to control type of RNNCell.
            depth=2,
            emb_partitions=1,
            objective='zero_z',  # TODO(ricshin): Use this value.
            masked_xent=False))

  def __init__(self, data_iterator, hparams, clause_metadata):
    """Constructor.

    Args:
      data_iterator: Python iterator which provides each instance as a dict.
      hparams: A HParams object with:
        - batch_size: Batch size to use.
        - embedding_length: Size of embedding used throughout the model.
        - l2_weight: Amount of weight to put on the L2 regularization.
        - depth: Depth of the LSTM.
        - emb_partitions: Number of partitions for all token embeddings.
        - objective: One of the following:
          - zero_z: Maximize p(x | z) where z is fixed to be 0.
        - masked_xent: When computing cross-entropy penalty, only use
            logits amongst the tokens that we would select from at that point.
      clause_metadata: A dictionary containing the following:
        - func_names_and_types: List of all (function/predicate names,
          is_predicate) pairs.
        - func_arities: Dictionary from function/predicate names to their
          arities.
        - var_names: List of all variables in the data.
        - numbers: Numbers which appear in the data.
    """
    self.hparams = hparams
    emb_size = self.emb_size = hparams.embedding_length

    func_names_and_types, func_arities, var_names, numbers = (
        clause_metadata['func_names'], clause_metadata['func_arities'],
        clause_metadata['var_names'], clause_metadata['numbers'])
    self.func_arities = func_arities
    # Used while sampling if we exceeded the token limit but have
    # functions/predicates without enough arguments.
    self.filler_number = 'NUM_' + numbers[-1]

    # TODO(ricshin): Handle functions and predicate separately.
    special = ['<s>', '</s>', '~', '|']
    all_names = (special + ['FUN_' + name for name, _ in func_names_and_types] +
                 ['VAR_' + name for name in var_names] +
                 ['NUM_' + num for num in numbers])

    self.name_to_id = dict(zip(all_names, itertools.count(0)))
    self.id_to_name = dict(zip(itertools.count(0), all_names))
    self.func_name_to_is_predicate = dict(func_names_and_types)

    # FUN_*
    self.func_ids = list(range(
        len(special), len(special) + len(func_names_and_types)))
    self.func_ids_map = dict(zip(self.func_ids, itertools.count(0)))
    # ~, FUN_*
    self.pred_start_ids = [self.name_to_id['~']] + self.func_ids
    self.pred_start_ids_map = dict(zip(self.pred_start_ids, itertools.count(0)))
    # FUN_*, NUM_*, VAR_*
    self.arg_ids = list(range(len(special), len(all_names)))
    self.arg_ids_map = dict(zip(self.arg_ids, itertools.count(0)))
    # </s>, ~, FUN_*
    self.pred_end_ids = (
        [self.name_to_id['</s>'], self.name_to_id['~']] + self.func_ids)
    self.pred_end_ids_map = dict(zip(self.pred_end_ids, itertools.count(0)))
    # These ids_maps are for mapping from the global IDs (in self.name_to_id)
    # to a renumbered list of subset of tokens, which is used later for
    # computing separate softmaxes for each subset.

    ids_masks = {
        SerializedCNFStateMachine.PRED_START: self.pred_start_ids,
        SerializedCNFStateMachine.FUNC_CALL: self.func_ids,
        SerializedCNFStateMachine.FUNC_ARG: self.arg_ids,
        SerializedCNFStateMachine.PRED_END: self.pred_end_ids
    }
    ids_rev_maps = {
        SerializedCNFStateMachine.PRED_START: self.pred_start_ids_map,
        SerializedCNFStateMachine.FUNC_CALL: self.func_ids_map,
        SerializedCNFStateMachine.FUNC_ARG: self.arg_ids_map,
        SerializedCNFStateMachine.PRED_END: self.pred_end_ids_map
    }

    def read_batch_func(data_iterator, batch_size):
      """Returns function for py_func that drives the graph."""
      iterator_lock = threading.Lock()

      def func():
        """Read input, batch it, and provide it to graph."""
        with iterator_lock:
          batch = [next(data_iterator) for _ in xrange(batch_size)]

        batch = [self.serialize_formula(f) for f in batch]
        batch_states = []
        for serialized in batch:
          state_machine = SerializedCNFStateMachine(self.func_arities)
          batch_states.append(state_machine.state)
          for token in serialized:
            state_machine.add_token(token)
            batch_states.append(state_machine.state)
        batch_states = np.array(batch_states, np.int32)

        batch = [[self.name_to_id[n] for n in ['<s>'] + f] for f in batch]
        max_length = max(len(item) for item in batch)
        input_ids = np.zeros((batch_size, max_length), np.int32)
        target_ids = []
        for i, serialized in enumerate(batch):
          input_ids[i, :len(serialized)] = serialized
          target_ids += serialized[1:] + [self.name_to_id['</s>']]

        target_ids = np.array(target_ids, np.int32)
        adjusted_target_ids = np.array(
            [ids_rev_maps[state][target_id]
             for state, target_id in zip(batch_states, target_ids)], np.int32)
        input_lengths = np.array(
            [len(serialized) for serialized in batch], dtype=np.int32)

        return (input_ids, target_ids, input_lengths, batch_states,
                adjusted_target_ids)

      return func

    py_func_results = tf.py_func(
        read_batch_func(data_iterator, hparams.batch_size), [],
        [tf.int32, tf.int32, tf.int32, tf.int32, tf.int32])
    input_ids, target_ids, input_lengths, cnf_states, adjusted_target_ids = (
        py_func_results)
    input_ids.set_shape([hparams.batch_size, None])

    with framework.arg_scope(
        [framework.variable],
        regularizer=layers.l2_regularizer(hparams.l2_weight)):
      def unit_cell():
        return tf.contrib.rnn.BasicLSTMCell(
            num_units=emb_size, state_is_tuple=True)

      cell = tf.contrib.rnn.MultiRNNCell(
          [unit_cell() for _ in xrange(hparams.depth)], state_is_tuple=True)

      options_per_partition = (
          (len(self.name_to_id) - 1) // hparams.emb_partitions + 1)
      embeddings = [tf.get_variable(
          'embs_%d' % i, [options_per_partition, emb_size],
          initializer=tf.random_uniform_initializer(-0.05, 0.05))
                    for i in xrange(hparams.emb_partitions)]

      input_embs = tf.nn.embedding_lookup(embeddings, input_ids)
      outputs, _ = tf.nn.dynamic_rnn(
          cell, input_embs, input_lengths, dtype=tf.float32)

      with tf.name_scope('gather_relevant'):
        relevant_indices = []
        max_length = tf.shape(input_ids)[1]
        for b in xrange(hparams.batch_size):
          base = b * max_length
          relevant_indices.append(tf.range(base, base + input_lengths[b]))
        relevant_indices = tf.concat(relevant_indices, 0)

        relevant_outputs = tf.gather(
            tf.reshape(outputs, [-1, emb_size]), relevant_indices)

      softmax_w = framework.variable('softmax_w',
                                     [emb_size, len(self.name_to_id)])
      softmax_b = framework.variable('softmax_b', [len(self.name_to_id)])
      logits = tf.matmul(relevant_outputs, softmax_w) + softmax_b

      # Create ops for sampling.
      self.single_state = nest.pack_sequence_as(
          structure=cell.state_size,
          flat_sequence=[
              tf.placeholder(tf.float32, [None, s])
              for s in nest.flatten(cell.state_size)
          ])
      self.state_size = cell.state_size
      self.single_input_ids = tf.placeholder(tf.int32, [None])
      single_input_embs = tf.nn.embedding_lookup(embeddings,
                                                 self.single_input_ids)
      with tf.variable_scope('rnn', reuse=True):
        single_outputs, self.single_new_state = cell(single_input_embs,
                                                     self.single_state)
      self.single_logits = tf.matmul(single_outputs, softmax_w) + softmax_b

    xent_cost = tf.reduce_sum(
        tf.contrib.nn.deprecated_flipped_sparse_softmax_cross_entropy_with_logits(  # pylint: disable=line-too-long
            logits, target_ids)) / hparams.batch_size

    masked_xent_costs = []
    logits_by_cnf_state = tf.dynamic_partition(
        logits, cnf_states, len(SerializedCNFStateMachine.ALL_STATES))
    target_ids_by_cnf_state = tf.dynamic_partition(
        adjusted_target_ids, cnf_states,
        len(SerializedCNFStateMachine.ALL_STATES))

    for state, (
        logits_for_state, target_ids_for_state
    ) in enumerate(zip(logits_by_cnf_state, target_ids_by_cnf_state)):
      # These default arguments are needed here because of how closures work in
      # Python. Without them, these variables would taken on the last value
      # they were assigned in the loop within all of the created closures, if
      # the closures were not evaluated until the loop was finished.
      # tf.cond currently evaluates the closures immediately which avoids this
      # problem, but we do this anyway to avoid linter errors (and in case
      # tf.cond changes in the future).
      def compute_masked_xent_cost(logits_for_state=logits_for_state,
                                   target_ids_for_state=target_ids_for_state,
                                   state=state):
        # TODO(ricshin): Replace this with an equivalent to tf.gather on the
        # second dimension once it exists.
        masked_logits_for_state = tf.transpose(
            tf.gather(tf.transpose(logits_for_state), ids_masks[state]))

        masked_xent_cost = (
            tf.contrib.nn.deprecated_flipped_sparse_softmax_cross_entropy_with_logits(  # pylint: disable=line-too-long
                masked_logits_for_state, target_ids_for_state))
        return tf.reduce_sum(masked_xent_cost)

      # Only compute the xent cost if logits_for_state actually contains
      # something, to work around bugs which occur when a tensor dimension
      # is 0.
      masked_xent_costs.append(
          tf.cond(
              tf.size(logits_for_state) > 0, compute_masked_xent_cost,
              lambda: tf.zeros([])))
    masked_xent_cost = tf.reduce_sum(masked_xent_costs) / hparams.batch_size

    self.metrics = {'xent_cost': xent_cost,
                    'log likelihood lower bound': masked_xent_cost}
    # Manual regularization losses are added for LSTM weights because
    # they do not use framework.variable.
    # TODO(ricshin): Update this if BasicLSTMCells are no longer used.
    self.loss = (
        masked_xent_cost if hparams.masked_xent else
        xent_cost + tf.add_n(losses.get_regularization_losses()) +
        hparams.l2_weight * tf.add_n([tf.nn.l2_loss(var)
                                      for var in tf.trainable_variables()
                                      if 'basic_lstm' in var.name]))

  @classmethod
  def serialize_formula(cls, formula):
    """Serialize a JSON-format formula into a sequence of tokens."""
    result = []
    for clause in formula['clauses']:
      result += cls.serialize_literal(clause)
    return result

  @classmethod
  def serialize_literal(cls, literal):
    """Serialize a literal into a sequence of tokens."""
    result = [] if literal['positive'] else ['~']
    return result + cls.serialize_function(literal)

  @classmethod
  def serialize_function(cls, function):
    """Serialize a function into a sequence of tokens."""
    if 'equal' in function:
      func_name = '__equal__'
      params = function['equal']
    elif 'func' in function:
      func_name = function['func']
      params = function['params']
    elif 'pred' in function:
      func_name = function['pred']
      params = function['params']

    return ['FUN_' + func_name] + nest.flatten(
        [cls.serialize_term(param) for param in params])

  @classmethod
  def serialize_term(cls, term):
    """Serialize a term into a sequence of tokens."""
    if 'func' in term:
      return cls.serialize_function(term)
    elif 'number' in term:
      return ['NUM_' + term['number']]
    elif 'var' in term:
      return ['VAR_' + term['var']]

  def unserialize_formula(self, serialized):
    """Convert a sequence of tokens into a JSON-format formula.

    Args:
      serialized: The sequence of tokens, as a collections.deque.
        It is mutated by this function.
    Returns:
      Formula in JSON format.
    """
    formula = {u'clauses': []}
    while serialized:
      formula['clauses'].append(self.unserialize_literal(serialized))
    return formula

  def unserialize_literal(self, serialized):
    """Convert a sequence of tokens into a literal."""
    if not serialized:
      raise ValueError('Ran out of tokens prematurely.')

    if serialized[0] == '~':
      positive = False
      serialized.popleft()
    else:
      positive = True
    literal = {u'positive': positive}
    return self.unserialize_function(serialized, literal)

  def unserialize_function(self, serialized, partial=None):
    """Convert a sequence of tokens into a literal."""
    if not serialized:
      raise ValueError('Ran out of tokens prematurely.')

    if partial is None:
      function = {}
    else:
      function = partial

    func_name = serialized.popleft()
    if not func_name.startswith('FUN_'):
      raise ValueError('Expected function name, got %s' % func_name)
    func_name = func_name[4:]

    is_predicate = self.func_name_to_is_predicate[func_name]
    num_arguments = self.func_arities[func_name]

    params = []
    if func_name == '__equal__':
      function[u'equal'] = params
    elif is_predicate:
      function[u'pred'] = func_name
      function[u'params'] = params
    else:
      function[u'func'] = func_name
      function[u'params'] = params

    for _ in xrange(num_arguments):
      params.append(self.unserialize_term(serialized))
    return function

  def unserialize_term(self, serialized):
    """Convert a sequence of tokens into a term."""
    if not serialized:
      raise ValueError('Ran out of tokens prematurely.')

    leftmost = serialized[0]
    if leftmost.startswith('FUN_'):
      return self.unserialize_function(serialized)
    elif leftmost.startswith('VAR_'):
      return {u'var': serialized.popleft()[4:]}
    elif leftmost.startswith('NUM_'):
      return {u'number': serialized.popleft()[4:]}
    else:
      raise ValueError('Invalid token for unserialize_term: %s' % leftmost)

  def sample(self, sess, temperature=1.0):
    """Sample a CNF statement from the trained model."""
    # TODO(ricshin): Handle batch sizes greater than 1.
    state = nest.pack_sequence_as(
        structure=self.state_size,
        flat_sequence=[np.zeros([1, s]) for s in nest.flatten(self.state_size)])
    input_id = self.name_to_id['<s>']

    # List of tokens that we have sampled.
    tokens = []

    state_machine = SerializedCNFStateMachine(self.func_arities)

    # Keeps track of how many arguments remain in the current function calls.
    num_remaining_args = []
    for _ in xrange(FLAGS.max_sequential_tokens):
      logits, state = sess.run([self.single_logits, self.single_new_state], {
          self.single_state: state,
          self.single_input_ids: [input_id],
      })

      id_subset = {state_machine.PRED_START: self.pred_start_ids,
                   state_machine.FUNC_CALL: self.func_ids,
                   state_machine.FUNC_ARG: self.arg_ids,
                   state_machine.PRED_END:
                       self.pred_end_ids}[state_machine.state]

      # Compute softmax in log space to avoid blowup to inf when temperature is
      # very low.
      logits = logits[0, id_subset] / temperature
      logits -= np.max(logits)
      logsumexp_logits = np.logaddexp.reduce(logits)
      probs = np.exp(logits - logsumexp_logits)
      if np.any(np.isnan(probs)):
        raise ValueError('NaN!')

      next_id = np.random.choice(id_subset, p=probs)
      if next_id == self.name_to_id['</s>']:
        assert not num_remaining_args
        break

      input_id = next_id
      next_name = self.id_to_name[next_id]
      tokens.append(next_name)
      state_machine.add_token(next_name)

    # Fix malformed sequence, if necessary
    for _ in xrange(sum(state_machine.num_remaining_args)):
      tokens.append(self.filler_number)

    return self.unserialize_formula(collections.deque(tokens))


class CNFTreeModel(object):
  """Builds the TF graph and feed dicts for the model."""

  @staticmethod
  def default_hparams():
    return tf.contrib.training.HParams(
        embedding_length=32,
        l2_weight=1e-6,
        batch_size=64,
        objective='fixed_z',
        mc_samples=1,
        min_kl=0.,
        min_kl_weight=0.1,
        kl_weight_start=5000,
        kl_weight_duration=20000,
        model_variants=[''],
        gate_type='sigmoid',
        gate_tied=False,
        act_fn='relu',
        highway_layers=0,
        op_hidden=0)

  def __init__(self, data_iterator, hparams, clause_metadata):
    """Constructor.

    Args:
      data_iterator: Python iterator which provides each instance as a dict.
      hparams: A HParams object with:
        - batch_size: Batch size to use.
        - embedding_length: Size of embedding used throughout the model.
        - objective: One of the following:
          - zero_z: Maximize p(x | z) where z is fixed to be 0.
          - fixed_z: Maximize p(x | z) where z is a learned fixed vector.
          - naive_lb*: Maximize ∑_{z_i} log p(x | z_i), z_i ~ N(0, 1).
          - mc_lb*: Maximize log ∑_{z_i} p(x | z_i), z_i ~ N(0, 1).
          - vae: Variational auto-encoder objective. Analogous to naive_lb.
          - vae_mix: Variational auto-encoder objective,
              with KL(p(z) || q(z)) instead of KL(q(z | x_i) || p(z)).
              KL(p(z) || q(z)) is computed by taking q(z) ≈ mean(q(z | x_i))
              over the minibatch and using a variational estimate for
              the KL between two Gaussian mixtures (see Hershey and Olsen),
              which is log N - log ∑ exp(-KL(p(z) || q(z_i))).
              Unfortunately, this doesn't seem to learn something useful.
          - iwae: Importance-weighted auto-encoder objective. Like mc_lb.
        - mc_samples: Number of Monte Carlo samples to take while estimating
            expectations.
        - l2_weight: Amount of weight to put on the L2 regularization.
        - min_kl: Set a floor on the KL divergence term to ensure that the model
            uses at least that much information.
        - min_kl_weight: Minimum weight placed on the KL penalty term. Set to 1
            to disable the gradual increase of KL weight over time.
        - kl_weight_start: Approx number of steps before increasing KL
            posterior-prior cost weight.
        - kl_weight_duration: Approx number of steps to cover while increasing
            KL cost weight from 0 to 1.
        - model_variants: List of strings. Possible values:
          - aux_lstm: Use an auxiliary LSTM on the side.
          - gated: Use an update-gate unit for merging/splitting embeddings.
          - layer_norm: Apply layer normalization to each output.
          - rev_read: In read_*, read lists of things right-to-left, analogous
              to how input sequences are reversed in seq2seq models.
          - uncond_sib: Function names and literal signs don't influence
                        the embeddings used to generate their siblings.
          Unimplemented:
          - int_rand*: Random internal states.
          - uniform_ops*: Use same split/merge ops everywhere.
          - mult_int*: Multiplicative integration (arxiv 1606.06630).
        - gate_type: 'sigmoid' or 'softmax' for weighting constituents.
        - gate_tied: Whether input weights should be the same across outputs.
        - act_fn: Activation function to apply. relu, elu, tanh, sigmoid.
        - highway_layers: Extra highway-network layers adjacent to z.
        - op_hidden: Number of hidden layers within each linear op.
        * Not fully implemented yet
      clause_metadata: A dictionary containing the following:
        - func_names_and_types: List of all (function/predicate names,
          is_predicate) pairs.
        - func_arities: Dictionary from function/predicate names to their
          arities.
        - var_names: List of all variables in the data.
        - numbers: Numbers which appear in the data.

    Raises:
      ValueError: Invalid value in hparams.
    """
    # Twiddle hparams
    batch_size = hparams.batch_size
    emb_size = self.emb_size = hparams.embedding_length
    self.model_variants = set(hparams.model_variants)
    self.hparams = hparams
    use_layer_norm = 'layer_norm' in self.model_variants
    try:
      act_fn = {
          'relu': tf.nn.relu,
          'elu': tf.nn.elu,
          'tanh': tf.tanh,
          'sigmoid': tf.sigmoid
      }[hparams.act_fn]
    except KeyError:
      raise ValueError('Invalid act_fn: %s' % act_fn)

    func_names_and_types, func_arities, var_names, numbers = (
        clause_metadata['func_names'], clause_metadata['func_arities'],
        clause_metadata['var_names'], clause_metadata['numbers'])

    with framework.arg_scope(
        [framework.variable],
        regularizer=layers.l2_regularizer(hparams.l2_weight)):
      named_tensors = {}
      if hparams.objective in ('naive_lb', 'mc_lb', 'vae', 'vae_mix', 'iwae'):
        named_tensors['root_emb'] = tf.random_normal([emb_size])
      elif hparams.objective == 'fixed_z':
        named_tensors['root_emb'] = framework.variable(
            'root_emb', [emb_size],
            initializer=tf.random_uniform_initializer(-0.05, 0.05))
      elif hparams.objective == 'zero_z':
        named_tensors['root_emb'] = tf.zeros([emb_size])
      else:
        raise ValueError('Invalid objective: %s' % hparams.objective)
      named_tensors['zero_emb'] = tf.zeros([emb_size])
      named_tensors['log_mc_samples'] = tf.constant(
          np.log(hparams.mc_samples), dtype=tf.float32)

      named_ops = {}

      if 'gated' in self.model_variants:
        split = functools.partial(
            loom_ops.split_gated,
            emb_size=emb_size,
            hidden_outputs=(1,) * hparams.op_hidden,
            layer_norm=use_layer_norm,
            gate_type=hparams.gate_type,
            tied_gates=hparams.gate_tied,
            activation=act_fn)
        merge = functools.partial(
            loom_ops.merge_gated,
            emb_size=emb_size,
            hidden_outputs=(1,) * hparams.op_hidden,
            layer_norm=use_layer_norm,
            gate_type=hparams.gate_type,
            tied_gates=hparams.gate_tied,
            activation=act_fn)
      else:
        split = functools.partial(
            loom_ops.split,
            emb_size=emb_size,
            hidden_sizes=(emb_size,) * hparams.op_hidden,
            layer_norm=use_layer_norm,
            activation=act_fn)
        merge = functools.partial(
            loom_ops.merge,
            emb_size=emb_size,
            hidden_sizes=(emb_size,) * hparams.op_hidden,
            layer_norm=use_layer_norm,
            activation=act_fn)

      # which should never use layer norm, because the output goes directly to a
      # softmax.
      which = functools.partial(
          loom_ops.which,
          emb_size=emb_size,
          hidden_sizes=(emb_size,) * hparams.op_hidden,
          hidden_activation=act_fn,
          layer_norm=False)

      if hparams.highway_layers > 0:
        highway = functools.partial(
            loom_ops.GatedLinearLoomOp,
            emb_size=emb_size,
            num_inputs=1,
            num_outputs=1,
            hidden_outputs=(1,) * (hparams.highway_layers - 1),
            activation=act_fn,
            layer_norm=False,
            gate_type='softmax')
        named_ops['decode_z'] = highway('decode_z')
        named_ops['encode_z'] = highway('encode_z')

      # 3 outputs from split_rec_cnf:
      # - Input to next split_rec_cnf
      # - Input to split_literal
      # - Decide whether to generate a new literal
      named_ops['split_rec_cnf'] = split('split_rec_cnf', 3)
      named_ops['which_stop'] = which('which_stop', 2)

      # 2 outputs from split_literal:
      # - Decide positive or negative
      # - Input to split_rec_func_args and merge_func_emb
      named_ops['which_sign'] = which('which_sign', 2)

      named_ops['emb_lookup_sign'] = loom_ops.EmbeddingLookupLoomOp(
          'emb_lookup_sign', 2, emb_size)

      num_functions = len(func_names_and_types)
      named_ops['which_func'] = which('which_func', num_functions)
      named_ops['emb_lookup_func'] = loom_ops.EmbeddingLookupLoomOp(
          'emb_lookup_func', num_functions, emb_size)

      if 'uncond_sib' not in self.model_variants:
        named_ops['merge_sign'] = merge('merge_sign', 2)
        named_ops['merge_func_emb'] = merge('merge_func_emb', 2)

      # 2 outputs from split_rec_func_args:
      # - Input to next split_rec_func_args
      # - Input to split_term
      named_ops['split_rec_func_args'] = split('split_rec_func_args', 2)

      # Decide between Function, Variable, Number
      named_ops['which_term_type'] = which('which_term_type', 3)
      named_ops['emb_lookup_term_type'] = loom_ops.EmbeddingLookupLoomOp(
          'emb_lookup_term_type', 3, emb_size)

      # Select variable
      num_variables = len(var_names)
      named_ops['which_variable'] = which('which_variable', num_variables)

      # Select number
      num_numbers = len(numbers)
      named_ops['which_number'] = which('which_number', num_numbers)

      # Compute cross-entropy loss, and sampling ops
      def choice(name, count):
        named_ops['xent_' +
                  name] = loom_ops.SparseSoftmaxCrossEntropyLossLoomOp(count)
        named_ops['sample_' + name] = loom_ops.MultinomialLoomOp(count)

      choice('stop', 2)
      choice('sign', 2)
      choice('func', num_functions)
      choice('term_type', 3)
      choice('number', num_numbers)
      choice('variable', num_variables)

      if 'aux_lstm' in self.model_variants:
        named_ops['lstm'] = loom_ops.LSTMLoomOp(
            'lstm', emb_size, layer_norm=use_layer_norm)
        named_ops['merge_lstm_h'] = merge('merge_lstm_h', 2)
        self.aux_lstm = self.aux_lstm_enabled
        self.eval_lstm_state = self.eval_lstm_state_enabled
      else:
        self.aux_lstm = self.aux_lstm_noop
        self.eval_lstm_state = self.eval_lstm_state_noop

      # Ops for reading and summarizing a tree
      named_ops['add_embs'] = loom_ops.BinaryLoomOp(tf.add,
                                                    loom.TypeShape(tf.float32,
                                                                   (emb_size,)))
      named_ops['div_emb'] = loom_ops.BinaryLoomOp(tf.div,
                                                   loom.TypeShape(tf.float32,
                                                                  (emb_size,)),
                                                   loom.TypeShape(tf.float32,
                                                                  (1,)))
      named_ops['merge_literals'] = merge('merge_literals', 2)
      named_ops['merge_sign_pred_embs'] = merge('merge_sign_pred_embs', 2)
      named_ops['merge_function_args'] = merge('merge_function_args', 2)
      named_ops['emb_lookup_number'] = loom_ops.EmbeddingLookupLoomOp(
          'emb_lookup_number', num_numbers, emb_size)
      named_ops['emb_lookup_variable'] = loom_ops.EmbeddingLookupLoomOp(
          'emb_lookup_variable', num_variables, emb_size)

      if hparams.objective in ('vae', 'vae_mix', 'iwae'):
        # Ops for autoencoder training
        named_ops['kl_cost'] = loom_ops.KLDivPosteriorPriorLoomOp(
            emb_size, reverse=hparams.objective == 'vae_mix')
        named_ops['sample_z'] = loom_ops.SampleFromGaussianLoomOp(emb_size)

        if hparams.objective == 'iwae':
          named_ops['detag_xent'] = loom_ops.TagLoomOp(
              tf.float32, (), input_tag='xent_cost', output_tag='')
          named_ops['tag_elbo'] = loom_ops.TagLoomOp(
              tf.float32, (), input_tag='', output_tag='elbo')
          named_ops['add_scalar'] = loom_ops.BinaryLoomOp(
              tf.add, loom.TypeShape(tf.float32, ()))
          named_ops['sub_scalar'] = loom_ops.BinaryLoomOp(
              tf.subtract, loom.TypeShape(tf.float32, ()))
          named_ops['max_scalar'] = loom_ops.BinaryLoomOp(
              tf.maximum, loom.TypeShape(tf.float32, ()))
          named_ops['exp_scalar'] = loom_ops.UnaryLoomOp(
              tf.exp, loom.TypeShape(tf.float32, ()))
          named_ops['log_scalar'] = loom_ops.UnaryLoomOp(
              tf.log, loom.TypeShape(tf.float32, ()))
          named_ops['neg_scalar'] = loom_ops.UnaryLoomOp(
              tf.negative, loom.TypeShape(tf.float32, ()))

        named_ops['log_p_z'] = loom_ops.LogPdfWithStandardNormalLoomOp(emb_size)
        named_ops['neg_log_q_z_x'] = (
            loom_ops.NegativeLogPdfWithMultivariateNormalLoomOp(emb_size))

      def read_batch_func(data_iterator, batch_size):
        iterator_lock = threading.Lock()

        def func():
          with iterator_lock:
            batch = [next(data_iterator) for _ in xrange(batch_size)]
          return np.array(self.prepare_loom_input(batch).serialize())

        return func

      self.batch_input, = tf.py_func(
          read_batch_func(data_iterator, batch_size), [], [tf.string])

      self.loom = loom.Loom(
          None, named_tensors, named_ops, loom_input_tensor=self.batch_input)

    func_names = [x[0] for x in func_names_and_types]

    self.func_name_to_id = dict(zip(func_names, itertools.count(0)))
    self.func_id_to_name = dict(zip(itertools.count(0), func_names))
    self.func_name_to_is_predicate = dict(func_names_and_types)
    self.var_name_to_id = dict(zip(var_names, itertools.count(0)))
    self.var_id_to_name = dict(zip(itertools.count(0), var_names))
    self.number_to_id = dict(zip(numbers, itertools.count(0)))
    self.id_to_number = dict(zip(itertools.count(0), numbers))
    self.func_arities = func_arities

    # Ultimate loss
    xent_cost = (tf.reduce_sum(
        self.loom.output_tensor(loom.TypeShape(tf.float32, [], 'xent_cost'))) /
                 hparams.batch_size)

    self.metrics = {'xent_cost': xent_cost}
    self.loss = xent_cost

    if hparams.objective in ('zero_z', 'fixed_z'):
      self.metrics['log likelihood lower bound'] = xent_cost
    elif hparams.objective in ('vae', 'vae_mix'):
      if hparams.objective == 'vae':
        kl_div = tf.reduce_mean(
            self.loom.output_tensor(loom.TypeShape(tf.float32, [], 'kl_div')))
      else:
        real_kl_div = tf.reduce_mean(
            self.loom.output_tensor(loom.TypeShape(tf.float32, [], 'kl_div')))
        kl_div = tf.log(np.float32(hparams.batch_size)) - loom_ops.logsumexp(
            -self.loom.output_tensor(
                loom.TypeShape(tf.float32, (), 'rev_kl_div')))

      global_step = framework.get_or_create_global_step()
      # sigmoid_divisor = 10 was determined by eyeballing.
      # sigmoid(-5) = 0.007
      # sigmoid(5)  = 0.993
      sigmoid_offset = hparams.kl_weight_start + hparams.kl_weight_duration / 2.
      sigmoid_divisor = hparams.kl_weight_duration / 10.
      if hparams.min_kl_weight >= 1:
        kl_cost_weight = 1
      else:
        kl_cost_weight = hparams.min_kl_weight + (
            1 - hparams.min_kl_weight) * tf.sigmoid((tf.cast(
                global_step, tf.float32) - sigmoid_offset) / sigmoid_divisor)

      self.metrics['kl_div'] = kl_div
      self.metrics['kl_cost_weight'] = kl_cost_weight
      # TOOD(ricshIn): Double check that this handling of min_kl is correct.
      # Alternatives:
      # - Apply min_kl to each of the dimensions of the Gaussian?
      # - Apply min_kl before using reduce_mean over the batch
      self.loss += tf.maximum(kl_div, hparams.min_kl) * kl_cost_weight

      if hparams.objective == 'vae':
        self.metrics['log likelihood lower bound'] = xent_cost + kl_div
      else:
        self.metrics['log likelihood lower bound'] = xent_cost + real_kl_div
    elif hparams.objective == 'iwae':
      if hparams.min_kl_weight != 1:
        logging.warn('min_kl_weight (%f) != 1 unsupported with IWAE',
                     hparams.min_kl_weight)
      if hparams.min_kl != 0:
        logging.warn('min_kl (%f) != 0 unsupported with IWAE', hparams.min_kl)

      self.loss = (tf.reduce_sum(
          self.loom.output_tensor(loom.TypeShape(tf.float32, (), 'elbo'))) /
                   hparams.batch_size)

      kl_div = tf.reduce_mean(
          self.loom.output_tensor(loom.TypeShape(tf.float32, [], 'kl_div')))
      self.metrics['kl_div'] = kl_div
      self.metrics['kl_cost_weight'] = 1
      self.metrics['log likelihood lower bound'] = self.loss

    self.loss += tf.add_n(losses.get_regularization_losses())

  def prepare_loom_input(self, formulas):
    """Create a Loom input for the formulas."""
    loom_input = LoomInputWithExtras(self.loom.make_weaver())
    for formula in formulas:
      if self.hparams.objective in ('naive_lb', 'mc_lb'):
        raise NotImplementedError

      elif self.hparams.objective in ('vae', 'vae_mix', 'iwae'):
        formula_emb = self.read_formula(loom_input, formula)
        if self.hparams.objective in ('vae', 'iwae'):
          # kl_cost is unused for iwae because the stochastic KL cost will be
          # computed later.
          kl_cost, z_mean, z_stdev = loom_input.kl_cost(formula_emb)
          loom_input.add_output(kl_cost)
        elif self.hparams.objective == 'vae_mix':
          kl_cost, rev_kl_cost, z_mean, z_stdev = loom_input.kl_cost(
              formula_emb)
          loom_input.add_output(kl_cost)
          loom_input.add_output(rev_kl_cost)

        z_samples = [loom_input.sample_z(z_mean, z_stdev)
                     for _ in xrange(self.hparams.mc_samples)]
        log_iw_ps = []
        for z_sample in z_samples:
          loom_input.reset_xent_costs()
          self.build_formula(loom_input, formula, z_sample)

          if self.hparams.objective == 'iwae':
            # log [p(x | z_sample) p(z_sample) / q(z_sample | x)]
            #  = log p(x | z_sample) + log p(z_sample) - log q(z_sample | x)
            log_iw_p = loom_ops.reduce_loom(
                loom_input.add_scalar,
                [loom_input.neg_scalar(loom_input.detag_xent(c))
                 for c in loom_input.xent_costs] +
                [loom_input.log_p_z(z_sample), loom_input.neg_log_q_z_x(
                    z_mean, z_stdev, z_sample)])
            log_iw_ps.append(log_iw_p)

        if self.hparams.objective == 'iwae':
          # log 1/k ∑_{z_sample} exp log [
          #       p(x | z_sample) p(z_sample) / q(z_sample | x)]
          #   where z_sample ~ q(z_sample | x)
          # ≈ log p(x)
          log_iw_p = loom_ops.logsumexp_loom(log_iw_ps, loom_input.max_scalar,
                                             loom_input.add_scalar,
                                             loom_input.sub_scalar,
                                             loom_input.exp_scalar,
                                             loom_input.log_scalar)
          log_iw_p = loom_input.sub_scalar(
              log_iw_p, loom_input.named_tensor('log_mc_samples'))
          loom_input.add_output(
              loom_input.tag_elbo(loom_input.neg_scalar(log_iw_p)))

      elif self.hparams.objective in ('zero_z', 'fixed_z'):
        self.build_formula(loom_input, formula,
                           loom_input.named_tensor('root_emb'))
    return loom_input

  @contextlib.contextmanager
  def aux_lstm_enabled(self, emb, loom_input, context, get_input=True):
    """Get embeddings to and from the auxiliary LSTM.

    The LSTM state is stored in `context['lstm_state']`.  Within the
    tree-construction model, they are `LoomResult`s.  While sampling, they are
    either NumPy arrays or `LoomResult`s in the current `loom_input`.

    Args:
      emb: Embedding to be added to the LSTM state.
      loom_input: A LoomInput object.
      context: Where the LSTM state is stored.
      get_input: Whether the result embedding should include state from the
                 LSTM.

    Yields:
      Embedding (with LSTM state added if `get_input` is true), and a function
      to call in order to add an embedding into the LSTM's state.
    """

    if not isinstance(context['lstm_state'].h, six.integer_types):
      context['lstm_state'] = LSTMStateTuple(*[loom_input.constant(v)
                                               for v in context['lstm_state']])

    if get_input:
      args = [context['lstm_state'].h, emb]
      new_emb = loom_input.merge_lstm_h(*args)
    else:
      new_emb = emb

    def add_to_lstm(emb):
      # `emb` is a function so that it is evaluated only if needed.
      if context is None:
        loom_input.lstm_state = LSTMStateTuple(*loom_input.lstm(
            emb(), *loom_input.lstm_state))
      else:
        context['lstm_state'] = LSTMStateTuple(*loom_input.lstm(
            emb(), *context['lstm_state']))

    yield new_emb, add_to_lstm

  @contextlib.contextmanager
  def aux_lstm_noop(self, emb, *unused_args, **unused_kwargs):
    yield emb, lambda x: None

  @contextlib.contextmanager
  def eval_lstm_state_enabled(self, sess, sampling_ctx, context):
    if all(isinstance(v, six.integer_types) for v in context['lstm_state']):
      lstm_state = sampling_ctx.register(context['lstm_state'])
      yield
      context['lstm_state'] = LSTMStateTuple(*sampling_ctx.run_and_insert(
          sess, lstm_state))
    else:
      # No need to fetch the LSTM state, as it's already concrete.
      yield

  @contextlib.contextmanager
  def eval_lstm_state_noop(self, *unused_args, **unused_kwargs):
    yield

  def build_formula(self, loom_input, formula, emb):
    """Add formula to loom_input."""
    context = {}
    if 'aux_lstm' in self.model_variants:
      context['lstm_state'] = LSTMStateTuple(
          loom_input.named_tensor('zero_emb'),
          loom_input.named_tensor('zero_emb'))

    if self.hparams.highway_layers > 0:
      emb = loom_input.decode_z(emb)

    prev_state = emb

    for i, clause in enumerate(formula['clauses']):
      next_state, literal_emb, stop_emb = loom_input.split_rec_cnf(prev_state)
      # TODO(ricshin):
      # Use a sigmoid to decide whether to stop. Don't use stop_emb.
      loom_input.supervise_choice('stop', loom_input.which_stop(stop_emb),
                                  int(i + 1 != len(formula['clauses'])))

      self.build_literal(loom_input, clause, literal_emb, context)

      prev_state = next_state

  def build_literal(self, loom_input, literal, emb, context):
    """Add literal to loom_input."""

    with self.aux_lstm(emb, loom_input, context) as (emb, add_to_lstm):
      loom_input.supervise_choice('sign', loom_input.which_sign(emb),
                                  int(literal['positive']))
      # TODO(ricshin): Avoid computing sign_emb if it's not needed.
      sign_emb = loom_input.emb_lookup_sign(
          loom_input.constant(np.int64(literal['positive'])))
      add_to_lstm(lambda: sign_emb)

    if 'uncond_sib' in self.model_variants:
      function_emb = emb
    else:
      function_emb = loom_input.merge_sign(sign_emb, emb)
    self.build_function(loom_input, literal, function_emb, context)

  def build_function(self, loom_input, function, emb, context):
    """Add function to loom_input."""
    if 'equal' in function:
      func_name = '__equal__'
      params = function['equal']
    elif 'func' in function:
      func_name = function['func']
      params = function['params']
    elif 'pred' in function:
      func_name = function['pred']
      params = function['params']

    with self.aux_lstm(emb, loom_input, context) as (emb, add_to_lstm):
      loom_input.supervise_choice('func', loom_input.which_func(emb),
                                  self.func_name_to_id[func_name])
      # TODO(ricshin): Avoid computing func_emb if it's not needed.
      func_emb = loom_input.emb_lookup_func(
          loom_input.constant(np.int64(self.func_name_to_id[func_name])))
      add_to_lstm(lambda: func_emb)

    if 'uncond_sib' in self.model_variants:
      prev_state = emb
    else:
      prev_state = loom_input.merge_func_emb(func_emb, emb)

    for i, param in enumerate(params):
      if i > 0 or 'uncond_sib' in self.model_variants:
        with self.aux_lstm(prev_state, loom_input, context) as (
            lstm_added_state, _):
          prev_state = lstm_added_state
      next_state, param_emb = loom_input.split_rec_func_args(prev_state)
      self.build_term(loom_input, param, param_emb, context)
      prev_state = next_state

  def build_term(self, loom_input, term, emb, context):
    """Add term to loom_input."""
    # pylint: disable=g-long-lambda

    # func: 0, number: 1, var: 2
    term_type_id = 1 * ('number' in term) + 2 * ('var' in term)

    with self.aux_lstm(emb, loom_input, context) as (emb, add_to_lstm):
      loom_input.supervise_choice('term_type', loom_input.which_term_type(emb),
                                  term_type_id)

      add_to_lstm(lambda: loom_input.emb_lookup_term_type(
          loom_input.constant(np.int64(term_type_id))))

    if 'func' in term:
      self.build_function(loom_input, term, emb, context)
    elif 'number' in term:
      number_id = self.number_to_id[term['number']]

      with self.aux_lstm(emb, loom_input, context) as (emb, add_to_lstm):
        loom_input.supervise_choice('number', loom_input.which_number(emb),
                                    number_id)
        add_to_lstm(lambda: loom_input.emb_lookup_number(
            loom_input.constant(np.int64(number_id))))

    elif 'var' in term:
      variable_id = self.var_name_to_id[term['var']]

      with self.aux_lstm(emb, loom_input, context) as (emb, add_to_lstm):
        loom_input.supervise_choice('variable', loom_input.which_variable(emb),
                                    variable_id)
        add_to_lstm(lambda: loom_input.emb_lookup_variable(
            loom_input.constant(np.int64(variable_id))))

  def read_formula(self, loom_input, formula):
    """Return a mean of the embeddings of the constituent literals."""
    literal_embs = [self.read_literal(loom_input, literal)
                    for literal in formula['clauses']]
    if 'rev_read' in self.model_variants:
      prev_state = loom_input.named_tensor('zero_emb')
      for literal_emb in reversed(literal_embs):
        prev_state = loom_input.merge_literals(prev_state, literal_emb)
    else:
      # Take the mean of the embeddings.
      # The rationale was that the ordering of the literals should be
      # irrelevant, but we actually want to be able to reconstruct the original
      # formula from the result, including the correct ordering. Therefore
      # combining the embeddings in an order-agnostic way like this is
      # probably quite bad.
      prev_state = loom_input.div_emb(
          loom_ops.reduce_loom(loom_input.add_embs, literal_embs),
          loom_input.constant(
              np.array(
                  [len(formula['clauses'])], dtype=np.float32)))

    if self.hparams.highway_layers > 0:
      return loom_input.encode_z(prev_state)
    else:
      return prev_state

  def read_literal(self, loom_input, literal):
    sign_emb = loom_input.emb_lookup_sign(
        loom_input.constant(np.int64(literal['positive'])))
    return loom_input.merge_sign_pred_embs(
        sign_emb, self.read_function(loom_input, literal))

  def read_function(self, loom_input, function):
    """Summarize a function invocation into an embedding."""
    if 'equal' in function:
      func_name = '__equal__'
      params = function['equal']
    elif 'func' in function:
      func_name = function['func']
      params = function['params']
    elif 'pred' in function:
      func_name = function['pred']
      params = function['params']

    func_emb = loom_input.emb_lookup_func(
        loom_input.constant(np.int64(self.func_name_to_id[func_name])))

    if 'rev_read' in self.model_variants:
      prev_state = loom_input.named_tensor('zero_emb')
      for param in reversed(params):
        prev_state = loom_input.merge_function_args(
            prev_state, self.read_term(loom_input, param))
      return loom_input.merge_function_args(prev_state, func_emb)
    else:
      prev_state = func_emb
      for param in params:
        prev_state = loom_input.merge_function_args(
            prev_state, self.read_term(loom_input, param))
      return prev_state

  def read_term(self, loom_input, term):
    if 'func' in term:
      return self.read_function(loom_input, term)
    elif 'number' in term:
      return loom_input.emb_lookup_number(
          loom_input.constant(np.int64(self.number_to_id[term['number']])))
    elif 'var' in term:
      return loom_input.emb_lookup_variable(
          loom_input.constant(np.int64(self.var_name_to_id[term['var']])))

  def sample(self, sess, temperature=1.0):
    """Sample a CNF statement from the trained model."""
    # TODO(ricshin): Sample with more batching for greater parallelism, both
    # while sampling one formula and while sampling multiple simultaneously.
    # Currently each decision is made entirely sequentially.

    prev_state = None
    generate_additional_literals = True
    literals = []
    temperature = np.array(temperature, np.float32)

    context = {'temp': temperature}
    if 'aux_lstm' in self.model_variants:
      context['lstm_state'] = LSTMStateTuple(
          np.zeros(
              [self.emb_size], dtype=np.float32),
          np.zeros(
              [self.emb_size], dtype=np.float32))

    while generate_additional_literals:
      loom_input = self.loom.make_weaver()
      sampling_ctx = SamplingContext(self.loom, loom_input)
      prev_state = (loom_input.root_emb if prev_state is None else
                    loom_input.constant(prev_state))

      next_state, literal_emb, stop_emb = loom_input.split_rec_cnf(prev_state)
      next_state = sampling_ctx.register(next_state)
      literal_emb = sampling_ctx.register(literal_emb)

      # Decide if we're stopping here
      stop = loom_input.which_stop(stop_emb)
      sampled_stop = loom_input.sample_stop(
          stop, loom_input.constant(context['temp']))
      sampled_stop = sampling_ctx.register(sampled_stop)

      (next_state_val, literal_emb_val,
       generate_additional_literals) = sampling_ctx.run_and_insert(
           sess, [next_state, literal_emb, sampled_stop])

      # Generate the literal
      literals.append(self.sample_literal(sess, literal_emb_val, context))

      # Move onto the next state
      prev_state = next_state_val

      if len(literals) > FLAGS.max_literals_in_clause:
        break

    return {u'clauses': literals}

  def sample_literal(self, sess, emb, context):
    """Sample one literal within a CNF statement."""
    loom_input = self.loom.make_weaver()
    sampling_ctx = SamplingContext(self.loom, loom_input)
    emb = loom_input.constant(emb)

    with self.aux_lstm(emb, loom_input, context=context) as (emb, add_to_lstm):
      sign = loom_input.which_sign(emb)
      sampled_sign = loom_input.sample_sign(
          sign, loom_input.constant(context['temp']))
      sign_emb = loom_input.emb_lookup_sign(sampled_sign)
      add_to_lstm(lambda: sign_emb)

    if 'uncond_sib' in self.model_variants:
      function_emb = emb
    else:
      function_emb = loom_input.merge_sign(sign_emb, emb)
    # pylint: disable=unnecessary-lambda
    literal = {u'positive': sampling_ctx.register(sampled_sign,
                                                  lambda arr: bool(arr))}
    function = self.sample_function(sess, function_emb, loom_input,
                                    sampling_ctx, 1, context)
    literal.update(function)

    return sampling_ctx.run_and_insert(sess, literal)

  def sample_function(self, sess, emb, loom_input, sampling_ctx, depth,
                      context):
    """Sample one function/predicate call."""
    if depth > FLAGS.max_function_depth:
      return {'error': 'Maximum depth exceeded'}

    function = {}

    with self.aux_lstm(emb, loom_input, context=context) as (emb, add_to_lstm):
      func = loom_input.which_func(emb)
      sampled_func = loom_input.sample_func(
          func, loom_input.constant(context['temp']))
      func_emb = loom_input.emb_lookup_func(sampled_func)
      add_to_lstm(lambda: func_emb)

    if 'uncond_sib' in self.model_variants:
      merged_emb = emb
    else:
      merged_emb = loom_input.merge_func_emb(func_emb, emb)

    func_name = sampling_ctx.register(sampled_func,
                                      lambda arr: self.func_id_to_name[arr])
    merged_emb = sampling_ctx.register(merged_emb)

    with self.eval_lstm_state(sess, sampling_ctx, context):
      func_name_val, merged_emb_val = sampling_ctx.run_and_insert(
          sess, [func_name, merged_emb])

    num_arguments = self.func_arities[func_name_val]
    is_predicate = self.func_name_to_is_predicate[func_name_val]

    params = []
    if func_name_val == '__equal__':
      function[u'equal'] = params
    elif is_predicate:
      function[u'pred'] = func_name_val
      function[u'params'] = params
    else:
      function[u'func'] = func_name_val
      function[u'params'] = params

    # Now generate each argument.
    prev_state = None
    for i in xrange(num_arguments):
      loom_input = self.loom.make_weaver()
      sampling_ctx = SamplingContext(self.loom, loom_input)
      prev_state = (loom_input.constant(merged_emb_val) if prev_state is None
                    else loom_input.constant(prev_state))
      if i > 0:
        with self.aux_lstm(
            prev_state, loom_input, context=context) as (lstm_added_state, _):
          prev_state = lstm_added_state

      next_state, param_emb = loom_input.split_rec_func_args(prev_state)
      next_state = sampling_ctx.register(next_state)
      param_emb = sampling_ctx.register(param_emb)

      with self.eval_lstm_state(sess, sampling_ctx, context):
        next_state_val, param_emb_val = sampling_ctx.run_and_insert(
            sess, [next_state, param_emb])

      sampled_term = self.sample_term(sess, param_emb_val, depth, context)
      params.append(sampled_term)
      prev_state = next_state_val

    return function

  def sample_term(self, sess, emb, depth, context):
    """Sample one term (argument to function or predicate)."""
    loom_input = self.loom.make_weaver()
    sampling_ctx = SamplingContext(self.loom, loom_input)
    emb = loom_input.constant(emb)

    with self.aux_lstm(emb, loom_input, context=context) as (emb, add_to_lstm):
      term_type_id = loom_input.sample_term_type(
          loom_input.which_term_type(emb), loom_input.constant(context['temp']))
      add_to_lstm(lambda: loom_input.emb_lookup_term_type(term_type_id))

    term_type_placeholder = sampling_ctx.register(
        term_type_id, lambda arr: ('func', 'number', 'var')[arr])
    emb_placeholder = sampling_ctx.register(emb)

    with self.eval_lstm_state(sess, sampling_ctx, context):
      term_type_val, emb_val = sampling_ctx.run_and_insert(
          sess, [term_type_placeholder, emb_placeholder])

    loom_input = self.loom.make_weaver()
    sampling_ctx = SamplingContext(self.loom, loom_input)
    emb = loom_input.constant(emb_val)
    if term_type_val == 'func':
      return self.sample_function(sess, emb, loom_input, sampling_ctx,
                                  depth + 1, context)
    elif term_type_val == 'number':
      with self.aux_lstm(
          emb, loom_input, context=context) as (emb, add_to_lstm):
        number_id = loom_input.sample_number(
            loom_input.which_number(emb), loom_input.constant(context['temp']))
        add_to_lstm(lambda: loom_input.emb_lookup_number(number_id))
      number_placeholder = sampling_ctx.register(
          number_id, lambda arr: self.id_to_number[arr])
      with self.eval_lstm_state(sess, sampling_ctx, context):
        result = {u'number': sampling_ctx.run_and_insert(sess,
                                                         number_placeholder)}
      return result
    elif term_type_val == 'var':
      with self.aux_lstm(
          emb, loom_input, context=context) as (emb, add_to_lstm):
        variable_id = loom_input.sample_variable(
            loom_input.which_variable(emb),
            loom_input.constant(context['temp']))
        add_to_lstm(lambda: loom_input.emb_lookup_variable(variable_id))

      var_placeholder = sampling_ctx.register(
          variable_id, lambda arr: self.var_id_to_name[arr])

      with self.eval_lstm_state(sess, sampling_ctx, context):
        result = {u'var': sampling_ctx.run_and_insert(sess, var_placeholder)}
      return result
    else:
      raise ValueError('Invalid term_type: %r' % term_type_val)
