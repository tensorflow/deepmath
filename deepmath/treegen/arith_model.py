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
"""A generative model for arithmetic expressions using TensorLoom.

Numbers are between 0 and 9, and only + and - are allowed.
Examples:
1 + 2 -> ['plus', ['number', [1]], ['number', [2]]] (depth 2)
3 - (4 + 5) ->
  ['minus', ['number', [3]], ['plus', ['number', [4]], ['number', [5]]]]
  (depth 3)
6 -> ['number', [6]] (depth 1)

The model works like this: we receive an embedding from above for the current
node. We use a softmax to decide if this node should be 'plus', 'minus', or
'number'. If 'plus' or 'minus', we multiply the embedding with a N x 2N matrix
to get two embeddings; these are used to determine the values of the two
children.  If 'number', we multiply the embedding with a N x N matrix, and then
use the result to choose between 0 and 9 by passing the result through a softmax
and using the resulting probabilities.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow_fold.public import loom
from deepmath.treegen import loom_ops
from tensorflow.python.util import nest

slim = tf.contrib.slim

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_sample_depth', 6,
                     'Maximum depth to reach while sampling from the model.')
flags.DEFINE_string('root_emb',
                    'static',  # ['static', 'random', 'vae']
                    'How to generate the root embedding.')


class TreeNode(object):

  def __init__(self, value=None, parent=None, embedding=None, children=None):
    self.value = value
    self.parent = parent
    self.embedding = embedding
    self.children = [] if children is None else children

  def extract(self):
    """Return the tree rooted at this node in list form."""
    return (self.value,) + tuple(child.extract() for child in self.children)

# Arithmetic grammar:
# expr
#   BinOp(binop, expr, expr)
#   Number(number)
# binop
#   +, -
# number
#   0, 1, .., 9
#
# Arithmetic grammar:
# expr
#   Plus(expr, expr)
#   Minus(expr, expr)
#   Number(number)
# binop
#   +, -
# number
#   0, 1, ..., 9


class ArithmeticRecursiveGenerativeModel(object):
  """Provides methods and properties for training and sampling."""

  def __init__(self, emb_size):
    named_tensors = {}
    for n in xrange(10):
      name = 'number_%d' % n
      named_tensors[name] = tf.get_variable(name, [emb_size])

    if FLAGS.root_emb == 'static':
      named_tensors['root_emb'] = tf.get_variable('root_emb', [emb_size])
    elif FLAGS.root_emb in ('random', 'vae'):
      named_tensors['root_emb'] = tf.random_normal([emb_size])

    named_ops = {}
    for binop in ('plus', 'minus'):
      named_ops['split_' + binop] = loom_ops.split('split_' + binop, 2,
                                                   emb_size)
      named_ops['merge_' + binop] = loom_ops.merge('merge_' + binop, 2,
                                                   emb_size)
    named_ops['split_number'] = loom_ops.split('split_number', 1, emb_size)
    named_ops['merge_number'] = loom_ops.merge('merge_number', 1, emb_size)

    named_ops['which_term'] = loom_ops.which('which_term', 3, emb_size)
    named_ops['which_number'] = loom_ops.which('which_number', 10, emb_size)
    named_ops['sample_term'] = loom_ops.MultinomialLoomOp(3, 'sampled_term')
    named_ops['sample_number'] = loom_ops.MultinomialLoomOp(10,
                                                            'sampled_number')

    named_ops['kl_cost'] = loom_ops.KLDivPosteriorPriorLoomOp(emb_size)
    named_ops['sample_z'] = loom_ops.SampleFromGaussianLoomOp(emb_size)

    self.loom = loom.Loom(named_tensors=named_tensors, named_ops=named_ops)

    # Setup the overall TensorFlow graph
    self.term_labels = tf.placeholder(tf.int32, [None])
    self.number_labels = tf.placeholder(tf.int32, [None])
    self.term_outputs = self.loom.output_tensor(
        loom.TypeShape(tf.float32, (3,)))
    self.number_outputs = self.loom.output_tensor(
        loom.TypeShape(tf.float32, (10,)))

    # Used for sampling
    self.sampled_terms = self.loom.output_tensor(
        loom.TypeShape(tf.int64, (), 'sampled_term'))
    self.sampled_numbers = self.loom.output_tensor(
        loom.TypeShape(tf.int64, (), 'sampled_number'))
    self.embs = self.loom.output_tensor(loom.TypeShape(tf.float32, (emb_size,)))

    term_losses = (
        tf.contrib.nn.deprecated_flipped_sparse_softmax_cross_entropy_with_logits(  # pylint: disable=line-too-long
            self.term_outputs, self.term_labels))
    number_losses = (
        tf.contrib.nn.deprecated_flipped_sparse_softmax_cross_entropy_with_logits(  # pylint: disable=line-too-long
            self.number_outputs, self.number_labels))
    self.loss = tf.reduce_mean(tf.concat([term_losses, number_losses], 0))

    if FLAGS.root_emb == 'vae':
      kl_div = self.loom.output_tensor(loom.TypeShape(tf.float32, (), 'kl_div'))
      self.kl_div_mean = tf.reduce_mean(kl_div)
      self.loss += self.kl_div_mean
    else:
      self.kl_div_mean = tf.constant(0.0)

  def build_feed_dict(self, expr_list):
    """Create a feed dict for running a train step with this batch of exprs."""
    loom_input = self.loom.make_weaver()

    for expr in expr_list:
      if FLAGS.root_emb == 'vae':
        # KL divergence between the prior p(z)
        # and the approximate posterior q(z | x)
        kl_cost, z_mean, z_stdev = loom_input.kl_cost(
            self.read_expr_upward(loom_input, expr))
        z_sample = loom_input.sample_z(z_mean, z_stdev)

        self.build_expr_downward(loom_input, expr, z_sample)
        loom_input.add_output(kl_cost)
      else:
        self.build_expr_downward(loom_input, expr,
                                 loom_input.named_tensor('root_emb'))

    feed_dict = loom_input.build_feed_dict()

    expr_list_flat = nest.flatten(expr_list)
    feed_dict[self.term_labels] = [('plus', 'minus', 'number').index(item)
                                   for item in expr_list_flat
                                   if isinstance(item, six.string_types)]
    feed_dict[self.number_labels] = [item for item in expr_list_flat
                                     if isinstance(item, int)]
    return feed_dict

  def build_expr_downward(self, loom_input, expr, embedding):
    """Call SplitLoomOp and WhichLoomOp downward on `expr` recursively.

    Args:
      loom_input: LoomInput object that we'll use for calling LoomOps.
      expr: The expression which we want the model to produce.
      embedding: A LoomResult containing the contextual embedding for the
        generation of `expr`.
    """

    # 'plus', 'minus', 'number', 0-9
    term_type = expr[0]

    # The output of this is put through a softmax to determine
    # the term_type. During training, we force the output of the softmax
    # to match the type reported here.
    if isinstance(term_type, six.string_types):
      # This forms a pre-order traversal of the 'which' decisions
      # we need to make throughout the tree.
      loom_input.add_output(loom_input.which_term(embedding))
      child_embeddings = loom_input.op('split_' + term_type, [embedding])
      for child_expr, child_embedding in zip(expr[1:], child_embeddings):
        self.build_expr_downward(loom_input, child_expr, child_embedding)
    else:
      loom_input.add_output(loom_input.which_number(embedding))

  def read_expr_upward(self, loom_input, expr):
    """Call MergeLoomOp on `expr` recursively.

    Args:
      loom_input: LoomInput object that we'll use for calling LoomOps.
      expr: The expression that we want to obtain an embeding for.

    Returns:
      A LoomResult containing embedding for `expr`.
    """

    # 'plus', 'minus', 'number', 0-9
    term_type = expr[0]

    if isinstance(term_type, six.string_types):
      child_embeddings = [self.read_expr_upward(loom_input, child_expr)
                          for child_expr in expr[1:]]
      return loom_input.op('merge_' + term_type, child_embeddings)[0]
    else:
      return loom_input.named_tensor('number_%d' % term_type)

  def sample_exprs(self, sess, count=1, temperature=1):
    """Sample `count` expressions from the model."""
    arity = {'plus': 2, 'minus': 2, 'number': 1}

    roots = [TreeNode() for _ in xrange(count)]
    layer = [TreeNode(children=[root]) for root in roots]
    next_layer = []
    for _ in xrange(FLAGS.max_sample_depth):
      # Sample the next layer of the tree
      loom_input = self.loom.make_weaver()
      self.sample_next_terms(loom_input, layer, temperature)
      feed_dict = loom_input.build_feed_dict()

      sampled_terms, sampled_numbers, embs = sess.run([self.sampled_terms,
                                                       self.sampled_numbers,
                                                       self.embs],
                                                      feed_dict=feed_dict)

      # Fill the corresponding nodes with the sampled values
      term_index = 0
      number_index = 0
      emb_index = 0

      for node in layer:
        for child_node in node.children:
          if (child_node.parent is not None and
              child_node.parent.value == 'number'):
            child_node.value = sampled_numbers[number_index]
            number_index += 1
          else:
            child_node.value = ('plus', 'minus',
                                'number')[sampled_terms[term_index]]
            child_node.embedding = embs[emb_index]
            child_node.children = [TreeNode(parent=child_node)
                                   for _ in range(arity[child_node.value])]
            term_index += 1
            emb_index += 1
            next_layer += [child_node]

      if not next_layer:
        break
      layer = next_layer
      next_layer = []

    return [root.extract() for root in roots]

  def sample_next_terms(self, loom_input, layer, temperature):
    for node in layer:
      emb = (loom_input.named_tensor('root_emb') if node.embedding is None else
             loom_input.constant(node.embedding))

      simple_type = ('term' if node.value in ('plus', 'minus', None) else
                     'number')
      which_type = 'which_' + simple_type
      sample_type = 'sample_' + simple_type

      if node.value is None:
        # We are handling the dummy node which is treated as
        # the parent of the true root node.
        split_embs = [emb]
      else:
        split_embs = loom_input.op('split_' + node.value, [emb])
      assert len(node.children) == len(split_embs)
      for split_emb in split_embs:
        sampled_type = loom_input.op(
            sample_type, [loom_input.op(which_type, [split_emb])[0],
                          loom_input.constant(np.float32(temperature))])[0]
        # The sampled which_* result.
        loom_input.add_output(sampled_type)
        if simple_type == 'term':
          # Embedding to be used for the next split.
          loom_input.add_output(split_emb)
