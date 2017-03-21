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
"""Generates training data for arith_train.py.

Expressions satisfy depth or length constraints and evaluate to a target value.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
import json
import random
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from deepmath.treegen import arith_utils
from deepmath.treegen import binary_trees

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('depth', None,
                     'Maximum depth (i.e. nesting) of expression.')
flags.DEFINE_integer('terminals', None,
                     'Maximum number of terminals in expression.')
flags.DEFINE_integer('count', 10000, 'Number of examples to generate.')
flags.DEFINE_integer('num_subtrees', 10000, 'Number of subtrees to generate.')
flags.DEFINE_integer('target', 0, 'Value that expressions should evaluate to.')

flags.DEFINE_string('out', 'expressions.jsonl', 'Output file.')
flags.DEFINE_boolean('print_output', False,
                     'Print out the generated expressions.')
flags.DEFINE_boolean('cnf_format', False,
                     'Use CNF format and write to a .jsonl file.')


def fill_tree_with_arithmetic(node):
  if node.left is None and node.right is None:
    node.value = ('number', (random.randint(0, 9),))
    return

  node.value = random.choice(['plus', 'minus'])
  fill_tree_with_arithmetic(node.left)
  fill_tree_with_arithmetic(node.right)
  return


def convert_binary_tree_to_tuple(node):
  result = node.value if isinstance(node.value, (list, tuple)) else [node.value]
  if node.left is not None:
    result.append(convert_binary_tree_to_tuple(node.left))
  if node.right is not None:
    result.append(convert_binary_tree_to_tuple(node.right))
  return tuple(result)


def flatten_dict_of_lists(dic):
  return [(key, v) for key, ls in dic.items() for v in ls]


def convert_tuple_to_cnf(expr):
  if expr[0] != 'number':
    return {'func': expr[0],
            'params': [convert_tuple_to_cnf(arg) for arg in expr[1:]]}
  else:
    return {'number': str(expr[1][0])}


def generate_trees_with_num_terminals(num_terminals, num_subtrees, target):
  """Generate trees with `num_terminals` terms which evaluate to `target`."""
  if num_terminals == 1:
    while True:
      yield ['number', [target]]
    return

  all_trees = collections.defaultdict(list)

  if num_terminals == 2:
    for i in xrange(10):
      all_trees[(1, i)].append(['number', [i]])
  else:
    trees_by_num_nodes = np.array([binary_trees.count_trees_with_num_nodes(n)
                                   for n in xrange(num_terminals - 1)],
                                  dtype=float)
    for i in xrange(len(trees_by_num_nodes) // 2):
      trees_by_num_nodes[i] = trees_by_num_nodes[-(i + 1)] = max(
          trees_by_num_nodes[i], trees_by_num_nodes[-(i + 1)])
    trees_by_num_nodes *= num_subtrees / np.sum(trees_by_num_nodes)

    trees_by_num_nodes = np.rint(trees_by_num_nodes).astype(int)
    trees_by_num_nodes[-1] += num_subtrees - sum(trees_by_num_nodes)

    for _, n in zip(
        range(num_subtrees), itertools.chain.from_iterable(
            itertools.repeat(num_nodes, num_nodes_count)
            for num_nodes, num_nodes_count in enumerate(trees_by_num_nodes))):
      tree = binary_trees.random_tree_with_num_nodes(n)
      fill_tree_with_arithmetic(tree)

      expr = convert_binary_tree_to_tuple(tree)
      value = arith_utils.eval_expr(expr)

      all_trees[(n + 1, value)].append(expr)

  all_values = set(all_trees.keys())
  plus_values = set(v for v in all_values
                    if (num_terminals - v[0], target - v[1]) in all_values)
  minus_values = set(v for v in all_values
                     if (num_terminals - v[0], v[1] - target) in all_values)

  plus_trees = {k: all_trees[k] for k in plus_values}
  minus_trees = {k: all_trees[k] for k in minus_values}
  plus_trees_list = flatten_dict_of_lists(plus_trees)
  minus_trees_list = flatten_dict_of_lists(minus_trees)
  plus_trees_count = len(plus_trees_list)
  minus_trees_count = len(minus_trees_list)

  plus_prob = plus_trees_count / (plus_trees_count + minus_trees_count)

  while True:
    op = np.random.choice(['plus', 'minus'], p=[plus_prob, 1 - plus_prob])
    if op == 'plus':
      right_target = lambda t, v: (num_terminals - t, target - v)
      trees_list = plus_trees_list
    else:
      right_target = lambda t, v: (num_terminals - t, v - target)
      trees_list = minus_trees_list

    (left_num_terminals, left_value), left_expr = random.choice(trees_list)
    right_expr = random.choice(all_trees[right_target(left_num_terminals,
                                                      left_value)])
    yield (op, left_expr, right_expr)


def generate_trees_with_depth(depth, num_subtrees, target):
  """Generate trees with `depth` which evaluate to `target`."""
  if depth == 1:
    while True:
      yield ['number', [target]]
    return

  # Short trees: trees with depth less than `depth`
  # Tall trees: trees with exactly `depth` depth
  short_trees_by_value = collections.defaultdict(list)
  tall_trees_by_value = collections.defaultdict(list)

  if depth == 2:
    for i in xrange(10):
      tall_trees_by_value[i].append(['number', [i]])
  else:
    trees_per_depth = np.array([binary_trees.count_tree_configs(1, d)
                                for d in xrange(depth - 1)])
    trees_per_depth *= num_subtrees / np.sum(trees_per_depth)
    trees_per_depth = np.rint(trees_per_depth).astype(int)
    trees_per_depth[-1] += num_subtrees - sum(trees_per_depth)

    for i, d in zip(
        range(num_subtrees), itertools.chain.from_iterable(
            itertools.repeat(depth + 1, depth_count)
            for depth, depth_count in enumerate(trees_per_depth))):

      tree = binary_trees.random_tree_of_depth(d)
      fill_tree_with_arithmetic(tree)

      expr = convert_binary_tree_to_tuple(tree)
      value = arith_utils.eval_expr(expr)

      if d < depth - 1:
        short_trees_by_value[value].append(expr)
      else:
        tall_trees_by_value[value].append(expr)

  all_values = set(short_trees_by_value) | set(tall_trees_by_value)

  # The values that can go on the left-hand side of a + or -, because
  # at least one tree exists for the value that would be needed on the
  # right-hand side.
  plus_values = set(v for v in all_values if target - v in all_values)
  minus_values = set(v for v in all_values if v - target in all_values)

  short_trees_plus = {k: short_trees_by_value[k] for k in plus_values}
  short_trees_minus = {k: short_trees_by_value[k] for k in minus_values}
  tall_trees_plus = {k: tall_trees_by_value[k] for k in plus_values}
  tall_trees_minus = {k: tall_trees_by_value[k] for k in minus_values}
  short_trees_plus_list = flatten_dict_of_lists(short_trees_plus)
  short_trees_minus_list = flatten_dict_of_lists(short_trees_minus)
  tall_trees_plus_list = flatten_dict_of_lists(tall_trees_plus)
  tall_trees_minus_list = flatten_dict_of_lists(tall_trees_minus)

  short_trees_plus_count = sum(len(v) for v in short_trees_plus.values())
  short_trees_minus_count = sum(len(v) for v in short_trees_minus.values())
  tall_trees_plus_count = sum(len(v) for v in tall_trees_plus.values())
  tall_trees_minus_count = sum(len(v) for v in tall_trees_minus.values())
  short_plus_prob = short_trees_plus_count / (short_trees_plus_count +
                                              tall_trees_plus_count)
  short_minus_prob = short_trees_minus_count / (short_trees_minus_count +
                                                tall_trees_minus_count)
  plus_prob = (short_trees_plus_count + tall_trees_plus_count
              ) / (short_trees_plus_count + tall_trees_plus_count +
                   short_trees_minus_count + tall_trees_minus_count)

  while True:
    op = np.random.choice(['plus', 'minus'], p=[plus_prob, 1 - plus_prob])
    if op == 'plus':
      right_target = lambda left: target - left
      short_prob = short_plus_prob
      short_list = short_trees_plus_list
      tall_list = tall_trees_plus_list
    else:
      right_target = lambda left: left - target
      short_prob = short_minus_prob
      short_list = short_trees_minus_list
      tall_list = tall_trees_minus_list

    if random.random() < short_prob:
      left_value, left_expr = random.choice(short_list)
      right_source = tall_trees_by_value[right_target(left_value)]
    else:
      left_value, left_expr = random.choice(tall_list)
      right_source = (short_trees_by_value[right_target(left_value)] +
                      tall_trees_by_value[right_target(left_value)])
    right_expr = random.choice(right_source)

    yield (op, left_expr, right_expr)


def main(unused_argv):
  out = open(FLAGS.out, 'w')

  if FLAGS.depth is not None:
    generator = generate_trees_with_depth(FLAGS.depth, FLAGS.num_subtrees,
                                          FLAGS.target)
  elif FLAGS.terminals is not None:
    generator = generate_trees_with_num_terminals(FLAGS.terminals,
                                                  FLAGS.num_subtrees,
                                                  FLAGS.target)
  else:
    tf.logging.fatal('Need to specify one of --depth or --terminals')
  generator = iter(generator)

  for _ in range(FLAGS.count):
    expr = next(generator)
    if FLAGS.print_output:
      print(expr)
      print(arith_utils.stringify_expr(expr))
    if FLAGS.cnf_format:
      generated = convert_tuple_to_cnf(expr)
      generated['positive'] = True
      out.write(json.dumps({'clauses': [generated]}))
      out.write('\n')
    else:
      out.write(json.dumps(expr))
      out.write('\n')


if __name__ == '__main__':
  tf.app.run()
