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
"""Tests for clause_search.clause_loom."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from tensorflow_fold.public import loom
from deepmath.eprover import prover_clause_examples_pb2
from deepmath.guidance import clause_loom
from deepmath.guidance import inputs
from deepmath.util import model_utils

FLAGS = tf.flags.FLAGS


# TypeShapes
VOCAB_ID = clause_loom.VOCAB_ID
DEPTH = loom.TypeShape(tf.int32, (), 'value')
COMBINATION = loom.TypeShape(tf.string, (2,), 'combination')


def embeddings(layers):
  """One scalar string TypeShape per layer."""
  assert layers in (1, 2)
  return [loom.TypeShape(tf.string, (), 'embedding%d' % i)
          for i in range(layers)]


def assert_same(x, y):
  check = tf.Assert(tf.reduce_all(tf.equal(x, y)), [x, y])
  with tf.control_dependencies([check]):
    return tf.identity(x)


def reverse(strings):
  """Reverse a tensor of strings as a pyfunc."""

  def reverse_py(strings):
    results = [s[::-1] for s in strings.flat]
    return np.array(results, dtype=object).reshape(strings.shape)

  rev, = tf.py_func(reverse_py, (strings,), (tf.string,))
  return rev


def apply_op(tag, layers):
  """Add one function argument to a function application.

  Args:
    tag: Used to distinguish conjecture and clause loom ops.
    layers: Either 1 or 2.  If 2, the second layer is the reverse
        of the first.

  Returns:
    A loom op.
  """
  def apply_batch(fs, args):
    results = []
    for f, arg in zip(fs, args):
      if b'(' in f:
        results.append(tag + f[:-1] + b',' + arg + b')')
      else:
        results.append(tag + f + b'(' + arg + b')')
    return np.array(results, dtype=object)

  @model_utils.as_loom_op(embeddings(layers) * 2, embeddings(layers))
  def apply_loom(*args):
    if layers == 1:
      f, arg = args
      return tf.py_func(apply_batch, (f, arg), (tf.string,))
    elif layers == 2:
      f0, f1, arg0, arg1 = args
      out0, = tf.py_func(apply_batch, (f0, arg0), (tf.string,))
      rev_out1, = tf.py_func(apply_batch, (reverse(f1), reverse(arg1)),
                             (tf.string,))
      out1 = reverse(assert_same(out0, rev_out1))
      return out0, out1

  return apply_loom


def not_op(tag, layers):
  """Negate an expression, using a tag to distinguish conjecture and clause."""
  op = tag + b'~'

  @model_utils.as_loom_op(embeddings(layers), embeddings(layers))
  def not_loom(*args):
    if layers == 1:
      x, = args
      return op + x,
    elif layers == 2:
      x0, x1 = args
      return op + x0, x1 + op[::-1]

  return not_loom


def binary_op(tag, op, layers):
  """Apply a binary op, using a tag to distinguish clause and conjecture."""
  assert layers in (1, 2)
  op = tag + op

  @model_utils.as_loom_op(embeddings(layers) * 2, embeddings(layers))
  def binary_loom(*args):
    if layers == 1:
      x, y = args
      return x + op + y,
    elif layers == 2:
      x0, x1, y0, y1 = args
      return x0 + op + y0, y1 + op[::-1] + x1

  return binary_loom


def random_list(limit, empty, separator, f):
  count = np.random.randint(limit)
  if not count:
    return empty
  return separator.join(f() for _ in range(count))


def random_name(prefix):
  return tf.compat.as_bytes('%s%d' % (prefix, np.random.randint(10)))


def random_term(term, depth):
  if depth == 0 or np.random.randint(3) == 0:
    if np.random.randint(2):
      name = term.variable.name = random_name('X')
      return name
    else:
      name = term.number.value = random_name('')
      return name
  else:
    name = term.function.name = random_name('f')

    def random_arg():
      return random_term(term.function.args.add(), depth=depth - 1)

    args = random_list(2, b'', b',', random_arg)
    return name + b'(' + args + b')' if args else name


def random_equation(equation):
  equation.negated = np.random.randint(2)
  s = random_term(equation.left, depth=2)
  if np.random.randint(2):
    s = b'=(' + s + b',' + random_term(equation.right, depth=1) + b')'
  return b'~' * equation.negated + s


def random_clause(clause):
  return random_list(4, b'$false', b'|',
                     lambda: random_equation(clause.clause.equations.add()))


def random_clauses(clauses):
  return random_list(4, b'$true', b'&', lambda: random_clause(clauses.add()))


class ClauseLoomTest(tf.test.TestCase):

  def _loomTest(self, shuffle, layers):
    # This test builds a loom that reconstructs the string representation of the
    # input.  Thus, all the "embeddings" are scalar strings, and the ops do
    # various kinds of string concatenation.  We then check that the
    # reconstructed representations match the ProverClauseExamples protos that
    # we constructed.
    np.random.seed(7)

    # Build vocabulary
    vocab_path = os.path.join(self.get_temp_dir(), 'vocab')
    with open(vocab_path, 'w') as vocab_file:
      for kind in 'f', 'X', '':
        for i in range(10):
          print('%s%d' % (kind, i), file=vocab_file)
    vocab_size, vocab_to_id = inputs.read_vocab(vocab_path)
    vocab = [''] * vocab_size
    for s, i in vocab_to_id.items():
      vocab[i] = s

    # We tag conjecture and clause ops to ensure the correct ones are used
    conjecture_tag = b'A'
    clause_tag = b'B'

    def order(s):
      if shuffle:
        return b'&'.join(sorted(b'|'.join(sorted(c.split(b'|')))
                                for c in s.split(b'&')))
      return s

    def clean_conjecture(s):
      return order(s.replace(conjecture_tag, b''))

    def clean_clause(s):
      return order(s.replace(clause_tag, b''))

    @model_utils.as_loom_op([VOCAB_ID], embeddings(layers))
    def embed(ids):
      """Turn a vocab_id back into the string it represents."""
      e0 = tf.gather(vocab, ids)
      if layers == 1:
        return e0,
      elif layers == 2:
        return e0, reverse(e0)

    @model_utils.as_loom_op(embeddings(layers) * 2, COMBINATION)
    def combine(*args):
      """Combine conjecture, clause, and label."""
      if layers == 1:
        x, y = args
        return tf.stack([x, y], axis=-1)
      elif layers == 2:
        x0, x1, y0, y1 = args
        c0 = tf.stack([x0, y0], axis=-1)
        c1 = tf.stack([x1, y1], axis=-1)
        return assert_same(c0, reverse(c1))

    # Make a loom that reconstructs the string representation of the input
    placeholder = tf.placeholder(tf.string)
    pairs, labels = clause_loom.weave_clauses(
        examples=placeholder,
        vocab=vocab_path,
        shuffle=shuffle,
        embed=embed,
        conjecture_apply=apply_op(
            conjecture_tag, layers=layers),
        conjecture_not=not_op(
            conjecture_tag, layers=layers),
        conjecture_or=binary_op(
            conjecture_tag, b'|', layers=layers),
        conjecture_and=binary_op(
            conjecture_tag, b'&', layers=layers),
        clause_apply=apply_op(
            clause_tag, layers=layers),
        clause_not=not_op(
            clause_tag, layers=layers),
        clause_or=binary_op(
            clause_tag, b'|', layers=layers),
        combine=combine)
    self.assertEqual(pairs.dtype, tf.string)
    self.assertEqual(labels.dtype, tf.bool)

    # Test it out
    with self.test_session() as sess:
      for batch_size in 0, 1, 3, 5:
        all_examples = []
        conjectures = []
        clauses = []
        for _ in range(batch_size):
          examples = prover_clause_examples_pb2.ProverClauseExamples()
          conjectures.extend(
              [order(random_clauses(examples.cnf.negated_conjecture))] * 2)
          clauses.extend([order(random_clause(examples.positives.add())),
                          order(random_clause(examples.negatives.add()))])
          all_examples.append(examples.SerializeToString())
        pairs_np, labels_np = sess.run([pairs, labels],
                                       feed_dict={placeholder: all_examples})
        self.assertEqual(pairs_np.shape, (2 * batch_size, 2))
        self.assertEqual(labels_np.shape, (2 * batch_size,))
        self.assertEqual(conjectures, [clean_conjecture(c)
                                       for c in pairs_np[:, 0]])
        self.assertEqual(clauses, [clean_clause(c) for c in pairs_np[:, 1]])
        self.assertEqual([True, False] * batch_size, list(labels_np))

  def testLoomOrdered(self):
    self._loomTest(shuffle=False, layers=1)

  def testLoomOrderedDeep(self):
    self._loomTest(shuffle=False, layers=2)

  def testLoomShuffled(self):
    self._loomTest(shuffle=True, layers=1)

  def testLoomShuffledDeep(self):
    self._loomTest(shuffle=True, layers=2)


if __name__ == '__main__':
  tf.test.main()
