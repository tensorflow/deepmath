# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Tests for util.model_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow_fold.public import loom
from deepmath.util import model_utils


class ModelUtilsTest(tf.test.TestCase):

  def testPadUpTo(self):
    np.random.seed(7)
    with self.test_session():
      x = np.random.randn(3, 5)
      for axis in 0, 1:
        for size in range(7):
          y = model_utils.pad_up_to(x, size=size, axis=axis).eval()
          self.assertLessEqual(size, y.shape[axis])
          shape = list(y.shape)
          shape[axis] = x.shape[axis]
          self.assertEqual(x.shape, tuple(shape))
          self.assertAllEqual(x, y[:3, :5])

  def testPadToMultiple(self):
    np.random.seed(7)
    with self.test_session():
      size = 3
      for axis in 0, 1:
        for start in range(7):
          x = np.random.randn(start, start)
          y = model_utils.pad_to_multiple(x, size=size, axis=axis).eval()
          self.assertAllEqual(x, y[:start, :start])
          self.assertEqual(y.shape[1 - axis], start)
          self.assertEqual(y.shape[axis], -start // size * -size)

  def testSharedEmbeddingLookup(self):
    np.random.seed(8)
    size = 100
    dim = 11
    shapes = (6,), (), (7, 3), (3, 2, 3)
    indices = [np.random.randint(size, size=shape) for shape in shapes]
    params = np.random.randn(size, dim).astype(np.float32)
    with self.test_session() as sess:
      results = model_utils.shared_embedding_lookup(params, indices)
      self.assertEqual(len(indices), len(results))
      results = sess.run(results)
      for ids, result in zip(indices, results):
        self.assertAllEqual(result, params[ids])

  def testSharedEmbeddingLayer(self):
    # Test that construction works, but rely on testSharedEmbeddingLookup to
    # test actual functionality.
    size = 100
    dim = 11
    shapes = (6,), (), (7, 3), (3, 2, 3)
    indices = [tf.random_uniform(shape, maxval=size, dtype=tf.int32)
               for shape in shapes]
    for shards in 1, 7:
      model_utils.shared_embedding_layer(indices, dim=dim, size=size,
                                         shards=shards, name='emb%d' % shards)

  def _asLoomOpTest(self, max_depth):
    with tf.Graph().as_default() as graph:
      tf.set_random_seed(8)
      np.random.seed(7)
      ts = loom.TypeShape(tf.int64, ())
      initializer = tf.random_uniform_initializer(dtype=tf.int64, minval=0,
                                                  maxval=1 << 60)

      @model_utils.as_loom_op([ts, ts], ts)
      def f(x, y):
        # Use a variable to make sure variable sharing works correctly
        rand = tf.get_variable('rand', shape=(), dtype=tf.int64,
                               initializer=initializer)
        return rand - x - y

      @model_utils.as_loom_op([ts, ts], [ts, ts])
      def g(x, y):
        # Test multiple outputs
        return x - y, x + y

      def make_h():
        # Ensure that we can reuse names for different calls to as_loom_op.
        # Also test that the name argument to as_loom_op works.
        @model_utils.as_loom_op([ts], ts, name='h')
        def not_h(x):
          v = tf.get_variable('yo', shape=(), dtype=tf.int64,
                              initializer=initializer)
          return x + v
        return not_h

      # Make two h's to ensure they make separate variables
      h1 = make_h()
      h2 = make_h()

      simple_loom = loom.Loom(named_ops={'f': f, 'g': g, 'h1': h1, 'h2': h2},
                              max_depth=max_depth)
      self.assertEqual(['f/rand', 'h/yo', 'h_1/yo'],
                       [v.op.name for v in tf.global_variables()])

      # Use f twice and (g,h1,h2) once each
      weaver = simple_loom.make_weaver()
      x, y, z = np.random.randint(1 << 60, size=3)
      wx, wy, wz = weaver(x), weaver(y), weaver(z)
      weaver.add_output(weaver.f(wx, weaver.f(wy, wz)))
      plus, minus = weaver.g(wx, wy)
      weaver.add_output(plus)
      weaver.add_output(minus)
      weaver.add_output(weaver.h1(wx))
      weaver.add_output(weaver.h2(wx))

      with self.test_session(graph=graph):
        tf.global_variables_initializer().run()
        out = simple_loom.output_tensor(ts).eval(weaver.build_feed_dict())
        self.assertEqual(out.shape, (5,))
        # out[0] works only if variables are shared between layers:
        #   rand - x - (rand - y - z) = y + z - x
        self.assertEqual(out[0], y + z - x)
        # out[1] and out[2] are simple
        self.assertEqual(out[1], x - y)
        self.assertEqual(out[2], x + y)
        # out[3] and out[4] should use different random variables
        self.assertNotEqual(out[3], out[4])

  def testAsLoomOpWhile(self):
    self._asLoomOpTest(max_depth=None)

  def testAsLoomOpUnrolled(self):
    self._asLoomOpTest(max_depth=2)

  def testAsLoomOpNames(self):
    # Test that as_loom_op makes scopes unique in the order the decorators are
    # called, not the order TensorLoom calls instantiate batch in.
    n = 29
    ts = loom.TypeShape(tf.int64, ())

    def make_op(i):

      @model_utils.as_loom_op([ts], ts)
      def name(x):
        var = tf.get_variable(
            'var%d' % i,
            shape=(),
            dtype=tf.int64,
            initializer=tf.constant_initializer(i))
        return x + var

      return name

    # Make a loom with a bunch of ops, with names unrelated to the order
    ops = {'op%d' % (i * 11 % n): make_op(i) for i in range(n)}
    loom.Loom(named_ops=ops)

    # Check that the variables were created in the right order
    self.assertItemsEqual(
        [('name_%d/var%d' % (i, i)).replace('_0', '') for i in range(n)],
        [v.op.name for v in tf.global_variables()])

  def testPairedJoint(self):
    n = 7
    np.random.seed(n)
    positives = np.random.randn(n)
    negatives = np.random.randn(n)
    logits = np.array([positives, negatives]).T.ravel()
    labels = np.array([True, False] * n)
    with self.test_session() as sess:
      joint_logits, joint_labels = sess.run(
          model_utils.paired_joint_logits_and_labels(logits, labels))
      self.assertEqual(joint_logits.shape, joint_labels.shape)
      self.assertAllClose(joint_logits, positives - negatives)
      self.assertAllEqual(joint_labels, [True] * n)

  def testMergeHParams(self):
    hp0 = tf.contrib.training.HParams(x=7)
    hp1 = tf.contrib.training.HParams(y=True)
    hp = model_utils.merge_hparams(hp0, hp1)
    self.assertEqual(hp.values(), {'x': 7, 'y': True})

  def testMergeHParamsConflict(self):
    hp = tf.contrib.training.HParams(x=7)
    with self.assertRaisesRegexp(
        ValueError, r"Hyperparameter 'x' occurs twice with values 7 and 7"):
      model_utils.merge_hparams(hp, hp)


if __name__ == '__main__':
  tf.test.main()
