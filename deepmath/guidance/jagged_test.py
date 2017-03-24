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
"""Tests for jagged."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from deepmath.guidance import jagged


class JaggedTest(tf.test.TestCase):

  def testBasicsAndPack(self):
    np.random.seed(7)
    with self.test_session():
      for _ in range(32):
        sizes = np.random.randint(3, size=1 + np.random.randint(3))
        seqs = [np.random.randn(s, 3) for s in sizes]
        jag = jagged.pack(seqs)
        self.assertEqual(type(jag), jagged.Jagged)
        self.assertAllEqual(jag.offsets.eval(),
                            np.cumsum(np.concatenate([[0], sizes])))
        self.assertAllEqual(jag.flat.eval(), np.concatenate(seqs))
        self.assertEqual(jag.size.eval(), len(seqs))
        for i, seq in enumerate(seqs):
          self.assertAllEqual(jag[i].eval(), seq)

  def testConv(self):
    np.random.seed(7)
    with self.test_session() as sess:
      for _ in range(16):
        sizes = np.random.randint(3, size=1 + np.random.randint(3))
        depth = np.random.randint(3)
        dims = np.random.randint(3, 6, size=1 + depth)
        seqs = jagged.pack([np.random.randn(s, dims[0]).astype(np.float32)
                            for s in sizes])
        widths = np.array([1, 3, 5, 7])[np.random.randint(4, size=depth)]
        filters = [np.random.randn(widths[i], dims[i],
                                   dims[i + 1]).astype(np.float32)
                   for i in range(depth)]
        activations = np.array(
            [None, tf.nn.relu, tf.sigmoid])[np.random.randint(
                3, size=depth)]
        out = jagged.conv1d_stack(seqs, filters, activations)
        self.assertEqual(type(out), jagged.Jagged)
        offsets, flat = sess.run([out.offsets, out.flat])
        for i in range(len(sizes)):
          x = seqs[i][None]
          for filt, activation in zip(filters, activations):
            x = tf.nn.conv1d(x, filt, stride=1, padding='SAME')
            if activation is not None:
              x = activation(x)
          x = tf.squeeze(x, [0])
          self.assertAllClose(flat[offsets[i]:offsets[i + 1]], x.eval())

  def testConvEmpty(self):
    with self.test_session() as sess:
      sizes = tf.constant([], dtype=tf.int32)
      seqs = jagged.jagged(sizes, tf.zeros([0, 7]))
      filters = [tf.zeros([3, 7, 13])]
      activations = [tf.nn.relu]
      out = jagged.conv1d_stack(seqs, filters, activations)
      out_sizes, out_flat = sess.run([out.sizes, out.flat])
      self.assertAllEqual(out_sizes, [])
      self.assertAllEqual(out_flat, np.zeros([0, 13]))

  def testMax(self):
    np.random.seed(7)
    with self.test_session():
      for _ in range(16):
        sizes = 1 + np.random.randint(4, size=1 + np.random.randint(3))
        dim = np.random.randint(3)
        seqs = jagged.pack([np.random.randn(s, dim) for s in sizes])
        maxes = jagged.reduce_max(seqs)
        np_maxes = maxes.eval()
        self.assertEqual(np_maxes.shape, (len(sizes), dim))
        for i in range(len(sizes)):
          self.assertAllEqual(seqs[i].eval().max(axis=0), np_maxes[i])
        # Check gradient
        error = tf.test.compute_gradient_error(seqs.flat, (np.sum(sizes), dim),
                                               maxes, np_maxes.shape)
        self.assertLess(error, 1e-10)

  def testRepeats(self):
    np.random.seed(7)
    with self.test_session():
      for _ in range(16):
        values = np.random.randn(1 + np.random.randint(4))
        times = np.random.randint(4, size=len(values))
        repeated = jagged.repeats(values, times).eval()
        correct = np.concatenate([[v] * t for v, t in zip(values, times)])
        self.assertAllEqual(correct, repeated)


if __name__ == '__main__':
  tf.test.main()
