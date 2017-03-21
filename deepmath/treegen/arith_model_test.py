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
"""Tests for treegen.arith_model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from deepmath.treegen import arith_make_data
from deepmath.treegen import arith_model

flags = tf.flags
FLAGS = flags.FLAGS


def test_memorization(test_case, embedding_size=32, num_iterations=50):
  with tf.Graph().as_default():
    tf.set_random_seed(1234)
    random.seed(1234)
    np.random.seed(1234)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    m = arith_model.ArithmeticRecursiveGenerativeModel(embedding_size)
    optimizer = tf.train.AdamOptimizer(0.01)
    variables = tf.trainable_variables()
    train_op = optimizer.minimize(
        m.loss, global_step=global_step, var_list=variables)
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())

      for _ in xrange(num_iterations):
        _, = sess.run([train_op], feed_dict=m.build_feed_dict([test_case.expr]))

      test_case.assertEqual(
          test_case.expr, m.sample_exprs(
              sess, temperature=1e-3)[0])


class ArithModelTest(tf.test.TestCase):

  def __init__(self, *args, **kwargs):
    tf.set_random_seed(1234)
    random.seed(1234)
    np.random.seed(1234)
    self.expr = next(
        iter(arith_make_data.generate_trees_with_depth(5, 1000, 0)))

    super(ArithModelTest, self).__init__(*args, **kwargs)

  def testModelMemorizesExprStaticRoot(self):
    FLAGS.root_emb = 'static'
    test_memorization(self, 32, 50)

  def testModelMemorizesExprRandomRoot(self):
    self.expr = next(
        iter(arith_make_data.generate_trees_with_depth(3, 1000, 0)))
    FLAGS.root_emb = 'random'
    test_memorization(self, 32, 150)

  def testModelMemorizesExprVaeRoot(self):
    self.expr = next(
        iter(arith_make_data.generate_trees_with_depth(3, 1000, 0)))
    FLAGS.root_emb = 'vae'
    test_memorization(self, 32, 125)


if __name__ == '__main__':
  tf.test.main()
