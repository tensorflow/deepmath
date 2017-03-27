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
"""Shared functions for cnf_model_test and cnf_model_big_test."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import json
import os
import pprint
import random
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from deepmath.treegen import cnf_model

flags = tf.flags
FLAGS = flags.FLAGS


def test_memorization(test_case,
                      expr,
                      num_iterations=75,
                      model_class=cnf_model.CNFTreeModel,
                      extra_hparams=''):
  """Run model on expression(s) and see if it works."""
  with tf.gfile.GFile(
      'deepmath/treegen/testdata/cnf_metadata.json') as f:
    clause_metadata = json.load(f)

  if not isinstance(expr, (list, tuple)):
    expr = [expr]

  with tf.Graph().as_default():
    random.seed(1234)
    np.random.seed(1234)
    tf.set_random_seed(1234)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    hparams = model_class.default_hparams()
    hparams.batch_size = len(expr)
    hparams.parse(extra_hparams)

    m = model_class(itertools.cycle(expr), hparams, clause_metadata)

    optimizer = tf.train.AdamOptimizer(0.01)
    variables = tf.trainable_variables()
    train_op = optimizer.minimize(
        m.loss, global_step=global_step, var_list=variables)
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())

      for _ in xrange(num_iterations):
        _, = sess.run([train_op])

      if num_iterations > 1:
        sampled = m.sample(sess, temperature=1e-6)
        test_case.assertEqual(expr[0], sampled, 'Original:\n%s\nSampled:\n%s' %
                              (pprint.pformat(expr), pprint.pformat(sampled)))
