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
"""Tests for treegen.arith_make_data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
import tensorflow as tf
from deepmath.treegen import arith_make_data
from deepmath.treegen import arith_utils


class ArithMakeDataTest(tf.test.TestCase):

  def testGenerateExpr(self):
    np.random.seed(1234)
    random.seed(1234)

    for target, size in zip(range(2, 6), range(2, 6)):
      expr = next(iter(arith_make_data.generate_trees_with_num_terminals(
          size, 1000, target)))
      self.assertEqual(arith_utils.eval_expr(expr), target)

      expr = next(iter(arith_make_data.generate_trees_with_depth(size, 1000,
                                                                 target)))
      self.assertEqual(arith_utils.eval_expr(expr), target)


if __name__ == '__main__':
  tf.test.main()
