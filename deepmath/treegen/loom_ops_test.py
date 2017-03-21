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
"""Tests for treegen.loom_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.misc
import tensorflow as tf
from deepmath.treegen import loom_ops


class LoomOpsTest(tf.test.TestCase):

  def testLogSumExpMatchesScipy(self):
    values = np.array([1000., 1001., 1002.])
    with self.test_session():
      self.assertAllClose(
          loom_ops.logsumexp(values).eval(), scipy.misc.logsumexp(values))


if __name__ == '__main__':
  tf.test.main()
