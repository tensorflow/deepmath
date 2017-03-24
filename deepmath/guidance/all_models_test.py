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
"""Tests for all_models.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from deepmath.guidance import all_models


class AllModelsTest(tf.test.TestCase):

  def testAllModels(self):
    for models in all_models.ALL_MODELS, all_models.PREMISE_MODELS:
      for name, module in models.items():
        self.assertEqual(name, module.__name__.split('.')[-1])
        self.assertEqual(module, all_models.model_module(name))
        hparams = all_models.model_hparams(name)
        self.assertTrue(isinstance(hparams, tf.contrib.training.HParams))
        for mode in 'train', 'eval':
          all_models.make_model(name, mode=mode, hparams=hparams)


if __name__ == '__main__':
  tf.test.main()
