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
"""Tests for train."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import numpy as np
import tensorflow as tf
from deepmath.guidance import train
from deepmath.util import test_utils

flags = tf.flags
FLAGS = flags.FLAGS


def matvec(a, b):
  return tf.squeeze(tf.matmul(a, tf.expand_dims(b, -1)), [-1])


class TrainTest(tf.test.TestCase):

  def train_test(self, make_model, hparam_str, joint_safe=False):
    hparams = train.default_hparams()
    hparams.parse(hparam_str)
    hparams.eval_batch_size = 1
    hparams.eval_examples = 4
    hparams.max_evals = 1

    with test_utils.temp_dir() as tmp:
      FLAGS.logdir = tmp

      # Train
      with tf.Graph().as_default():
        tf.set_random_seed(8)
        train.sigmoid_train(make_model, hparams, joint_safe=joint_safe)

      # Eval
      with tf.Graph().as_default():
        tf.set_random_seed(8)
        train.sigmoid_eval(make_model, hparams, joint_safe=joint_safe)

      # Check eval
      event_path, = glob.glob(os.path.join(tmp, 'eval/events*'))
      tag = 'eval %saccuracy' % ('joint ' * (hparams.loss == 'joint'))
      accuracy, = [value.simple_value
                   for event in tf.train.summary_iterator(event_path)
                   for value in event.summary.value
                   if value.tag == tag]
      print('accuracy = %g' % accuracy)
      self.assertLess(0.9, accuracy)
      self.assertLess(accuracy, 0.99)

  def testTrain(self):
    np.random.seed(7)
    secret = np.random.randn(10).astype(np.float32)

    def make_model():
      c = tf.get_variable('c', shape=secret.shape,
                          initializer=tf.random_normal_initializer())
      x = tf.random_normal([128, len(secret)])
      logits = matvec(x, c)
      labels = matvec(x, secret) > 0
      return logits, labels

    self.train_test(make_model,
                    'learning_rate=0.1,max_steps=30,use_averages=false')

  def testTrainJoint(self):
    np.random.seed(7)
    secret = np.random.randn(10).astype(np.float32)

    def make_model():
      c = tf.get_variable('c', shape=secret.shape,
                          initializer=tf.random_normal_initializer())
      x = tf.random_normal([128, len(secret)])
      truth = matvec(x, secret)
      scale = 2 * tf.cast(truth[::2] > truth[1::2], tf.float32) - 1
      x *= tf.reshape(tf.transpose(tf.stack([scale, scale])), [-1, 1])
      logits = matvec(x, c)
      labels = tf.constant([True, False] * 64)
      return logits, labels

    self.train_test(make_model,
                    'learning_rate=0.3,max_steps=30,use_averages=false,'
                    'loss=joint', joint_safe=True)

  def testTrainAverage(self):
    # TODO(geoffreyi): Enable once Polyak averaging works
    if 0:  # pylint: disable=using-constant-test
      self.train_test(None, 'use_averages=true')


if __name__ == '__main__':
  tf.test.main()
