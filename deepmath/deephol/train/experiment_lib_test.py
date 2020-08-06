"""Tests for deepmath.deephol.public.experiment_lib."""

import tensorflow.compat.v1 as tf
from deepmath.deephol.train import experiments


class ExperimentLibTest(tf.test.TestCase):

  def test_dummy(self):
    self.assertIsNotNone(experiments)


if __name__ == '__main__':
  tf.test.main()
