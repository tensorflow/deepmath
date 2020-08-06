"""Tests for deepmath.deephol.public.utils."""

import tensorflow.compat.v1 as tf
from deepmath.deephol.train import utils


class UtilsTest(tf.test.TestCase):

  def test_dummy(self):
    self.assertIsNotNone(utils)


if __name__ == '__main__':
  tf.test.main()
