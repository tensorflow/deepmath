"""Tests for deepmath.deephol.public.data."""

import tensorflow.compat.v1 as tf
from deepmath.deephol.train import data


class DataTest(tf.test.TestCase):

  def test_dummy(self):
    self.assertIsNotNone(data)


if __name__ == '__main__':
  tf.test.main()
