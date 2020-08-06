"""Tests for deepmath.deephol.public.model."""

import tensorflow.compat.v1 as tf
from deepmath.deephol.train import model


class ModelTest(tf.test.TestCase):

  def test_dummy(self):
    self.assertIsNotNone(model)


if __name__ == '__main__':
  tf.test.main()
