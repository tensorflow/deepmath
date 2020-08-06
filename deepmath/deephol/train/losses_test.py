"""Tests for deepmath.deephol.public.losses."""

import tensorflow.compat.v1 as tf
from deepmath.deephol.train import losses


class LossesTest(tf.test.TestCase):

  def test_dummy(self):
    self.assertIsNotNone(losses)


if __name__ == '__main__':
  tf.test.main()
