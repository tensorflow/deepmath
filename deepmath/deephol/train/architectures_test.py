"""Tests for deepmath.deephol.public.architectures."""

import tensorflow.compat.v1 as tf
from deepmath.deephol.train import architectures


class ArchitecturesTest(tf.test.TestCase):

  def test_dummy(self):
    self.assertIsNotNone(architectures)


if __name__ == '__main__':
  tf.test.main()
