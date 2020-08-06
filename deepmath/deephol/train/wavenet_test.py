"""Tests for deepmath.deephol.public.wavenet."""

import tensorflow.compat.v1 as tf
from deepmath.deephol.train import wavenet


class WavenetTest(tf.test.TestCase):

  def test_dummy(self):
    self.assertIsNotNone(wavenet)


if __name__ == '__main__':
  tf.test.main()
