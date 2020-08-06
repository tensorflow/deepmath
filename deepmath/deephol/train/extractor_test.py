"""Tests for deepmath.deephol.public.extractor."""

import tensorflow.compat.v1 as tf
from deepmath.deephol.train import extractor


class ExtractorTest(tf.test.TestCase):

  def test_dummy(self):
    self.assertIsNotNone(extractor)


if __name__ == '__main__':
  tf.test.main()
