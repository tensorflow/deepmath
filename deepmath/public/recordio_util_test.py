"""Tests for deepmath.deephol.public.recordio_util."""

import tensorflow.compat.v1 as tf
from deepmath.public import recordio_util


class RecordioUtilTest(tf.test.TestCase):

  def test_dummy(self):
    self.assertIsNotNone(recordio_util)


if __name__ == '__main__':
  tf.test.main()
