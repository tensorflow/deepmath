"""Tests for deepmath.public.error."""

import tensorflow.compat.v1 as tf
from deepmath.public import error


class ErrorTest(tf.test.TestCase):

  def test_error(self):
    self.assertEqual('error', error.StatusNotOk('error').message)


if __name__ == '__main__':
  tf.test.main()
