"""Tests for deepmath.public.build_data."""

import tensorflow.compat.v1 as tf

from deepmath.public import build_data


class BuildDataTest(tf.test.TestCase):

  def test_me(self):
    self.assertIsNotNone(build_data.BuildData())


if __name__ == '__main__':
  tf.test.main()
