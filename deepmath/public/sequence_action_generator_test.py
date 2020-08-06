"""Tests for deepmath.deephol.google.sequence_action_generator."""

import tensorflow.compat.v1 as tf
from deepmath.public import sequence_action_generator


class SequenceActionGeneratorTest(tf.test.TestCase):

  def testNothing(self):
    _ = sequence_action_generator.SequenceActionGenerator


if __name__ == "__main__":
  tf.test.main()
