"""Tests for deepmath.deephol.public.proof_assistant."""

import tensorflow.compat.v1 as tf
from deepmath.public import proof_assistant


class ProofAssistantTest(tf.test.TestCase):

  def test_dummy(self):
    self.assertIsNotNone(proof_assistant)


if __name__ == '__main__':
  tf.test.main()
