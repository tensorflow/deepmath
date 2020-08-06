"""Tests for deepmath.deephol.proof_assistant_builder."""

import tensorflow.compat.v1 as tf
from deepmath.deephol import proof_assistant_builder
from deepmath.proof_assistant import proof_assistant_pb2

TRUE_THEOREM = proof_assistant_pb2.Theorem(
    conclusion='(c (bool) T)', tag=proof_assistant_pb2.Theorem.Split.TRAINING)

BAD_THEOREM = proof_assistant_pb2.Theorem(
    conclusion='bad', tag=proof_assistant_pb2.Theorem.Split.TRAINING)


class ProofAssistantBuilderTest(tf.test.TestCase):

  def test_build_smoke(self):
    proof_assistant_builder.build(
        proof_assistant_pb2.TheoremDatabase(theorems=[TRUE_THEOREM]))

  def test_build_error(self):
    with self.assertRaisesRegex(ValueError, 'Registration failed'):
      proof_assistant_builder.build(
          proof_assistant_pb2.TheoremDatabase(theorems=[BAD_THEOREM]))


if __name__ == '__main__':
  tf.test.main()
