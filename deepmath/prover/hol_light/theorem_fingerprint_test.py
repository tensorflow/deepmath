"""Tests for deepmath.prover.hol_light.theorem_fingerprint."""

import tensorflow as tf
from deepmath.prover.hol_light import hol_light_pb2
from deepmath.prover.hol_light import theorem_fingerprint


class TheoremFingerprintTest(tf.test.TestCase):

  def testStableFingerprint(self):
    """Tests that the theorem fingerprint function is stable."""
    theorem = hol_light_pb2.Theorem(
        conclusion="concl", hypotheses=["hyp1", "hyp2"])
    self.assertEqual(
        theorem_fingerprint.Fingerprint(theorem), 198703484454304307)
    self.assertEqual(
        theorem_fingerprint.ToTacticArgument(theorem), "THM 198703484454304307")


if __name__ == "__main__":
  tf.test.main()
