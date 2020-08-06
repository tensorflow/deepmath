"""Tests for deepmath.prover.hol_light.theorem_fingerprint."""

import tensorflow.compat.v1 as tf
from deepmath.deephol import theorem_fingerprint
from deepmath.proof_assistant import proof_assistant_pb2


class TheoremFingerprintTest(tf.test.TestCase):

  def setUp(self):
    tf.test.TestCase.setUp(self)
    self.true_entails_true = proof_assistant_pb2.Theorem(
        conclusion="(c (bool) T)",
        hypotheses=["(c (bool) T)"],
        tag=proof_assistant_pb2.Theorem.THEOREM)

  def testStableFingerprint(self):
    """Tests that the theorem fingerprint function is stable."""
    theorem = proof_assistant_pb2.Theorem(
        conclusion="concl", hypotheses=["hyp1", "hyp2"])
    self.assertEqual(
        theorem_fingerprint.Fingerprint(theorem), 198703484454304307)
    self.assertEqual(
        theorem_fingerprint.ToTacticArgument(theorem), "THM 198703484454304307")

  def testStableFingerprintWithAssumptions(self):
    """Fingerprint does distinguish between assumptions and hypotheses.

    This is somewhat unreliable though: The differentiation only comes from the
    fact that the first bits of the fingerprint are shaved off.
    """
    theorem = proof_assistant_pb2.Theorem(
        conclusion="concl",
        tag=proof_assistant_pb2.Theorem.GOAL,
        assumptions=[
            proof_assistant_pb2.Theorem(
                conclusion="hyp1", tag=proof_assistant_pb2.Theorem.THEOREM),
            proof_assistant_pb2.Theorem(
                conclusion="hyp2", tag=proof_assistant_pb2.Theorem.DEFINITION)
        ])
    self.assertEqual(
        theorem_fingerprint.Fingerprint(theorem), 73736987001343503)
    self.assertEqual(
        theorem_fingerprint.ToTacticArgument(theorem), "THM 73736987001343503")

  def testSimpleTermFingerprint(self):
    simple_term = proof_assistant_pb2.Theorem(
        conclusion="(c (bool) T)", tag=proof_assistant_pb2.Theorem.THEOREM)
    self.assertEqual(
        theorem_fingerprint.Fingerprint(simple_term), 70761607289060832)

  def testTheoremFingerprintWithHypothesis(self):
    self.assertEqual(
        theorem_fingerprint.Fingerprint(self.true_entails_true),
        4420649969775231556)

  def testTheoremFingerprintWithTwoHypotheses(self):
    multi_hypotheses_thm = proof_assistant_pb2.Theorem(
        conclusion="(c (bool) T)",
        hypotheses=["(c (bool) F)", "(c (bool) T)"],
        tag=proof_assistant_pb2.Theorem.THEOREM)
    self.assertEqual(
        theorem_fingerprint.Fingerprint(multi_hypotheses_thm),
        2661800747689726299)

  def testGoalFingerprintWithAssumption(self):
    theorem = proof_assistant_pb2.Theorem(
        conclusion="(c (bool) T)",
        tag=proof_assistant_pb2.Theorem.GOAL,
        assumptions=[self.true_entails_true])
    self.assertEqual(
        theorem_fingerprint.Fingerprint(theorem), 4196583489440000546)

  def testEmptyTaskFingerprint(self):
    empty_task = proof_assistant_pb2.ProverTask()
    self.assertEqual(
        theorem_fingerprint.TaskFingerprint(empty_task), 3652755499133195071)

  def testTaskFingerprintWithGoal(self):
    goal = proof_assistant_pb2.Theorem(conclusion="(c (bool) T)")
    task = proof_assistant_pb2.ProverTask(goals=[goal])
    self.assertEqual(
        theorem_fingerprint.TaskFingerprint(task), 3206729314916035051)
    self.assertNotEqual(
        theorem_fingerprint.TaskFingerprint(task),
        theorem_fingerprint.Fingerprint(goal))

  def testTaskFingerprintWithTarget(self):
    goal = proof_assistant_pb2.Theorem(conclusion="(c (bool) T)")
    task = proof_assistant_pb2.ProverTask(targets=[goal])
    self.assertEqual(
        theorem_fingerprint.TaskFingerprint(task), 4078458407704493490)

  def testTaskFingerprintWithGoalAndTarget(self):
    goal = proof_assistant_pb2.Theorem(conclusion="(c (bool) T)")
    task = proof_assistant_pb2.ProverTask(goals=[goal], targets=[goal])
    self.assertEqual(
        theorem_fingerprint.TaskFingerprint(task), 3390660262360462238)

  def testTaskFingerprintWithGoalAndPremiseSet(self):
    goal = proof_assistant_pb2.Theorem(conclusion="(c (bool) T)")
    section = proof_assistant_pb2.DatabaseSection(
        database_name="test_name", before_premise=42)
    premise_set = proof_assistant_pb2.PremiseSet(sections=[section])
    task = proof_assistant_pb2.ProverTask(goals=[goal], premise_set=premise_set)
    self.assertEqual(
        theorem_fingerprint.TaskFingerprint(task), 175258283192803724)


if __name__ == "__main__":
  tf.test.main()
