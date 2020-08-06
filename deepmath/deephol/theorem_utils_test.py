# Lint as: python3
"""Tests for deepmath.deephol.theorem_utils."""

from absl.testing import parameterized
import tensorflow.compat.v1 as tf
from deepmath.public import compare
from google.protobuf import text_format
from deepmath.deephol import theorem_fingerprint
from deepmath.deephol import theorem_utils
from deepmath.proof_assistant import proof_assistant_pb2


class ToTheoremTest(parameterized.TestCase):

  def test_assume_term(self):
    expected_thm = proof_assistant_pb2.Theorem(
        hypotheses=['term'],
        conclusion='term',
        tag=proof_assistant_pb2.Theorem.THEOREM)
    actual_thm = theorem_utils.assume_term('term')
    compare.assertProto2Equal(self, expected_thm, actual_thm)

  @parameterized.parameters((proof_assistant_pb2.Theorem.GOAL),
                            (proof_assistant_pb2.Theorem.THEOREM),
                            (proof_assistant_pb2.Theorem.DEFINITION),
                            (proof_assistant_pb2.Theorem.TYPE_DEFINITION))
  def test_theorem_to_goal_proto(self, theorem_tag):
    theorem = proof_assistant_pb2.Theorem(
        conclusion='(a b c)',
        tag=theorem_tag,
        training_split=proof_assistant_pb2.Theorem.TRAINING)
    theorem.fingerprint = theorem_fingerprint.Fingerprint(theorem)
    expected_theorem = proof_assistant_pb2.Theorem()
    expected_theorem.CopyFrom(theorem)
    expected_goal = proof_assistant_pb2.Theorem(
        conclusion='(a b c)',
        tag=proof_assistant_pb2.Theorem.GOAL,
        training_split=proof_assistant_pb2.Theorem.TRAINING)
    expected_goal.fingerprint = theorem_fingerprint.Fingerprint(expected_goal)

    goal = theorem_utils.theorem_to_goal(theorem)
    # Assert that the function call did not alter the argument.
    self.assertEqual(expected_theorem, theorem)
    self.assertEqual(expected_goal, goal)

  def test_convert_legacy_goal(self):
    """We represented assumptions of goals as hypotheses."""
    expected = text_format.Parse(
        """conclusion: "1 + 1 = 2"
           tag: GOAL
           fingerprint: 2638304774168458552
           assumptions {
             hypotheses: "x = y"
             conclusion: "x = y"
             fingerprint: 4550921288261253481
             assumption_index: 0
             tag: THEOREM
           }""", proof_assistant_pb2.Theorem())
    legacy_goal = text_format.Parse(
        """
        hypotheses: "x = y"
        conclusion: "1 + 1 = 2"
        tag: GOAL""", proof_assistant_pb2.Theorem())
    theorem_utils.convert_legacy_goal(legacy_goal)
    compare.assertProto2Equal(self, expected, legacy_goal)

  def test_convert_legacy_goal_on_theorems(self):
    """convert_legacy_goal turns theorems into goals."""
    expected = text_format.Parse(
        """
        conclusion: "1 + 1 = 2"
        tag: GOAL
        fingerprint: 2638304774168458552
        assumptions {
          conclusion: "x = y"
          hypotheses: "x = y"
          assumption_index: 0
          tag: THEOREM
          fingerprint: 4550921288261253481
        }
        """, proof_assistant_pb2.Theorem())
    theorem = text_format.Parse(
        """
        hypotheses: "x = y"
        conclusion: "1 + 1 = 2"
        tag: THEOREM""", proof_assistant_pb2.Theorem())
    theorem_utils.convert_legacy_goal(theorem)
    compare.assertProto2Equal(self, expected, theorem)

  def test_convert_legacy_goal_unchanged(self):
    """convert_legacy_goal does not change new goals."""
    expected = text_format.Parse(
        """conclusion: "1 + 1 = 2"
           tag: GOAL
           fingerprint: 3233380672155685496
           assumptions {
             conclusion: "x = y"
             tag: THEOREM
             fingerprint: 778225982051770180
             assumption_index: 0
           }""", proof_assistant_pb2.Theorem())
    non_legacy_goal = text_format.Parse(
        """conclusion: "1 + 1 = 2"
           tag: GOAL
           fingerprint: 3233380672155685496
           assumptions {
             conclusion: "x = y"
             tag: THEOREM
             fingerprint: 778225982051770180
             assumption_index: 0
           }""", proof_assistant_pb2.Theorem())
    theorem_utils.convert_legacy_goal(non_legacy_goal)
    compare.assertProto2Equal(self, expected, non_legacy_goal)


if __name__ == '__main__':
  tf.test.main()
