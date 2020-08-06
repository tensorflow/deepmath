"""Tests for deepmath.deephol.proof_checker.proof_checker_lib."""

from absl.testing import parameterized
import tensorflow.compat.v1 as tf
from google.protobuf import text_format
from deepmath.deephol import deephol_pb2
from deepmath.deephol import tactic_utils
from deepmath.deephol.utilities import proof_checker_lib
from deepmath.proof_assistant import proof_assistant_pb2


class ProofCheckerTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(ProofCheckerTest, self).setUp()
    self.eq_refl_thm = proof_assistant_pb2.Theorem()
    text_format.Parse(
        'name: "EQ_REFL"'
        'conclusion: "(a (c (fun (fun A (bool)) (bool)) !) (l (v A x) (a (a'
        ' (c (fun A (fun A (bool))) =) (v A x)) (v A x))))"'
        'tag: THEOREM', self.eq_refl_thm)
    self.two_step_proof = deephol_pb2.ProofLog()
    text_format.Parse(
        'theorem_in_database {'
        '    conclusion: "(a (c (fun (fun A (bool)) (bool)) !) (l (v A x) (a (a'
        ' (c (fun A (fun A (bool))) =) (v A x)) (v A x))))"'
        '    tag: THEOREM'
        '    name: "EQ_REFL"'
        '}'
        'nodes {'
        '  goal {'
        '    conclusion: "(a (a (c (fun A (fun A (bool))) =) (v A x)) (v A x))"'
        '    tag: GOAL'
        '  }'
        '  proofs {'
        '    tactic: "REFL_TAC"'
        '  }'
        '}'
        'nodes {'
        '  goal {'
        '    conclusion: "(a (c (fun (fun A (bool)) (bool)) !) (l (v A x) (a (a'
        ' (c (fun A (fun A (bool))) =) (v A x)) (v A x))))"'
        '    tag: THEOREM'
        '    name: "EQ_REFL"'
        '  }'
        '  proofs {'
        '    tactic: "GEN_TAC"'
        '    subgoals {'
        '      conclusion: "(a (a (c (fun A (fun A (bool))) =) (v A x)) '
        '(v A x))"'
        '      tag: GOAL'
        '    }'
        '  }'
        '}', self.two_step_proof)

    self.proof_with_param = deephol_pb2.ProofLog()
    text_format.Parse(
        'theorem_in_database {'
        '  conclusion: "(a (c (fun (fun A (bool)) (bool)) !) (l (v A x) (a '
        '(a (c (fun A (fun A (bool))) =) (v A x)) (v A x))))"'
        '  tag: THEOREM'
        '  name: "EQ_REFL"'
        '}'
        'nodes {'
        '  goal {'
        '    conclusion: "(a (c (fun (fun A (bool)) (bool)) !) (l (v A x) (a '
        '(a (c (fun A (fun A (bool))) =) (v A x)) (v A x))))"'
        '    tag: THEOREM'
        '    name: "EQ_REFL"'
        '  }'
        '  proofs {'
        '    tactic: "ACCEPT_TAC"'
        '    parameters {'
        '      parameter_type: THEOREM'
        '      theorems {'
        '        conclusion: "(a (c (fun (fun A (bool)) (bool)) !) (l (v A x) '
        '(a (a (c (fun A (fun A (bool))) =) (v A x)) (v A x))))"'
        '        tag: THEOREM'
        '        name: "EQ_REFL"'
        '      }'
        '    }'
        '  }'
        '}', self.proof_with_param)

  def test_tactic_extraction(self):
    tactic_applications = proof_checker_lib.proof_linearization(
        self.two_step_proof)
    self.assertLen(tactic_applications, 2)
    tactic_strings = [
        tactic_utils.tactic_application_to_string(t_app)
        for t_app in tactic_applications
    ]
    self.assertEqual(tactic_strings, ['GEN_TAC', 'REFL_TAC'])

  def test_thm_parameter(self):
    tactic_applications = proof_checker_lib.proof_linearization(
        self.proof_with_param)
    self.assertLen(tactic_applications, 1)
    tactic_strings = list(
        map(tactic_utils.tactic_application_to_string, tactic_applications))
    self.assertEqual(tactic_strings, ['ACCEPT_TAC THM 3459657839204525272'])


if __name__ == '__main__':
  tf.test.main()
