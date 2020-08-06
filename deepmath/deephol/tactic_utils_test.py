# Lint as: python3
"""Tests for deepmath.deephol.tactic_utils."""

from typing import List
from absl.testing import parameterized
import tensorflow.compat.v1 as tf
from deepmath.public import compare
from google.protobuf import text_format
from deepmath.deephol import deephol_pb2
from deepmath.deephol import tactic_utils
from deepmath.proof_assistant import proof_assistant_pb2


def _tactic_parameter(parameter_type: int,
                      fingerprints: List[int]) -> deephol_pb2.TacticParameter:
  theorems = []
  for fingerprint in fingerprints:
    theorems.append(proof_assistant_pb2.Theorem(fingerprint=fingerprint))
  return deephol_pb2.TacticParameter(
      parameter_type=parameter_type, theorems=theorems)


class TacticUtilsTest(parameterized.TestCase):

  def test_build_tactic_string(self):
    param = deephol_pb2.TacticParameter(
        theorems=[
            proof_assistant_pb2.Theorem(fingerprint=123456),
            proof_assistant_pb2.Theorem(fingerprint=321, assumption_index=1)
        ],
        parameter_type=deephol_pb2.Tactic.THEOREM_LIST)
    expected_tactic = 'MESON_TAC [ THM 123456 ; ASSUM 1 ]'
    actual_tactic = tactic_utils.tactic_string('MESON_TAC', [param])
    self.assertEqual(actual_tactic, expected_tactic)

  def test_build_tactic_string_empty_param_list(self):
    param = deephol_pb2.TacticParameter(
        theorems=[], parameter_type=deephol_pb2.Tactic.THEOREM_LIST)
    expected_tactic = 'MESON_TAC [ ]'
    actual_tactic = tactic_utils.tactic_string('MESON_TAC', [param])
    self.assertEqual(actual_tactic, expected_tactic)

  def test_build_tactic_string_assumption_index_override(self):
    premise = text_format.Parse(
        """
        assumption_index: 0
        fingerprint: 1337
        tag: THEOREM
        """, proof_assistant_pb2.Theorem())
    assumption_indices = {1337: 42}
    parameter = deephol_pb2.TacticParameter(
        theorems=[premise], parameter_type=deephol_pb2.Tactic.THEOREM_LIST)
    expected_tactic = 'MESON_TAC [ ASSUM 42 ]'
    actual_tactic = tactic_utils.tactic_string(
        'MESON_TAC', [parameter], asm_indices=assumption_indices)
    self.assertEqual(actual_tactic, expected_tactic)

  def test_assumption_indices_inconsistency_raises(self):
    goal = text_format.Parse(
        """conclusion: "1 + 1 = 2"
           tag: GOAL
           assumptions {
             conclusion: "x = y"
             tag: THEOREM
             fingerprint: 778225982051770180
             assumption_index: 0
           }
           assumptions {
             conclusion: "x = y"
             tag: THEOREM
             fingerprint: 778225982051770180
             assumption_index: 2
           }""", proof_assistant_pb2.Theorem())
    with self.assertRaises(ValueError):
      tactic_utils.assumption_indices(goal)

  def test_assumption_indices_missing_ok(self):
    goal = text_format.Parse(
        """conclusion: "1 + 1 = 2"
           tag: GOAL
           assumptions {
             conclusion: "x = y"
             tag: THEOREM
             fingerprint: 778225982051770180
             assumption_index: 0
           }
           assumptions {
             conclusion: "a = b"
             tag: THEOREM
             fingerprint: 3398700195961844123
           }""", proof_assistant_pb2.Theorem())
    indices = tactic_utils.assumption_indices(goal)
    self.assertEqual(indices[3398700195961844123], 1)

  @parameterized.parameters(
      ('MESON_TAC [ THM 123456 ] ', 'MESON_TAC',
       deephol_pb2.Tactic.THEOREM_LIST,
       [proof_assistant_pb2.Theorem(fingerprint=123456)]),
      ('MESON_TAC [ ASSUM 0 ] ', 'MESON_TAC', deephol_pb2.Tactic.THEOREM_LIST, [
          proof_assistant_pb2.Theorem(
              conclusion='hyp0',
              hypotheses=['hyp0'],
              tag=proof_assistant_pb2.Theorem.THEOREM,
              assumption_index=0)
      ]),
      ('MESON_TAC [ THM 123456 ; ASSUM 1 ] ', 'MESON_TAC',
       deephol_pb2.Tactic.THEOREM_LIST, [
           proof_assistant_pb2.Theorem(fingerprint=123456),
           proof_assistant_pb2.Theorem(
               conclusion='hyp1',
               hypotheses=['hyp1'],
               tag=proof_assistant_pb2.Theorem.THEOREM,
               assumption_index=1)
       ]),
      ('SOME_THEOREM_TAC ASSUM 1', 'SOME_THEOREM_TAC',
       deephol_pb2.Tactic.THEOREM, [
           proof_assistant_pb2.Theorem(
               conclusion='hyp1',
               hypotheses=['hyp1'],
               tag=proof_assistant_pb2.Theorem.THEOREM,
               assumption_index=1)
       ]),
  )
  def test_extract_tactic_and_parameters(self, tactic_string, expected_tactic,
                                         expected_type, expected_theorems):
    expected_parameters = [
        deephol_pb2.TacticParameter(
            parameter_type=expected_type, theorems=expected_theorems)
    ]

    input_goal = text_format.Parse(
        """conclusion: "conclusion"
           tag: GOAL
           assumptions {
             conclusion: "hyp0"
             hypotheses: "hyp0"
             tag: THEOREM
             assumption_index: 0
           }
           assumptions {
             conclusion: "hyp1"
             hypotheses: "hyp1"
             tag: THEOREM
             assumption_index: 1
           }
           assumptions {
             conclusion: "hyp2"
             hypotheses: "hyp2"
             tag: THEOREM
             assumption_index: 2
           }""", proof_assistant_pb2.Theorem())
    actual_tactic, actual_parameters = (
        tactic_utils.extract_tactic_and_parameters(input_goal, tactic_string))

    self.assertEqual(expected_tactic, actual_tactic)
    self.assertLen(actual_parameters, 1)
    compare.assertProto2Equal(self, expected_parameters[0],
                              actual_parameters[0])

  def test_simple_tacticapplication(self):
    app = deephol_pb2.TacticApplication()
    text_format.Parse('tactic: "REFL_TAC"', app)
    tactic_str = tactic_utils.tactic_application_to_string(app)
    self.assertEqual(tactic_str, 'REFL_TAC')

  def test_complex_tacticapplication(self):
    app = deephol_pb2.TacticApplication()
    text_format.Parse(
        'tactic: "ACCEPT_TAC"'
        'parameters {'
        '  parameter_type: THEOREM'
        '  theorems {'
        '    conclusion: "(a (c (fun (fun A (bool)) (bool)) !) (l (v A x) '
        '(a (a (c (fun A (fun A (bool))) =) (v A x)) (v A x))))"'
        '    tag: THEOREM'
        '    name: "EQ_REFL"'
        '  }'
        '}', app)
    tactic_str = tactic_utils.tactic_application_to_string(app)
    self.assertEqual(tactic_str, 'ACCEPT_TAC THM 3459657839204525272')

  @parameterized.parameters(
      ('CONJ_TAC'), ('NOPE_TAC'), ('MATCH_MP_TAC THM 70976574409338618'),
      ('MADEUP-NAME THM 1337'), ('ASM_MESON_TAC [ ]'), ('PURE_REWRITE_TAC [ ]'),
      ('MESON_TAC [ THM 42 ]'),
      ('ASM_MESON_TAC [ THM 70976574409338618 ; THM 1913997688145082844 ; '
       'THM 347381884292247905 ; THM 2777965601207471458 ; '
       'THM 2510595948398174933 ; THM 3083213255038154291 ; '
       'THM 4501799997755390398 ; THM 3928630694610455386 ; '
       'THM 2169195022456018580 ; THM 1609619079003522969 ; '
       'THM 4236553219677541542 ; THM 1899501979378539728 ; '
       'THM 978834442080561172 ; THM 667963597854362001 ; '
       'THM 2341234844993805968 ]'),
      ('X_GEN_TAC `(v (fun (fun A (cart (real) (1))) (bool)) P)`'),
      ('RAW_POP_TAC 0'))
  def test_tactic_extract_and_to_string(self, tactic_string):
    """Function tactic_application_to_string inverts function extract_tactic."""
    tactic, parameters = tactic_utils.extract_tactic_and_parameters(
        proof_assistant_pb2.Theorem(), tactic_string)
    application = deephol_pb2.TacticApplication(
        tactic=tactic, parameters=parameters)
    self.assertEqual(
        tactic_utils.tactic_application_to_string(application), tactic_string)

  def test_tactic_to_string_and_extract_tactic_only(self):
    """Function extract_tactic inverts function tactic_application_to_string."""
    expected_name = 'TEST_TAC'
    application = deephol_pb2.TacticApplication(tactic=expected_name)
    tactic_string = tactic_utils.tactic_application_to_string(application)
    name, parameters = tactic_utils.extract_tactic_and_parameters(
        proof_assistant_pb2.Theorem(), tactic_string)
    self.assertEqual(expected_name, name)
    self.assertEmpty(parameters)

  @parameterized.named_parameters(
      ('theorem', deephol_pb2.Tactic.THEOREM, [12345]),
      ('theorem_list_zero', deephol_pb2.Tactic.THEOREM_LIST, []),
      ('theorem_list_one', deephol_pb2.Tactic.THEOREM_LIST, [123]),
      ('theorem_list_multiple', deephol_pb2.Tactic.THEOREM_LIST, [1, 2, 3]))
  def test_tactic_to_string_and_extract(self, parameter_type, fingerprints):
    """Function extract_tactic inverts function tactic_application_to_string."""
    expected_name = 'TEST_TAC'
    expected_parameters = [_tactic_parameter(parameter_type, fingerprints)]
    application = deephol_pb2.TacticApplication(
        tactic=expected_name, parameters=expected_parameters)
    tactic_string = tactic_utils.tactic_application_to_string(application)
    name, parameters = tactic_utils.extract_tactic_and_parameters(
        proof_assistant_pb2.Theorem(), tactic_string)
    self.assertEqual(expected_name, name)
    self.assertEqual(expected_parameters, parameters)

  def test_tactic_to_string_and_extract_term_parameter(self):
    """Function extract_tactic inverts function tactic_application_to_string."""
    expected_name = 'TERM_TAC'
    expected_parameters = [
        deephol_pb2.TacticParameter(
            parameter_type=deephol_pb2.Tactic.TERM, term='(a (b c))')
    ]
    application = deephol_pb2.TacticApplication(
        tactic=expected_name, parameters=expected_parameters)
    tactic_string = tactic_utils.tactic_application_to_string(application)
    name, parameters = tactic_utils.extract_tactic_and_parameters(
        proof_assistant_pb2.Theorem(), tactic_string)
    self.assertEqual(expected_name, name)
    self.assertEqual(expected_parameters, parameters)

  def test_tactic_to_string_and_extract_unknown_parameter(self):
    """Function extract_tactic inverts function tactic_application_to_string."""
    expected_name = 'TACTIC_WITH_UNKNOWN_PARAMETER_TYPE'
    expected_parameters = [
        deephol_pb2.TacticParameter(
            parameter_type=deephol_pb2.Tactic.UNKNOWN, unknown='unknown 123')
    ]
    application = deephol_pb2.TacticApplication(
        tactic=expected_name, parameters=expected_parameters)
    tactic_string = tactic_utils.tactic_application_to_string(application)
    name, parameters = tactic_utils.extract_tactic_and_parameters(
        proof_assistant_pb2.Theorem(), tactic_string)
    self.assertEqual(expected_name, name)
    self.assertEqual(expected_parameters, parameters)


if __name__ == '__main__':
  tf.test.main()
