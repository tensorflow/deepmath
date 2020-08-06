"""Tests for deepmath.deephol.to_sexpression."""

from typing import List, Optional, Text
from absl.testing import parameterized
import tensorflow.compat.v1 as tf
from deepmath.deephol import predictions
from deepmath.deephol import to_sexpression
from deepmath.deephol.utilities import sexpression_parser
from deepmath.proof_assistant import proof_assistant_pb2


def _goal(conclusion: Text, assumptions_strings: Optional[List[Text]] = None):
  assumptions = []
  if assumptions_strings:
    for term in assumptions_strings:
      assumptions.append(
          proof_assistant_pb2.Theorem(
              conclusion=term,
              hypotheses=[term],
              tag=proof_assistant_pb2.Theorem.THEOREM))
  return proof_assistant_pb2.Theorem(
      conclusion=conclusion,
      assumptions=assumptions,
      tag=proof_assistant_pb2.Theorem.GOAL)


def _theorems(terms):
  result = []
  for term in terms:
    result.append(
        proof_assistant_pb2.Theorem(
            conclusion=term,
            hypotheses=[term],
            tag=proof_assistant_pb2.Theorem.THEOREM))
  return result


class ToSexpressionTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('goal', proof_assistant_pb2.Theorem.GOAL, '<goal>'),
      ('theorem', proof_assistant_pb2.Theorem.THEOREM, '<theorem>'),
      ('definition', proof_assistant_pb2.Theorem.DEFINITION, '<theorem>'),
      ('type_definition', proof_assistant_pb2.Theorem.TYPE_DEFINITION,
       '<theorem>'))
  def test_no_hypotheses(self, theorem_tag, expected_special_token):
    theorem = proof_assistant_pb2.Theorem(
        conclusion='(v (fun (prod ?0 ?0) (bool)) P)',
        hypotheses=[],
        tag=theorem_tag)
    if theorem_tag == proof_assistant_pb2.Theorem.GOAL:
      self.assertEqual(
          to_sexpression.convert_goal(theorem, conclusion_only=True),
          '(v (fun (prod ?0 ?0) (bool)) P)')
      self.assertEqual(
          to_sexpression.convert_goal(theorem, conclusion_only=False),
          '(%s (v (fun (prod ?0 ?0) (bool)) P))' % expected_special_token)
      # theorem with GOAL tag
      with self.assertRaises(ValueError):
        to_sexpression.convert_theorem(theorem, conclusion_only=True)
      with self.assertRaises(ValueError):
        to_sexpression.convert_theorem(theorem, conclusion_only=False)
    else:
      self.assertEqual(
          to_sexpression.convert_theorem(theorem, conclusion_only=True),
          '(v (fun (prod ?0 ?0) (bool)) P)')
      self.assertEqual(
          to_sexpression.convert_theorem(theorem, conclusion_only=False),
          '(%s (v (fun (prod ?0 ?0) (bool)) P))' % expected_special_token)
      # goal with nonGOAL tag
      with self.assertRaises(ValueError):
        to_sexpression.convert_goal(theorem, conclusion_only=True)
      with self.assertRaises(ValueError):
        to_sexpression.convert_goal(theorem, conclusion_only=False)

  def test_one_assumption(self):
    theorem = proof_assistant_pb2.Theorem(
        conclusion='(a b c)',
        assumptions=_theorems(['(d e f)']),
        tag=proof_assistant_pb2.Theorem.GOAL)
    self.assertEqual(
        to_sexpression.convert_goal(theorem, conclusion_only=True), '(a b c)')
    self.assertEqual(
        to_sexpression.convert_goal(theorem, conclusion_only=False),
        '(<goal> (<theorem> (d e f) (d e f)) (a b c))')

  @parameterized.named_parameters(
      ('theorem', proof_assistant_pb2.Theorem.THEOREM),
      ('definition', proof_assistant_pb2.Theorem.DEFINITION),
      ('type_definition', proof_assistant_pb2.Theorem.TYPE_DEFINITION))
  def test_one_hypothesis(self, theorem_tag):
    theorem = proof_assistant_pb2.Theorem(
        conclusion='(a b c)', hypotheses=['(d e f)'], tag=theorem_tag)
    if theorem_tag == proof_assistant_pb2.Theorem.THEOREM:
      self.assertEqual(
          to_sexpression.convert_theorem(theorem, conclusion_only=True),
          '(a b c)')
      self.assertEqual(
          to_sexpression.convert_theorem(theorem, conclusion_only=False),
          '(<theorem> (d e f) (a b c))')
    else:
      # (type) definition with hypothesis terms
      with self.assertRaises(ValueError):
        to_sexpression.convert_theorem(theorem, conclusion_only=True)

  @parameterized.named_parameters(
      ('theorem', proof_assistant_pb2.Theorem.THEOREM),
      ('definition', proof_assistant_pb2.Theorem.DEFINITION),
      ('type_definition', proof_assistant_pb2.Theorem.TYPE_DEFINITION))
  def test_multiple_hypotheses(self, theorem_tag):
    theorem = proof_assistant_pb2.Theorem(
        conclusion='(con clu (s i on))',
        hypotheses=['(d e f)', 'asdf', '(i (j k l) m)', 'asdf'],
        tag=theorem_tag)
    if theorem_tag == proof_assistant_pb2.Theorem.THEOREM:
      self.assertEqual(
          to_sexpression.convert_theorem(theorem, conclusion_only=True),
          '(con clu (s i on))')
      self.assertEqual(
          to_sexpression.convert_theorem(theorem, conclusion_only=False),
          '(<theorem> (d e f) asdf (i (j k l) m) asdf (con clu (s i on)))')
    else:
      # (type) definition with hypothesis terms
      with self.assertRaises(ValueError):
        to_sexpression.convert_theorem(theorem, conclusion_only=True)

  def test_multiple_assumptions(self):
    theorem = proof_assistant_pb2.Theorem(
        conclusion='(con clu (s i on))',
        assumptions=_theorems(['(d e f)', 'asdf', '(i (j k l) m)', 'asdf']),
        tag=proof_assistant_pb2.Theorem.GOAL)
    self.assertEqual(
        to_sexpression.convert_goal(theorem, conclusion_only=True),
        '(con clu (s i on))')
    self.assertEqual(
        to_sexpression.convert_goal(theorem, conclusion_only=False),
        '(<goal> (<theorem> (d e f) (d e f)) (<theorem> asdf asdf)'
        ' (<theorem> (i (j k l) m) (i (j k l) m)) (<theorem> asdf asdf)'
        ' (con clu (s i on)))')

  @parameterized.parameters(
      (['(this one (is fi ne))', '(h (ypothesis) ypothesis'],),
      (['(h y p o t h e s i s'],))
  def test_faulty_hypothesis(self, asm_list):
    # We expect theorem protos of tag 'goal' to store each assumption theorem
    # as a hypothesis term 'x' that represents a trivial theorem 'x |- x'.
    # Some data sources may actually have hypothesis terms stored in the format
    # '(h x x)' or '(h (x) x)', we have a guard to catch and become aware of it.
    # Note this check is not performed if assumptions are marked to be ignored.
    theorem = proof_assistant_pb2.Theorem(
        conclusion='(con clu (s i on))',
        assumptions=_theorems(asm_list),
        tag=proof_assistant_pb2.Theorem.GOAL)
    self.assertEqual(
        to_sexpression.convert_goal(theorem, conclusion_only=True),
        '(con clu (s i on))')  # assumptions ignored

  @parameterized.parameters(([],), (['hypothesis'],))
  def test_empty_conclusion(self, hypo_list):
    # A theorem with '' conclusion and no hypotheses is passed to represent an
    # empty list of theorems (e.g. an empty list of tactic parameters).
    # A theorem with '' conclusion and hypotheses should never occur.
    # A goal with '' conclusion should never occur (regardless of assumptions).
    theorem = proof_assistant_pb2.Theorem(
        conclusion='',
        hypotheses=hypo_list,
        tag=proof_assistant_pb2.Theorem.THEOREM)
    if hypo_list:
      with self.assertRaises(ValueError):
        to_sexpression.convert_theorem(theorem, conclusion_only=False)
    else:
      self.assertEqual(
          to_sexpression.convert_theorem(theorem, conclusion_only=False), '')
    self.assertEqual(
        to_sexpression.convert_theorem(theorem, conclusion_only=True), '')
    theorem.tag = proof_assistant_pb2.Theorem.GOAL
    theorem.ClearField('hypotheses')  # otherwise an exception is thrown
    with self.assertRaises(ValueError):
      to_sexpression.convert_goal(theorem, conclusion_only=False)
    self.assertEqual(
        to_sexpression.convert_goal(theorem, conclusion_only=True), '')

  def test_convert_search_state(self):
    search_state = [
        _goal('con0', ['as0a']),
        _goal('con1', []),
        _goal('con2', ['clash']),
        _goal('con3', ['as3a', 'clash'])
    ]
    proof_state = predictions.ProofState(goal=search_state[0])
    with self.assertRaises(AssertionError):
      to_sexpression._convert_search_state(proof_state, conclusion_only=True)
    proof_state = predictions.ProofState(
        goal=search_state[0], search_state=search_state)
    self.assertEqual(
        to_sexpression._convert_search_state(proof_state, conclusion_only=True),
        '(<search_state> con0 con1 con2 con3)')
    self.assertEqual(
        to_sexpression._convert_search_state(
            proof_state, conclusion_only=False), '(<search_state> '
        '(<goal> (<theorem> as0a as0a) con0) (<goal> con1) '
        '(<goal> (<theorem> clash clash) con2) '
        '(<goal> (<theorem> as3a as3a) (<theorem> clash clash) con3))')

  def test_convert_proof_state(self):
    proof_state = predictions.ProofState(
        goal=_goal('con0', ['as0a']), previous_proof_state=None)
    proof_state = predictions.ProofState(
        goal=_goal('con1', []), previous_proof_state=proof_state)
    proof_state = predictions.ProofState(
        goal=_goal('con2', ['clash']), previous_proof_state=proof_state)
    proof_state = predictions.ProofState(
        goal=_goal('con3', ['as3a', 'clash']), previous_proof_state=proof_state)
    with self.assertRaises(AssertionError):
      to_sexpression.convert_proof_state(
          proof_state, history_bound=-1, conclusion_only=True)
    self.assertEqual(
        to_sexpression.convert_proof_state(
            proof_state, history_bound=0, conclusion_only=True), 'con3')
    self.assertLen(proof_state.goal.assumptions, 2)
    self.assertEqual(
        to_sexpression.convert_proof_state(
            proof_state, history_bound=0, conclusion_only=False),
        '(<goal> (<theorem> as3a as3a) (<theorem> clash clash) con3)')
    self.assertEqual(
        to_sexpression.convert_proof_state(
            proof_state, history_bound=3, conclusion_only=True),
        '(<proof_state_history> con0 con1 con2 con3)')
    self.assertEqual(
        to_sexpression.convert_proof_state(
            proof_state, history_bound=3, conclusion_only=False),
        '(<proof_state_history> '
        '(<goal> (<theorem> as0a as0a) con0) (<goal> con1) '
        '(<goal> (<theorem> clash clash) con2) '
        '(<goal> (<theorem> as3a as3a) (<theorem> clash clash) con3))')

  def test_convert_proof_state_history_bound_cutoff(self):
    proof_state = predictions.ProofState(
        goal=_goal('3_steps_old', None), previous_proof_state=None)
    proof_state = predictions.ProofState(
        goal=_goal('2_steps_old', None), previous_proof_state=proof_state)
    proof_state = predictions.ProofState(
        goal=_goal('1_step_old', None), previous_proof_state=proof_state)
    proof_state = predictions.ProofState(
        goal=_goal('current', None), previous_proof_state=proof_state)
    self.assertEqual(
        to_sexpression.convert_proof_state(
            proof_state, history_bound=0, conclusion_only=True), 'current')
    self.assertEqual(
        to_sexpression.convert_proof_state(
            proof_state, history_bound=0, conclusion_only=False),
        '(<goal> current)')
    self.assertEqual(
        to_sexpression.convert_proof_state(
            proof_state, history_bound=2, conclusion_only=True),
        '(<proof_state_history> 2_steps_old 1_step_old current)')
    self.assertEqual(
        to_sexpression.convert_proof_state(
            proof_state, history_bound=2, conclusion_only=False),
        '(<proof_state_history> '
        '(<goal> 2_steps_old) (<goal> 1_step_old) (<goal> current))')

  def test_convert_proof_state_with_search_state(self):
    proof_state = predictions.ProofState(
        search_state=[_goal('state0_g0', None)], previous_proof_state=None)
    proof_state = predictions.ProofState(
        search_state=[_goal('state1_g0', None),
                      _goal('state1_g1', None)],
        previous_proof_state=proof_state)
    proof_state = predictions.ProofState(
        search_state=[
            _goal('state2_g0', None),
            _goal('state2_g1', None),
            _goal('state2_g2', None)
        ],
        previous_proof_state=proof_state)
    self.assertEqual(
        to_sexpression.convert_proof_state(
            proof_state, history_bound=0, conclusion_only=True),
        '(<search_state> state2_g0 state2_g1 state2_g2)')
    self.assertEqual(
        to_sexpression.convert_proof_state(
            proof_state, history_bound=0, conclusion_only=False),
        '(<search_state> '
        '(<goal> state2_g0) (<goal> state2_g1) (<goal> state2_g2))')
    self.assertEqual(
        to_sexpression.convert_proof_state(
            proof_state, history_bound=1, conclusion_only=True),
        '(<proof_state_history> '
        '(<search_state> state1_g0 state1_g1) '
        '(<search_state> state2_g0 state2_g1 state2_g2))')
    self.assertEqual(
        to_sexpression.convert_proof_state(
            proof_state, history_bound=1, conclusion_only=False),
        '(<proof_state_history> '
        '(<search_state> (<goal> state1_g0) (<goal> state1_g1)) '
        '(<search_state> (<goal> state2_g0) (<goal> state2_g1) '
        '(<goal> state2_g2)))')
    self.assertEqual(
        to_sexpression.convert_proof_state(
            proof_state, history_bound=2, conclusion_only=True),
        '(<proof_state_history> '
        '(<search_state> state0_g0) '
        '(<search_state> state1_g0 state1_g1) '
        '(<search_state> state2_g0 state2_g1 state2_g2))')
    self.assertEqual(
        to_sexpression.convert_proof_state(
            proof_state, history_bound=2, conclusion_only=False),
        '(<proof_state_history> '
        '(<search_state> (<goal> state0_g0)) '
        '(<search_state> (<goal> state1_g0) (<goal> state1_g1)) '
        '(<search_state> (<goal> state2_g0) (<goal> state2_g1) '
        '(<goal> state2_g2)))')

  def test_convert_proof_state_with_hindsight(self):
    proof_state = predictions.ProofState(
        goal=_goal('con0', ['as0a']), previous_proof_state=None)

    without_target = to_sexpression.convert_proof_state(
        proof_state, history_bound=0, conclusion_only=False)
    sexpression_tree = sexpression_parser.to_tree(without_target)
    self.assertNotEqual('<HER>', str(sexpression_tree.children[0]))

    # Add a target and check out conditions on target.
    proof_state = proof_state._replace(targets=[proof_state.goal])
    with_target = to_sexpression.convert_proof_state(
        proof_state, history_bound=0, conclusion_only=False)
    sexpression_tree = sexpression_parser.to_tree(with_target)

    self.assertLen(sexpression_tree.children, 4)
    tag, left, _, right = sexpression_tree.children
    self.assertEqual('<HER>', str(tag))
    self.assertEqual(without_target, str(left))  # Original goal.
    right_tag, right_tree = right.children
    self.assertEqual('<target>', str(right_tag))
    self.assertEqual(without_target, str(right_tree))  # goal same as target.


if __name__ == '__main__':
  tf.test.main()
