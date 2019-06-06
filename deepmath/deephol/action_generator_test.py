"""Tests for third_party.deepmath.deephol.action_generator."""

from __future__ import absolute_import
from __future__ import division
# Import Type Annotations
from __future__ import print_function
from absl import flags
from absl.testing import parameterized
import tensorflow as tf
from typing import List
from google.protobuf import text_format
from deepmath.deephol import action_generator
from deepmath.deephol import deephol_pb2
from deepmath.deephol import embedding_store
from deepmath.deephol import holparam_predictor
from deepmath.deephol import proof_search_tree
from deepmath.deephol import prover_util
from deepmath.deephol import test_util
from deepmath.deephol import theorem_fingerprint
from deepmath.proof_assistant import proof_assistant_pb2

FLAGS = flags.FLAGS
PREDICTIONS_MODEL_PREFIX = test_util.test_src_dir_path(
    'deephol/test_data/default_ckpt/model.ckpt-0')
HOLLIGHT_TACTICS_TEXTPB_PATH = test_util.test_src_dir_path(
    'deephol/data/hollight_tactics.textpb')

EQ_REFL = (
    '(a (c (fun (fun A (bool)) (bool)) !) (l (v A x) (a (a (c (fun A (fun A '
    '(bool))) =) (v A x)) (v A x))))')

EQ_SYM = (
    '(a (c (fun (fun A (bool)) (bool)) !) (l (v A x) (a (c (fun (fun A '
    '(bool)) (bool)) !) (l (v A y) (a (a (c (fun (bool) (fun (bool) (bool))) '
    '==>) (a (a (c (fun A (fun A (bool))) =) (v A x)) (v A y))) (a (a (c '
    '(fun A (fun A (bool))) =) (v A y)) (v A x)))))))')


def load_tactics(filename) -> List[deephol_pb2.Tactic]:
  tactics_info = deephol_pb2.TacticsInfo()
  with tf.gfile.GFile(filename, 'r') as f:
    text_format.MergeLines(f, tactics_info)
  return tactics_info.tactics


class MockProofAssistantWrapper(object):

  # We need to conform to the existing API naming.
  def ApplyTactic(self, request):  # pylint: disable=invalid-name
    response = proof_assistant_pb2.ApplyTacticResponse()
    tactic = str(request.tactic)
    head = tactic[0]
    tail = tactic[1:]
    if head == 'a':  # Append the rest of the characters to goal
      response.goals.goals.add(goal=str(request.goal.goal) + tail)
    elif head == 'b':  # Branch: append each character in separation.
      for c in tail:
        response.goals.goals.add(goal=str(request.goal.goal) + c)
    elif head == 'c':  # Close the goal.
      response.goals.goals.extend([])
    elif head == 'r':  # Replace: replace the goal with the string in tail.
      response.goals.goals.add(goal=tail)
    elif head == 'e':  # Error: produce the error specified in tail.
      response.error = tail
    return response


class MockActionGenerator(action_generator.ActionGenerator):

  def __init__(self, suggestions):
    self.suggestions = suggestions

  def step(self, goal, premise_set):
    return self.suggestions


def mock_generator(*tactic_scores):
  return MockActionGenerator([
      action_generator.Suggestion(string=tactic, score=score)
      for tactic, score in tactic_scores
  ])


MOCK_WRAPPER = MockProofAssistantWrapper()
MOCK_THEOREM = proof_assistant_pb2.Theorem(
    name='TH1',
    hypotheses=['h'],
    conclusion='c',
    tag=proof_assistant_pb2.Theorem.THEOREM,
    training_split=proof_assistant_pb2.Theorem.TESTING)
MOCK_PREMISE_SET = prover_util.make_premise_set(MOCK_THEOREM, 'default')


class ActionGeneratorTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    """Restoring the graph takes a lot of memory, so we do it only once here."""

    cls.predictor = holparam_predictor.HolparamPredictor(
        PREDICTIONS_MODEL_PREFIX)
    cls.model_architecture = deephol_pb2.ProverOptions.PAIR_DEFAULT
    cls.theorem_database = proof_assistant_pb2.TheoremDatabase()
    cls.theorem_database.theorems.add(name='EQ_REFL', conclusion=EQ_REFL)
    cls.theorem_database.theorems.add(name='EQ_SYM', conclusion=EQ_SYM)
    cls.test_goal_index = 1
    cls.test_theorem = cls.theorem_database.theorems[cls.test_goal_index]
    cls.test_goal = cls.theorem_database.theorems[cls.test_goal_index]
    cls.test_premise_set = prover_util.make_premise_set(cls.test_theorem,
                                                        'default')
    cls.options = deephol_pb2.ActionGeneratorOptions()

  def setUp(self):
    self.tree = proof_search_tree.ProofSearchTree(MOCK_WRAPPER, MOCK_THEOREM)
    self.node = proof_search_tree.ProofSearchNode(self.tree, 0, self.test_goal)

  def test_action_generator_hol_light_tactics_sanity_check(self):
    """HolLight tactics sanity test.

    This is a sanity check to ensure action generator works with actual HolLight
    tactics on which the test model was trained.
    """
    hollight_tactics = load_tactics(HOLLIGHT_TACTICS_TEXTPB_PATH)
    action_gen = action_generator.ActionGenerator(self.theorem_database,
                                                  hollight_tactics,
                                                  self.predictor, self.options,
                                                  self.model_architecture)
    actions_with_scores = action_gen.step(self.node, self.test_premise_set)
    for action, score in sorted(actions_with_scores, key=lambda x: x.score):
      tf.logging.info(str(score) + ': ' + str(action))
    self.assertIn('EQ_TAC', [action for action, _ in actions_with_scores])

  def test_action_generator_no_parameter_tactic(self):
    no_param_tactic = deephol_pb2.Tactic(name='TAC')
    action_gen = action_generator.ActionGenerator(self.theorem_database,
                                                  [no_param_tactic],
                                                  self.predictor, self.options,
                                                  self.model_architecture)
    actions_scores = action_gen.step(self.node, self.test_premise_set)
    self.assertEqual(1, len(actions_scores))
    self.assertEqual(actions_scores[0].string, 'TAC')

  def test_action_generator_unknown_parameter_tactic(self):
    unknown_param_tactic = deephol_pb2.Tactic(
        name='TAC', parameter_types=[deephol_pb2.Tactic.UNKNOWN])
    action_gen = action_generator.ActionGenerator(self.theorem_database,
                                                  [unknown_param_tactic],
                                                  self.predictor, self.options,
                                                  self.model_architecture)
    actions_scores = action_gen.step(self.node, self.test_premise_set)
    self.assertEqual(0, len(actions_scores))

  def test_action_generator_theorem_parameter_tactic(self):
    thm_param_tactic = deephol_pb2.Tactic(
        name='TAC', parameter_types=[deephol_pb2.Tactic.THEOREM])
    action_gen = action_generator.ActionGenerator(self.theorem_database,
                                                  [thm_param_tactic],
                                                  self.predictor, self.options,
                                                  self.model_architecture)
    actions_scores = action_gen.step(self.node, self.test_premise_set)
    self.assertEqual(1, len(actions_scores))
    expected = 'TAC ' + theorem_fingerprint.ToTacticArgument(
        self.theorem_database.theorems[0])
    self.assertEqual(expected, actions_scores[0].string)

  @parameterized.named_parameters(('WithEmbeddingStore', True),
                                  ('WithoutEmbeddingStore', False))
  def test_action_generator_theorem_list_parameter_tactic(
      self, use_embedding_store):
    """Checks max_theorem_parameters parameters are passed for a thmlist tactic.

    Args:
      use_embedding_store: True if the embedding store should be used.
    """
    max_parameters = self.options.max_theorem_parameters
    emb_store = None
    thmlist_param_tactic = deephol_pb2.Tactic(
        name='TAC', parameter_types=[deephol_pb2.Tactic.THEOREM_LIST])
    dummy_theorem = proof_assistant_pb2.Theorem(name='THM', conclusion='foo')
    theorem_database = proof_assistant_pb2.TheoremDatabase()
    theorem_database.theorems.extend([
        proof_assistant_pb2.Theorem(name='THM%d' % i, conclusion='foo')
        for i in range(2 * max_parameters + 1)
    ])
    if use_embedding_store:
      emb_store = embedding_store.TheoremEmbeddingStore(self.predictor)
      emb_store.compute_embeddings_for_thms_from_db(theorem_database)
    action_gen = action_generator.ActionGenerator(theorem_database,
                                                  [thmlist_param_tactic],
                                                  self.predictor, self.options,
                                                  self.model_architecture,
                                                  emb_store)
    test_theorem = theorem_database.theorems[2 * max_parameters]
    actions_scores = action_gen.step(
        self.node, prover_util.make_premise_set(test_theorem, 'default'))
    self.assertStartsWith(actions_scores[-1].string, 'TAC')
    self.assertEqual(max_parameters, actions_scores[-1].string.count('THM'))

  @parameterized.named_parameters(
      ('EMPTY', [], [''], False),
      ('THEOREM', [deephol_pb2.Tactic.THEOREM], [' THM1'], False),
      ('THEOREM_LIST', [deephol_pb2.Tactic.THEOREM_LIST
                       ], [' [ THM1 ; THM2 ; THM3 ]'], False),
      ('EMPTY_THEOREM_LIST', [deephol_pb2.Tactic.THEOREM_LIST
                             ], [' [ ]', ' [ THM1 ; THM2 ; THM3 ]'], True))
  def test_compute_parameter_string(self, types, expected_params,
                                    pass_no_arguments):
    actual_params = action_generator._compute_parameter_string(
        types=types,
        pass_no_arguments=pass_no_arguments,
        thm_ranked=[(10, 'THM1'), (1, 'THM2'), (-10, 'THM3')])
    self.assertSameElements(expected_params, actual_params)

  def test_compute_parameter_string_unknown(self):
    types = [deephol_pb2.Tactic.UNKNOWN]
    self.assertRaises(ValueError, action_generator._compute_parameter_string,
                      types, False, [(1, ' THM1')])


if __name__ == '__main__':
  tf.test.main()
