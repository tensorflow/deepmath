"""Tests for deepmath.deephol.mcts_prover."""

from absl.testing import parameterized
import numpy
import tensorflow.compat.v1 as tf
from deepmath.deephol import action_generator
from deepmath.deephol import deephol_pb2
from deepmath.deephol import mcts
from deepmath.deephol import mcts_prover
from deepmath.deephol import mock_predictions
from deepmath.deephol import prover_environment
from deepmath.deephol import prover_util
from deepmath.deephol import theorem_utils
from deepmath.proof_assistant import proof_assistant_pb2

MOCK_THEOREM = proof_assistant_pb2.Theorem(
    name='theorem_to_prove',
    conclusion='theorem_conclusion',
    tag=proof_assistant_pb2.Theorem.THEOREM,
    training_split=proof_assistant_pb2.Theorem.TESTING)

MOCK_TARGET = proof_assistant_pb2.Theorem(
    conclusion='target', tag=proof_assistant_pb2.Theorem.GOAL)

MOCK_PREDICTOR = mock_predictions.MOCK_PREDICTOR


def has_proof(proof_log: deephol_pb2.ProofLog) -> bool:
  root, *_ = (node for node in proof_log.nodes if node.root_goal)
  return root.status == deephol_pb2.ProofNode.PROVED


class MockProofAssistant(object):

  def ApplyTactic(self, request):  # pylint: disable=invalid-name
    response = proof_assistant_pb2.ApplyTacticResponse()
    tactic = str(request.tactic)
    if tactic == 'prove_all':
      response.goals.goals.extend([])  # empty goal list means closed goal
    elif tactic == 'successful_tactic':
      next_goal = proof_assistant_pb2.Theorem()
      next_goal.tag = proof_assistant_pb2.Theorem.GOAL
      next_goal.conclusion = (
          request.goal.conclusion + '__applied_successful_tactic')
      response.goals.goals.extend([next_goal])
    elif tactic == 'fail_always':
      response.error = 'error: failure'
    elif tactic == 'timeout_tactic':
      response.error = 'error: timeout'
    elif tactic == 'zero_probability':
      raise ValueError('This action is not supposed to be selected.')
    elif tactic == 'unchanged_tac':
      response.goals.goals.extend([request.goal])
    elif tactic == 'branch_tactic':
      new_goal0 = proof_assistant_pb2.Theorem(
          conclusion=request.goal.conclusion + '_0',
          tag=proof_assistant_pb2.Theorem.GOAL)
      new_goal1 = proof_assistant_pb2.Theorem(
          conclusion=request.goal.conclusion + '_1',
          tag=proof_assistant_pb2.Theorem.GOAL)
      response.goals.goals.extend([new_goal0, new_goal1])
    elif tactic == 'reach_target':
      new_goal = proof_assistant_pb2.Theorem()
      new_goal.CopyFrom(MOCK_TARGET)
      response.goals.goals.extend([new_goal])
    else:
      raise ValueError('unknown tactic in mock: %s' % tactic)
    return response


class MockActionGenerator(action_generator.ActionGenerator):

  def __init__(self, suggestions):
    """Initializer with suggestions given with probabilities instead of logits."""
    assert abs(1.0 - sum([p for _, p in suggestions])) < 0.00000001
    self.suggestions = [
        action_generator.Suggestion(
            tactic=tactic, params=[], score=self.probability_to_logit(score))
        for tactic, score in suggestions
    ]

  def probability_to_logit(self, p: float) -> float:
    return numpy.log(p / (1.0 - p))

  def step(self, goal, premise_set):
    return self.suggestions


MOCK_WRAPPER = MockProofAssistant()
MOCK_PREMISE_SET = prover_util.make_premise_set(MOCK_THEOREM, 'default')
MOCK_THEOREM_DATABASE = proof_assistant_pb2.TheoremDatabase(
    theorems=[MOCK_THEOREM])
MOCK_TASK = proof_assistant_pb2.ProverTask(
    goals=[theorem_utils.theorem_to_goal(MOCK_THEOREM)],
    premise_set=MOCK_PREMISE_SET)


class MCTSProverTest(tf.test.TestCase, parameterized.TestCase):

  def build_mock_prover(self, suggestions=None, mcts_options=None):
    if mcts_options:
      self.search_options = mcts_options
    else:
      self.search_options = deephol_pb2.MCTSOptions(max_expansions=100)
    self.options = deephol_pb2.ProverOptions(mcts_options=self.search_options)
    if suggestions:
      self.mock_suggestions = suggestions
    else:
      self.mock_suggestions = [('successful_tactic', .5), ('unchanged_tac', .2),
                               ('prove_all', .1), ('fail_always', .1),
                               ('timeout_tactic', .09), ('unchanged_tac', .01),
                               ('zero_probability', .0)]
    self.mock_generator = MockActionGenerator(self.mock_suggestions)
    self.env = prover_environment.ProverEnvironment(self.options, MOCK_WRAPPER,
                                                    self.mock_generator,
                                                    MOCK_PREDICTOR)
    self.prover = mcts_prover.MCTSProver(
        self.options, MOCK_WRAPPER, MOCK_THEOREM_DATABASE, env=self.env)
    self.root_state = self.env.reset(MOCK_TASK)
    self.search = mcts.MonteCarloTreeSearch(self.search_options)

  def setUp(self):
    super(MCTSProverTest, self).setUp()
    self.build_mock_prover()

  def test_root_not_terminal(self):
    self.assertFalse(self.root_state.is_terminal())

  def test_generate_actions(self):
    self.assertLen(self.root_state.actions(), 7)

  def test_select_first_action(self):
    action = self.root_state.select_action()
    self.assertEqual(action.tactic, 'successful_tactic')

  def test_select_second_action(self):
    action = self.root_state.select_action()
    # Increasing the visit count will reduce the value of the first tactic
    self.root_state.visit_count = 10
    action.visit_count = 10
    action = self.root_state.select_action()
    self.assertEqual(action.tactic, 'unchanged_tac')

  def test_select(self):
    path = self.search._select(self.root_state)
    self.assertLen(path, 1)
    self.assertEqual(path[0].tactic, 'successful_tactic')

  def test_expand(self):
    path = [self.root_state.actions()[0]]
    state = path[-1].expand()
    self.assertLen(state.nodes, 1)
    self.assertEqual(state.nodes[0].goal.conclusion,
                     'theorem_conclusion__applied_successful_tactic')

  def test_backup(self):
    self.assertEqual(self.root_state.visit_count, 0)
    self.assertEqual(self.root_state.visit_count, 0)
    path = [self.root_state.actions()[0]]
    _ = path[-1].expand()
    self.search._backup(path)
    self.assertEqual(self.root_state.visit_count, 1)
    self.assertEqual(path[0].visit_count, 1)
    self.assertEqual(path[0].accumulated_value, 0.0)

  def test_close_proof(self):
    self.assertFalse(self.root_state.nodes[0].closed)
    path = [self.root_state.actions()[2]]
    _ = path[-1].expand()
    self.search._backup(path)
    self.assertEqual(self.root_state.visit_count, 1)
    self.assertEqual(path[0].visit_count, 1)
    self.assertEqual(path[0].accumulated_value, 1.0)
    self.assertTrue(self.root_state.nodes[0].closed)

  def test_two_step_proof(self):
    path = [self.root_state.actions()[0]]
    _ = path[-1].expand()
    self.search._backup(path)

    path.append(path[0].successor.actions()[2])
    _ = path[-1].expand()
    self.search._backup(path)
    self.assertEqual(self.root_state.visit_count, 2)
    self.assertEqual(path[0].visit_count, 2)
    self.assertEqual(path[1].state.visit_count, 1)
    self.assertEqual(path[1].visit_count, 1)
    self.assertEqual(path[0].accumulated_value,
                     self.search_options.discount_factor)

  def test_select_actions_frequencies(self):
    self.build_mock_prover([('successful_tactic', .7), ('fail_always', .3)])
    actions = self.root_state.actions()
    self.assertLen(actions, 2)
    self.assertAlmostEqual(actions[0].probability, 0.7)
    self.assertAlmostEqual(actions[1].probability, 0.3)
    for _ in range(10):
      action = self.root_state.select_action()
      if not action.is_expanded():
        _ = action.expand()
      self.search._backup([action])
      for a in actions:
        if not a.successor:
          continue
        self.assertEqual(a.successor.value(), 0.0)
    self.assertEqual(actions[0].visit_count, 7)
    self.assertEqual(actions[1].visit_count, 3)

  def test_action_frequencies_with_successful_tactic(self):
    self.build_mock_prover([('successful_tactic', .7), ('prove_all', .3)])
    actions = self.root_state.actions()
    self.assertLen(actions, 2)
    self.assertAlmostEqual(actions[0].probability, 0.7)
    self.assertAlmostEqual(actions[1].probability, 0.3)
    for _ in range(10):
      action = self.root_state.select_action()
      if not action.is_expanded():
        _ = action.expand()
      self.search._backup([action])
      for a in actions:
        if not a.successor:
          continue
    self.assertLess(actions[0].visit_count, 7)
    self.assertGreater(actions[1].visit_count, 3)

  def test_prove_smoke_test(self):
    timeout_seconds = 1.0
    self.search.search(self.root_state, timeout_seconds)

  def test_proof_successful(self):
    self.prover.timeout_seconds = 1
    proof_log = self.prover.prove(MOCK_TASK)
    self.assertEqual(self.prover.search.best_path[0].best_action().tactic,
                     'prove_all')
    self.assertEqual(proof_log.error_message, '')

  def test_proof_len_1(self):
    self.prover.timeout_seconds = 1
    self.prover.prove(MOCK_TASK)
    self.assertLen(self.prover.search.best_path, 2)

  def test_unsuccessful_search(self):
    self.build_mock_prover([('successful_tactic', .7), ('fail_always', .3)])
    self.prover.timeout_seconds = .5
    proof_log = self.prover.prove(MOCK_TASK)
    self.assertGreater(len(self.prover.search.best_path), 30)
    self.assertEqual(proof_log.nodes[0].status, deephol_pb2.ProofNode.UNKNOWN)

  def test_unsuccessful_search_with_branching(self):
    self.build_mock_prover([('successful_tactic', .7), ('branch_tactic', .3)])
    self.prover.timeout_seconds = .5
    self.prover.prove(MOCK_TASK)
    self.assertGreater(len(self.prover.search.best_path), 1)
    self.assertLess(len(self.prover.search.best_path), 100)

  def test_unsuccessful_search_total_expansion_limit(self):
    self.build_mock_prover(
        mcts_options=deephol_pb2.MCTSOptions(
            max_expansions=100, max_total_expansions=50),
        suggestions=[('successful_tactic', .7), ('branch_tactic', .3)])
    self.prover.timeout_seconds = .5
    self.prover.prove(MOCK_TASK)
    # because unfinished searches should not result in a training example:
    self.assertEmpty(self.prover.search.best_path)

  def test_unsuccessful_search_search_depth(self):
    self.build_mock_prover(
        suggestions=[('successful_tactic', .7), ('fail_always', .3)],
        mcts_options=deephol_pb2.MCTSOptions(
            max_expansions=100, max_search_depth=9))
    self.prover.timeout_seconds = .5
    self.prover.prove(MOCK_TASK)
    self.assertLen(self.prover.search.best_path, 9)

  def test_reach_target_fail(self):
    self.build_mock_prover(
        suggestions=[('reach_target', 0.9), ('reach_target', 0.1)],  # HACK
        mcts_options=deephol_pb2.MCTSOptions(
            max_expansions=100, max_search_depth=3))
    self.prover.timeout_seconds = .5

    proof_log = self.prover.prove(MOCK_TASK)
    self.assertFalse(has_proof(proof_log))
    self.assertEmpty(self.prover.env.tree.targets)

    for state in self.prover.search.best_path[:-1]:
      self.assertLen(state.nodes, 1)
    self.assertFalse(self.prover.search.best_path[-1].target_reached())

    self.assertLen(self.prover.search.best_path, 3)

  def test_reach_target_success(self):
    self.build_mock_prover(
        suggestions=[('reach_target', 0.9), ('reach_target', 0.1)],  # HACK
        mcts_options=deephol_pb2.MCTSOptions(
            max_expansions=4, max_search_depth=4))
    self.prover.timeout_seconds = 1

    targeted_task = proof_assistant_pb2.ProverTask()
    targeted_task.CopyFrom(MOCK_TASK)
    targeted_task.targets.append(MOCK_TARGET)

    proof_log = self.prover.prove(targeted_task)
    self.assertLen(self.prover.env.tree.targets, 1)
    self.assertTrue(self.prover.search.best_path[-1].target_reached())
    self.assertLen(self.prover.search.best_path, 2)
    self.assertTrue(has_proof(proof_log))


if __name__ == '__main__':
  tf.test.main()
