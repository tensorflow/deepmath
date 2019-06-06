"""Tests for third_party.deepmath.deephol.proof_search_tree."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from deepmath.deephol import action_generator
from deepmath.deephol import proof_search_tree
from deepmath.deephol import prover_util
from deepmath.proof_assistant import proof_assistant_pb2

PER_TACTIC_TIMEOUT_MS = 5000


class MockHolLightWrapper(object):

  # We need to conform to the existing API naming.
  def ApplyTactic(self, request):  # pylint: disable=invalid-name
    response = proof_assistant_pb2.ApplyTacticResponse()
    tactic = str(request.tactic)
    head = tactic[0]
    tail = tactic[1:]
    if head == 'a':  # Append the rest of the characters to goal
      response.goals.goals.add(conclusion=str(request.goal.conclusion) + tail)
    elif head == 'b':  # Branch: append each character in separation.
      for c in tail:
        response.goals.goals.add(conclusion=str(request.goal.conclusion) + c)
    elif head == 'c':  # Close the goal.
      response.goals.goals.extend([])
    elif head == 'r':  # Replace: replace the goal with the string in tail.
      response.goals.goals.add(conclusion=tail)
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


MOCK_WRAPPER = MockHolLightWrapper()
MOCK_THEOREM = proof_assistant_pb2.Theorem(
    name='TH1',
    hypotheses=['h'],
    conclusion='c',
    tag=proof_assistant_pb2.Theorem.THEOREM,
    training_split=proof_assistant_pb2.Theorem.TESTING)
MOCK_PREMISE_SET = prover_util.make_premise_set(MOCK_THEOREM, 'default')


class ProverUtilTest(tf.test.TestCase):

  def setUp(self):
    tf.logging.info('Setting up tree...')
    self.tree = proof_search_tree.ProofSearchTree(MOCK_WRAPPER, MOCK_THEOREM)

  def test_create_initial_tree(self):
    tree = self.tree
    self.assertEqual(len(tree.nodes), 1)
    self.assertEqual(len(tree.nodes_map), 1)

    def test_node(node):
      self.assertEqual(node.index, 0)
      self.assertEqual(str(node.goal.conclusion), 'c')
      self.assertEqual([str(hyp) for hyp in node.goal.hypotheses], ['h'])

    test_node(tree.nodes[0])
    self.assertEqual(tree.nodes_map['h|:|c'], 0)
    request = proof_assistant_pb2.ApplyTacticRequest(
        goal=proof_assistant_pb2.Theorem(conclusion='g'), tactic='axy')
    self.assertEqual([
        str(goal.conclusion)
        for goal in tree.proof_assistant.ApplyTactic(request).goals.goals
    ], ['gxy'])
    request = proof_assistant_pb2.ApplyTacticRequest(
        goal=proof_assistant_pb2.Theorem(conclusion='g'), tactic='rxy')
    self.assertEqual([
        str(goal.conclusion)
        for goal in tree.proof_assistant.ApplyTactic(request).goals.goals
    ], ['xy'])
    request = proof_assistant_pb2.ApplyTacticRequest(
        goal=proof_assistant_pb2.Theorem(conclusion='g'), tactic='bxy')
    self.assertEqual([
        str(goal.conclusion)
        for goal in tree.proof_assistant.ApplyTactic(request).goals.goals
    ], ['gx', 'gy'])
    request = proof_assistant_pb2.ApplyTacticRequest(
        goal=proof_assistant_pb2.Theorem(conclusion='g'), tactic='c')
    self.assertEqual([
        str(goal.conclusion)
        for goal in tree.proof_assistant.ApplyTactic(request).goals.goals
    ], [])
    request = proof_assistant_pb2.ApplyTacticRequest(
        goal=proof_assistant_pb2.Theorem(conclusion='g'), tactic='err')
    self.assertEqual(str(tree.proof_assistant.ApplyTactic(request).error), 'rr')

  def check_log_consistency(self):
    tree = self.tree
    proof_log = tree.to_proto()
    self.assertEqual(len(proof_log.nodes), len(tree.nodes))
    for i, node in enumerate(tree.nodes):
      log_node = proof_log.nodes[i]
      log_goal = log_node.goal
      self.assertEqual(node.goal.conclusion, log_goal.conclusion)
      for j, hyp in enumerate(node.goal.hypotheses):
        self.assertEqual(hyp, log_goal.hypotheses[j])
      self.assertEqual(
          len(node.failed_attempts) + len(node.successful_attempts),
          len(log_node.proofs))

  def test_apply_one_tactic(self):
    tree = self.tree
    root = tree.nodes[0]
    gen = mock_generator(('axy', 1.0))
    prover_util.try_tactics(root, 10, 0, 10, MOCK_PREMISE_SET, gen,
                            PER_TACTIC_TIMEOUT_MS)
    self.assertEqual(len(tree.nodes), 2)
    for i, node in enumerate(tree.nodes):
      self.assertEqual(node.index, i)
    node = tree.nodes[1]
    self.assertEqual(len(node.parents), 1)
    self.assertEqual(str(node.goal.conclusion), 'cxy')
    self.assertEqual(len(root.successful_attempts), 1)
    self.assertEqual(len(root.failed_attempts), 0)
    self.assertEqual(root.closed, False)
    self.assertEqual(node.closed, None)
    proof_search_tree.check_tree_consistency(tree)
    self.check_log_consistency()

  def test_apply_one_and_close(self):
    tree = self.tree
    root = tree.nodes[0]
    gen = mock_generator(('axy', 1.0))
    prover_util.try_tactics(root, 10, 0, 10, MOCK_PREMISE_SET, gen,
                            PER_TACTIC_TIMEOUT_MS)
    self.assertEqual(len(tree.nodes), 2)
    node = tree.nodes[1]
    gen = mock_generator(('c', 1.0))
    prover_util.try_tactics(node, 10, 0, 10, MOCK_PREMISE_SET, gen,
                            PER_TACTIC_TIMEOUT_MS)
    self.assertEqual(len(tree.nodes), 2)
    self.assertEqual(tree.nodes[1].closed, True)
    self.assertEqual(tree.nodes[0].closed, True)
    proof_search_tree.check_tree_consistency(tree)
    self.check_log_consistency()

  def test_apply_two_in_a_row_and_close(self):
    tree = self.tree
    root = tree.nodes[0]
    gen = mock_generator(('axy', 1.0))
    prover_util.try_tactics(root, 10, 0, 10, MOCK_PREMISE_SET, gen,
                            PER_TACTIC_TIMEOUT_MS)
    self.assertEqual(len(tree.nodes), 2)
    node = tree.nodes[1]
    gen = mock_generator(('az', 1.0))
    prover_util.try_tactics(node, 10, 0, 10, MOCK_PREMISE_SET, gen,
                            PER_TACTIC_TIMEOUT_MS)
    self.assertEqual(len(tree.nodes), 3)
    node = tree.nodes[2]
    self.assertEqual(str(node.goal.conclusion), 'cxyz')
    gen = mock_generator(('c', 1.0))
    prover_util.try_tactics(node, 10, 0, 10, MOCK_PREMISE_SET, gen,
                            PER_TACTIC_TIMEOUT_MS)
    self.assertEqual(len(tree.nodes), 3)
    self.assertEqual(tree.nodes[2].closed, True)
    self.assertEqual(tree.nodes[1].closed, True)
    self.assertEqual(tree.nodes[0].closed, True)
    proof_search_tree.check_tree_consistency(tree)
    self.check_log_consistency()

  def test_apply_two_in_parallel_and_close_both(self):
    tree = self.tree
    root = tree.nodes[0]
    gen = mock_generator(('bxy', 1.0))
    prover_util.try_tactics(root, 10, 0, 10, MOCK_PREMISE_SET, gen,
                            PER_TACTIC_TIMEOUT_MS)
    self.assertEqual(len(tree.nodes), 3)
    self.assertEqual(str(tree.nodes[1].goal.conclusion), 'cx')
    self.assertEqual(str(tree.nodes[2].goal.conclusion), 'cy')
    gen = mock_generator(('c', 1.0))
    prover_util.try_tactics(tree.nodes[1], 10, 0, 10, MOCK_PREMISE_SET, gen,
                            PER_TACTIC_TIMEOUT_MS)
    self.assertEqual(tree.nodes[2].closed, None)
    self.assertEqual(tree.nodes[1].closed, True)
    self.assertEqual(tree.nodes[0].closed, False)
    gen = mock_generator(('c', 1.0))
    prover_util.try_tactics(tree.nodes[2], 10, 0, 10, MOCK_PREMISE_SET, gen,
                            PER_TACTIC_TIMEOUT_MS)
    self.assertEqual(tree.nodes[2].closed, True)
    self.assertEqual(tree.nodes[0].closed, True)
    proof_search_tree.check_tree_consistency(tree)
    self.check_log_consistency()

  def test_apply_two_different_and_close_one(self):
    tree = self.tree
    root = tree.nodes[0]
    gen = mock_generator(('ay', 1.0), ('ax', 2.0))
    prover_util.try_tactics(root, 10, 0, 10, MOCK_PREMISE_SET, gen,
                            PER_TACTIC_TIMEOUT_MS)
    self.assertEqual(len(tree.nodes), 3)
    self.assertEqual(str(tree.nodes[1].goal.conclusion), 'cx')
    self.assertEqual(str(tree.nodes[2].goal.conclusion), 'cy')
    gen = mock_generator(('c', 1.0))
    prover_util.try_tactics(tree.nodes[1], 10, 0, 10, MOCK_PREMISE_SET, gen,
                            PER_TACTIC_TIMEOUT_MS)
    self.assertEqual(tree.nodes[2].closed, None)
    self.assertEqual(tree.nodes[1].closed, True)
    self.assertEqual(tree.nodes[0].closed, True)
    gen = mock_generator(('c', 1.0))
    prover_util.try_tactics(tree.nodes[2], 10, 0, 10, MOCK_PREMISE_SET, gen,
                            PER_TACTIC_TIMEOUT_MS)
    self.assertEqual(tree.nodes[2].closed, True)
    self.assertEqual(tree.nodes[0].closed, True)
    proof_search_tree.check_tree_consistency(tree)
    self.check_log_consistency()

  def test_shared_nodes_in_different_branches(self):
    tree = self.tree
    root = tree.nodes[0]
    gen = mock_generator(('ay', 1.0), ('ax', 2.0))
    prover_util.try_tactics(root, 10, 0, 10, MOCK_PREMISE_SET, gen,
                            PER_TACTIC_TIMEOUT_MS)
    self.assertEqual(len(tree.nodes), 3)
    self.assertEqual(str(tree.nodes[1].goal.conclusion), 'cx')
    self.assertEqual(str(tree.nodes[2].goal.conclusion), 'cy')
    gen = mock_generator(('rz', 1.0))
    prover_util.try_tactics(tree.nodes[1], 10, 0, 10, MOCK_PREMISE_SET, gen,
                            PER_TACTIC_TIMEOUT_MS)
    self.assertEqual(len(tree.nodes), 4)
    self.assertEqual(tree.nodes[3].closed, None)
    self.assertEqual(tree.nodes[2].closed, None)
    self.assertEqual(tree.nodes[1].closed, False)
    self.assertEqual(tree.nodes[0].closed, False)
    gen = mock_generator(('rz', 1.0))
    prover_util.try_tactics(tree.nodes[2], 10, 0, 10, MOCK_PREMISE_SET, gen,
                            PER_TACTIC_TIMEOUT_MS)
    self.assertEqual(len(tree.nodes), 4)
    self.assertEqual(len(tree.nodes[3].parents), 2)
    self.assertEqual(tree.nodes[3].closed, None)
    self.assertEqual(tree.nodes[2].closed, False)
    self.assertEqual(tree.nodes[1].closed, False)
    self.assertEqual(tree.nodes[0].closed, False)
    gen = mock_generator(('c', 1.0))
    prover_util.try_tactics(tree.nodes[3], 10, 0, 10, MOCK_PREMISE_SET, gen,
                            PER_TACTIC_TIMEOUT_MS)
    self.assertEqual(len(tree.nodes), 4)
    self.assertEqual(tree.nodes[3].closed, True)
    self.assertEqual(tree.nodes[2].closed, True)
    self.assertEqual(tree.nodes[1].closed, True)
    self.assertEqual(tree.nodes[0].closed, True)
    proof_search_tree.check_tree_consistency(tree)
    self.check_log_consistency()


if __name__ == '__main__':
  tf.test.main()
