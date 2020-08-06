"""Tests for deepmath.deephol.proof_search_tree."""

from typing import Optional, Text
from absl.testing import parameterized
import tensorflow.compat.v1 as tf
from deepmath.public import compare
from google.protobuf import text_format
from deepmath.deephol import deephol_pb2
from deepmath.deephol import io_util
from deepmath.deephol import proof_search_tree
from deepmath.deephol import tactic_utils
from deepmath.deephol import theorem_fingerprint
from deepmath.deephol import theorem_utils
from deepmath.proof_assistant import proof_assistant_pb2

PB_FORWARD_TEST = (
    ' prover_task {goals {conclusion: "(s_target)" tag: THEOREM}} nodes { '
    'root_goal: true goal { fingerprint: 1694572729299891902 conclusion: '
    '"(s0)" tag: GOAL}status: PROVED proofs { tactic: "REWRITE_RULE"  '
    'parameters { parameter_type: THEOREM_LIST  theorems { fingerprint: '
    '2319920848908768608 conclusion: "(theorem_a0)" tag: THEOREM} }  result: '
    'SUCCESS closed: true} } nodes {goal { conclusion: "(s1)" tag: GOAL}  '
    'status: PROVED  proofs { tactic: "REWRITE_RULE" parameters '
    '{parameter_type: THEOREM_LIST  theorems { conclusion: "(theorem_a1)" tag:'
    ' THEOREM} } result: SUCCESS closed: true}}nodes {goal { conclusion: '
    '"(s2)" tag: GOAL}  status: PROVED  proofs { tactic: "REWRITE_RULE" '
    'parameters {parameter_type: THEOREM_LIST  theorems { conclusion: '
    '"(theorem_a2)" tag: THEOREM} } result: SUCCESS closed: true}}nodes {goal '
    '{ conclusion: "(s3)" tag: GOAL}  status: PROVED  proofs { result: SUCCESS'
    ' closed: true}} ')


def _goal(conclusion: Text) -> proof_assistant_pb2.Theorem:
  return proof_assistant_pb2.Theorem(
      assumptions=[theorem_utils.assume_term('h', assumption_index=0)],
      conclusion=conclusion,
      tag=proof_assistant_pb2.Theorem.GOAL)


def _mock_tactic_application(
    parent: proof_search_tree.ProofSearchNode,
    child_id: int,
    conclusion_term: Optional[Text] = None,
    failed: bool = False) -> proof_search_tree.TacticApplication:
  conclusion = (
      conclusion_term if conclusion_term is not None else '%s%d' %
      (parent.goal.conclusion, child_id))
  goal = _goal(conclusion)

  application = proof_search_tree.TacticApplication(
      parent=parent,
      tactic='MOCK_TACTIC_TO_%s' % conclusion,
      parameters=[],
      score=42,
      rank=len(parent.failed_attempts) + len(parent.successful_attempts),
      index=(len(parent.failed_attempts)
             if failed else len(parent.successful_attempts)),
      result=(deephol_pb2.TacticApplication.ERROR
              if failed else deephol_pb2.TacticApplication.SUCCESS),
      error_message=None,
      time_spent=42,
      subgoals=[],
      closed=False,
      failed=failed)

  if not failed:
    subgoal_ref = proof_search_tree.SubGoalRef(
        tactic_application=application, subgoal_index=0)
    application.subgoals = [parent.tree.add_node(goal=goal, parent=subgoal_ref)]
    parent.successful_attempts.append(application)
  else:
    parent.failed_attempts.append(application)

  return application


class ProofSearchTreeTest(parameterized.TestCase):

  def setUp(self):
    super(ProofSearchTreeTest, self).setUp()
    goal = _goal('c0')
    self.tree = proof_search_tree.ProofSearchTree(
        proof_assistant_obj=None, goal=goal)
    self.node0 = self.tree.nodes[0]
    self.tac00 = _mock_tactic_application(self.node0, 0)
    _mock_tactic_application(self.node0, 424242, failed=True)
    self.node00 = self.tac00.subgoals[0]
    self.tac01 = _mock_tactic_application(self.node0, 1)
    self.node01 = self.tac01.subgoals[0]
    self.tac010 = _mock_tactic_application(self.node01, 0)
    self.node010 = self.tac010.subgoals[0]
    self.tac02 = _mock_tactic_application(self.node0, 2)
    self.node02 = self.tac02.subgoals[0]
    _mock_tactic_application(self.node02, 1337, failed=True)
    _mock_tactic_application(self.node02, 1338, failed=True)

  def _assert_application_to_proto_map(self, tree, application_to_proto_map):
    """Asserts map is populated and keys-values correspond to each other."""
    for node in tree.nodes:
      for application in node.successful_attempts + node.failed_attempts:
        assert application in application_to_proto_map
        application_proto = application_to_proto_map[application]
        self.assertEqual(application_proto.result, application.result)
        self.assertEqual(application_proto.error_message,
                         application.error_message)
        self.assertEqual(application_proto.time_spent, application.time_spent)
        self.assertEqual(application_proto.closed, application.closed)
        self.assertEqual(application_proto.score, application.score)
        self.assertCountEqual(application_proto.subgoals,
                              [sub.goal for sub in application.subgoals])

  def _assert_equal_proof_search_subtree(self, expected_node, node, visited):
    """Helper to assert equality of two subtrees rooted at input nodes."""
    self.assertLen(expected_node.parents, len(node.parents))
    self.assertEqual(expected_node.goal, node.goal)
    expected_failed_tactics = [
        application.tactic for application in expected_node.failed_attempts
    ]
    failed_tactics = [
        application.tactic for application in node.failed_attempts
    ]
    self.assertCountEqual(expected_failed_tactics, failed_tactics)
    expected_tactics = {
        applic.tactic: applic for applic in expected_node.successful_attempts
    }
    tactics = [application.tactic for application in node.successful_attempts]
    self.assertCountEqual(expected_tactics.keys(), tactics)

    visited.add(theorem_fingerprint.Fingerprint(node.goal))
    for application in node.successful_attempts:
      expected_application = expected_tactics[application.tactic]
      for i, subgoal_node in enumerate(application.subgoals):
        if theorem_fingerprint.Fingerprint(subgoal_node.goal) not in visited:
          self._assert_equal_proof_search_subtree(
              expected_application.subgoals[i], subgoal_node, visited)

  def _assert_equal_proof_search_tree(self, expected_tree, tree):
    """Helper to assert equality of two proof search trees."""
    self.assertLen(expected_tree.nodes, len(tree.nodes))
    self._assert_equal_proof_search_subtree(
        expected_tree.nodes[0], tree.nodes[0], visited=set())

  def _assert_proof_logs_equal_modulo_deduplication(self, original_log,
                                                    deduplicated_log):
    """Helper to assert that two logs are equal modulo deduplication.

    Deduplicated proof log has deduplicated proof log nodes with the same goal,
    but in the unique node it contains all the tactic applications coming
    from all of the corresponding duplicates in original proof log.

    Args:
      original_log: Original proof log.
      deduplicated_log: Proof log obtained by deserialization and serialization.
    """
    expected_nodes = {}
    for node in original_log.nodes:
      fingerprint = theorem_fingerprint.Fingerprint(node.goal)
      if fingerprint in expected_nodes:
        expected_nodes[fingerprint].append(node)
      else:
        expected_nodes[fingerprint] = [node]

    for node in deduplicated_log.nodes:
      fingerprint = theorem_fingerprint.Fingerprint(node.goal)
      assert fingerprint in expected_nodes, 'New node: %d' % fingerprint
      applications = {}
      for application in node.proofs:
        application_string = tactic_utils.tactic_application_to_string(
            application)
        assert application_string not in applications
        applications[application_string] = application

      for expected_node in expected_nodes[fingerprint]:
        self.assertEqual(expected_node.goal, node.goal)
        self.assertEqual(expected_node.status, node.status)
        self.assertEqual(expected_node.action_generation_time_millisec,
                         node.action_generation_time_millisec)
        self.assertEqual(expected_node.root_goal, node.root_goal)
        for expected_application in expected_node.proofs:
          expected_string = tactic_utils.tactic_application_to_string(
              expected_application)
          assert expected_string in applications
          application = applications[expected_string]
          self.assertEqual(expected_application.tactic, application.tactic)
          self.assertEqual(expected_application.result, application.result)
          self.assertEqual(expected_application.error_message,
                           application.error_message)
          self.assertEqual(expected_application.time_spent,
                           application.time_spent)
          self.assertEqual(expected_application.closed, application.closed)
          self.assertEqual(expected_application.score, application.score)
          self.assertCountEqual(expected_application.subgoals,
                                application.subgoals)

  def test_proof_state_from_proof_search_node_simple(self):
    """Proof state computation on a simple proof tree.

            c0
             |
       -------------
       |     |     |
      c00   c01   c02
             |
            c010
    """
    st = proof_search_tree.proof_state_from_proof_search_node(
        self.node0, history_bound=None)
    self.assertEqual(st.goal.conclusion, 'c0')
    self.assertEqual(st.previous_proof_state, None)

    st = proof_search_tree.proof_state_from_proof_search_node(
        self.node02, history_bound=None)
    self.assertEqual(st.goal.conclusion, 'c02')
    st = st.previous_proof_state
    self.assertEqual(st.goal.conclusion, 'c0')
    self.assertEqual(st.previous_proof_state, None)

    st = proof_search_tree.proof_state_from_proof_search_node(
        self.node02, history_bound=0)
    self.assertEqual(st.goal.conclusion, 'c02')
    self.assertEqual(st.previous_proof_state, None)

    st = proof_search_tree.proof_state_from_proof_search_node(
        self.node010, history_bound=None)
    self.assertEqual(st.goal.conclusion, 'c010')
    st = st.previous_proof_state
    self.assertEqual(st.goal.conclusion, 'c01')
    st = st.previous_proof_state
    self.assertEqual(st.goal.conclusion, 'c0')
    self.assertEqual(st.previous_proof_state, None)

    st = proof_search_tree.proof_state_from_proof_search_node(
        self.node010, history_bound=1)
    self.assertEqual(st.goal.conclusion, 'c010')
    st = st.previous_proof_state
    self.assertEqual(st.goal.conclusion, 'c01')
    self.assertEqual(st.previous_proof_state, None)

    st = proof_search_tree.proof_state_from_proof_search_node(
        self.node010, history_bound=0)
    self.assertEqual(st.goal.conclusion, 'c010')
    self.assertEqual(st.previous_proof_state, None)

    with self.assertRaises(AssertionError):
      proof_search_tree.proof_state_from_proof_search_node(
          self.node010, history_bound=-1)

  def test_proof_state_from_proof_search_node_multiple_subgoals(self):
    """Proof state computation when tactic applications have multiple subgoals.

            c0
             |
       ------------------------------
       |     |              --------|--------
      c00   c01            c02             c02x
             |        ------|------
            c010    c020  c020x  c020y
    """
    goal = _goal('c02x')
    subgoal_ref = proof_search_tree.SubGoalRef(
        tactic_application=self.tac02, subgoal_index=1)
    self.tac02.subgoals.append(
        self.tree.add_node(goal=goal, parent=subgoal_ref))
    node02x = self.tac02.subgoals[1]
    tac020 = _mock_tactic_application(self.node02, 0)
    goal = _goal('c020x')
    subgoal_ref = proof_search_tree.SubGoalRef(
        tactic_application=tac020, subgoal_index=1)
    tac020.subgoals.append(self.tree.add_node(goal=goal, parent=subgoal_ref))
    node020x = tac020.subgoals[1]
    goal = _goal('c020y')
    subgoal_ref = proof_search_tree.SubGoalRef(
        tactic_application=tac020, subgoal_index=2)
    tac020.subgoals.append(self.tree.add_node(goal=goal, parent=subgoal_ref))

    st = proof_search_tree.proof_state_from_proof_search_node(
        node02x, history_bound=None)
    self.assertEqual(st.goal.conclusion, 'c02x')
    st = st.previous_proof_state
    self.assertEqual(st.goal.conclusion, 'c0')
    self.assertEqual(st.previous_proof_state, None)

    st = proof_search_tree.proof_state_from_proof_search_node(
        node020x, history_bound=None)
    self.assertEqual(st.goal.conclusion, 'c020x')
    st = st.previous_proof_state
    self.assertEqual(st.goal.conclusion, 'c02')
    st = st.previous_proof_state
    self.assertEqual(st.goal.conclusion, 'c0')
    self.assertEqual(st.previous_proof_state, None)

  def test_proof_state_from_proof_search_node_collision(self):
    """Proof state computation when a node has multiple parents.

            c0
             |
       -------------
       |     |     |
      c00   c01   c02
       |     |
       -----c010
    """
    tac_clash = _mock_tactic_application(self.node00, 0, conclusion_term='c010')
    node_clash = tac_clash.subgoals[0]

    # First parent of 'c010' is 'c01', and hence 'c01' is considered
    # the previous proof state of 'c010'
    st = proof_search_tree.proof_state_from_proof_search_node(
        node_clash, history_bound=None)
    self.assertEqual(st.goal.conclusion, 'c010')
    st = st.previous_proof_state
    self.assertEqual(st.goal.conclusion, 'c01')
    st = st.previous_proof_state
    self.assertEqual(st.goal.conclusion, 'c0')
    self.assertEqual(st.previous_proof_state, None)

  def test_proof_state_from_proof_search_loop(self):
    """Proof state computation when there is a loop.

            c0
             |
       ---------------
       |     |       |
      c00   c01 <-  c02
             |   |
            c010-|
    """
    tac_loop = _mock_tactic_application(self.node010, 0, conclusion_term='c01')
    node_loop = tac_loop.subgoals[0]

    # First parent of 'c01' is 'c0', and hence 'c0' is considered
    # the previous proof state of 'c01'
    st = proof_search_tree.proof_state_from_proof_search_node(
        node_loop, history_bound=None)
    self.assertEqual(st.goal.conclusion, 'c01')
    st = st.previous_proof_state
    self.assertEqual(st.goal.conclusion, 'c0')
    self.assertEqual(st.previous_proof_state, None)

    st = proof_search_tree.proof_state_from_proof_search_node(
        self.node010, history_bound=None)
    self.assertEqual(st.goal.conclusion, 'c010')
    st = st.previous_proof_state
    self.assertEqual(st.goal.conclusion, 'c01')
    st = st.previous_proof_state
    self.assertEqual(st.goal.conclusion, 'c0')
    self.assertEqual(st.previous_proof_state, None)

  def test_proof_state_from_proof_search_root_loop(self):
    """Proof state computation when there is a loop with root node.

            c0  <------
             |        |
       -------------  |
       |     |     |  |
      c00   c01   c02-|
             |
            c010
    """
    tac_loop = _mock_tactic_application(self.node02, 0, conclusion_term='c0')
    node_loop = tac_loop.subgoals[0]

    # 'c0' is root so it has no previous proof state, even if it has a parent
    st = proof_search_tree.proof_state_from_proof_search_node(
        node_loop, history_bound=None)
    self.assertEqual(st.goal.conclusion, 'c0')
    self.assertEqual(st.previous_proof_state, None)

    st = proof_search_tree.proof_state_from_proof_search_node(
        self.node02, history_bound=None)
    self.assertEqual(st.goal.conclusion, 'c02')
    st = st.previous_proof_state
    self.assertEqual(st.goal.conclusion, 'c0')
    self.assertEqual(st.previous_proof_state, None)

  def test_proof_state_from_proof_search_disconnected_loop(self):
    """Proof state computation with loop as a disconnected component.

            c0
             |
       ---------------
       |             |
      c00   c01 <-  c02
             |   |
            c010-|

    We consider this case even though we do not create such proof graphs atm.
    """
    self.node01.parents = []
    tac_loop = _mock_tactic_application(self.node010, 0, conclusion_term='c01')
    node_loop = tac_loop.subgoals[0]

    # We track visited nodes and do not enter infinite loop 'c010' <-> 'c01'
    st = proof_search_tree.proof_state_from_proof_search_node(
        node_loop, history_bound=None)
    self.assertEqual(st.goal.conclusion, 'c01')
    st = st.previous_proof_state
    self.assertEqual(st.goal.conclusion, 'c010')
    self.assertEqual(st.previous_proof_state, None)

    st = proof_search_tree.proof_state_from_proof_search_node(
        self.node010, history_bound=None)
    self.assertEqual(st.goal.conclusion, 'c010')
    st = st.previous_proof_state
    self.assertEqual(st.goal.conclusion, 'c01')
    self.assertEqual(st.previous_proof_state, None)

  def test_tree_to_proto_and_back_simple(self):
    """Simple proof search tree - serialize it and deserialize it back.

            c0
             |
       -------------
       |     |     |
      c00   c01   c02
             |
            c010
    """
    proof_log = self.tree.to_proto()
    self.assertLen(proof_log.nodes, 5)
    tree, application_to_proto_map = proof_search_tree.tree_from_proof_log(
        proof_log)
    self._assert_application_to_proto_map(tree, application_to_proto_map)
    self._assert_equal_proof_search_tree(self.tree, tree)

  def test_tree_to_proto_and_back_multiple_subgoals_collision(self):
    """Tree multi subgoals and collision - serialize it and deserialize it back.

            c0
             |
       ------------------------------
       |     |              --------|--------
      c00   c01            c02             c02x
       |     |        ------|------
       |----c010    c020  c020x  c020y
    """
    goal = _goal('c02x')
    subgoal_ref = proof_search_tree.SubGoalRef(
        tactic_application=self.tac02, subgoal_index=1)
    self.tac02.subgoals.append(
        self.tree.add_node(goal=goal, parent=subgoal_ref))
    tac020 = _mock_tactic_application(self.node02, 0)
    goal = _goal('c020x')
    subgoal_ref = proof_search_tree.SubGoalRef(
        tactic_application=tac020, subgoal_index=1)
    tac020.subgoals.append(self.tree.add_node(goal=goal, parent=subgoal_ref))
    goal = _goal('c020y')
    subgoal_ref = proof_search_tree.SubGoalRef(
        tactic_application=tac020, subgoal_index=2)
    tac020.subgoals.append(self.tree.add_node(goal=goal, parent=subgoal_ref))
    _mock_tactic_application(self.node00, 0, conclusion_term='c010')

    proof_log = self.tree.to_proto()
    self.assertLen(proof_log.nodes, 9)
    tree, application_to_proto_map = proof_search_tree.tree_from_proof_log(
        proof_log)
    self._assert_application_to_proto_map(tree, application_to_proto_map)
    self._assert_equal_proof_search_tree(self.tree, tree)

  def test_tree_to_proto_and_back_loops(self):
    """Tree with loops - serialize it and deserialize it back.

            c0 <---------
             |          |
       ---------------  |
       |     |       |  |
      c00   c01 <-  c02-|
             |   |
            c010-|
    """
    _mock_tactic_application(self.node010, 0, conclusion_term='c01')
    _mock_tactic_application(self.node02, 0, conclusion_term='c0')

    proof_log = self.tree.to_proto()
    self.assertLen(proof_log.nodes, 5)
    tree, application_to_proto_map = proof_search_tree.tree_from_proof_log(
        proof_log)
    self._assert_application_to_proto_map(tree, application_to_proto_map)
    self._assert_equal_proof_search_tree(self.tree, tree)

  @parameterized.named_parameters(
      ('faked_proof_log',
       'theorem_in_database { fingerprint: 3315754850497711862   conclusion: '
       '"(a b c)" tag: THEOREM } nodes { goal { fingerprint: '
       '3315754850497711862   conclusion: "(a b c)" tag: GOAL } status: '
       'PROVED proofs { tactic: "X_GEN_TAC" parameters { parameter_type: TERM '
       'term: "(v x y)"} subgoals { fingerprint: 2590611553234554735 '
       'conclusion: "(a a a)" tag: GOAL} result: SUCCESS closed: true}} nodes '
       '{ goal { fingerprint: 2590611553234554735   conclusion: "(a a a)" tag: '
       'GOAL} status: PROVED proofs { tactic: "X_GEN_TAC" parameters { '
       'parameter_type: TERM term: "(v x y)"} result: SUCCESS closed: true}}'))
  def test_proof_log_to_tree_and_back(self, log_string):
    """Deserialize a human proof log to tree and serialize it back."""
    expected_proof_log = text_format.Parse(log_string, deephol_pb2.ProofLog())
    io_util.fix_legacy_proof_log(expected_proof_log)  # Marks first node as root
    self.assertLen(expected_proof_log.nodes, 2)
    tree, application_to_proto_map = proof_search_tree.tree_from_proof_log(
        expected_proof_log)
    self._assert_application_to_proto_map(tree, application_to_proto_map)
    self.assertLen(tree.nodes, 2)  # One node deduplicated from the proof log
    proof_log = tree.to_proto()
    self._assert_proof_logs_equal_modulo_deduplication(
        original_log=expected_proof_log, deduplicated_log=proof_log)

  def test_targeted_provertask(self):
    """Test a case where there is a prover_task with and without a target."""
    example_proof_log = text_format.Parse(PB_FORWARD_TEST,
                                          deephol_pb2.ProofLog())
    nodes0 = example_proof_log.nodes[0]
    nodes1 = example_proof_log.nodes[1]
    self.assertTrue(nodes0.root_goal)
    self.assertFalse(nodes1.root_goal)

    prover_task = example_proof_log.prover_task

    tree1 = proof_search_tree.ProofSearchTree(
        proof_assistant_obj=None, goal=nodes0.goal, prover_task=prover_task)

    self.assertEmpty(tree1.targets)
    self.assertFalse(any(node.closed for node in tree1.nodes))

    prover_task.targets.append(nodes1.goal)
    tree1 = proof_search_tree.ProofSearchTree(
        proof_assistant_obj=None, goal=nodes0.goal, prover_task=prover_task)
    self.assertLen(tree1.targets, 1)

  def test_tree_from_proof_log_and_back_with_provertask(self):
    """Test that we get the tree out of the whole proof log."""

    example_proof_log = text_format.Parse(PB_FORWARD_TEST,
                                          deephol_pb2.ProofLog())
    tree_fwd, application_to_proto_map = proof_search_tree.tree_from_proof_log(
        example_proof_log)

    self._assert_application_to_proto_map(tree_fwd, application_to_proto_map)
    self.assertLen(example_proof_log.nodes, 4)

    prover_task = 'prover_task {goals {conclusion: "(s_target)" tag: THEOREM}}'
    prover_task_pb = text_format.Parse(prover_task, deephol_pb2.ProofLog())

    compare.assertProto2Equal(self, tree_fwd.prover_task,
                              prover_task_pb.prover_task)

    proof_log = tree_fwd.to_proto()

    self._assert_proof_logs_equal_modulo_deduplication(
        original_log=example_proof_log, deduplicated_log=proof_log)

  def test_tactic_application_examples(self):
    """Test extracting (goal, tactic) examples from proof search tree."""
    proof_log = text_format.Parse(PB_FORWARD_TEST, deephol_pb2.ProofLog())
    tree, _ = proof_search_tree.tree_from_proof_log(proof_log)
    examples = list(proof_search_tree.tactic_application_examples(tree))

    # This tree is a chain, thus we should output an example for each node.
    self.assertLen(examples, len(tree.nodes))

    for _, target in examples:
      self.assertIsNone(target)

  def test_her_examples(self):
    """Test extracting (goal, tactic) examples from proof search tree."""
    proof_log = text_format.Parse(PB_FORWARD_TEST, deephol_pb2.ProofLog())
    tree, _ = proof_search_tree.tree_from_proof_log(proof_log)

    # Check monotonicity and size bound.
    prev = 0
    for k in range(5):
      samples = proof_search_tree.her_examples(tree, sources_per_target=k)
      size = len(list(samples))
      self.assertLessEqual(prev, size)
      self.assertLessEqual(size, k * (len(tree.nodes) - 1))
      prev = size

    for _, target in samples:
      self.assertIsNotNone(target)


if __name__ == '__main__':
  tf.test.main()
