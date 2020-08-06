"""Tests for deepmath.deephol.proof_analysis."""
import tensorflow.compat.v1 as tf
from deepmath.deephol.utilities import proof_analysis
from deepmath.deephol.utilities import proof_test_util


class ProofAnalysisTest(tf.test.TestCase):

  def test_empty_log_reasons(self):
    proof_log = proof_test_util.new_log(num_proofs=0)
    self.assertEqual(proof_analysis.find_reasons(proof_log), ([], []))

  def test_root_no_proof_reasons(self):
    proof_log = proof_test_util.new_log(num_proofs=0)
    proof_test_util.add_node(proof_log, [], False, True)
    self.assertEqual(proof_analysis.find_reasons(proof_log), ([], []))

  def test_root_not_marked(self):
    proof_log = proof_test_util.new_log(num_proofs=0)
    proof_test_util.add_node(proof_log, [], False, False)
    self.assertEqual(proof_analysis.find_reasons(proof_log), ([], []))

  def test_root_no_reasons(self):
    proof_log = proof_test_util.new_log(num_proofs=0)
    proof_test_util.add_node(proof_log, [], True, True)
    self.assertIsNone(proof_analysis.find_reasons(proof_log))

  def test_multi_root_has_simple_proof(self):
    proof_log = proof_test_util.new_log(num_proofs=1)
    proof_test_util.add_node(proof_log, [[2]], True, True)
    proof_test_util.add_node(proof_log, [[2]], True, True)
    proof_test_util.add_node(proof_log, [[]], True)
    reasons, nodes = proof_analysis.find_reasons(proof_log)
    self.assertEqual(nodes, [0, 1, 2])
    self.assertEqual(reasons, [(0, 0, [2]), (1, 0, [2]), (2, 0, [])])

  def test_multi_root_reorder_has_simple_proof(self):
    proof_log = proof_test_util.new_log(num_proofs=1)
    proof_test_util.add_node(proof_log, [[]], True)
    proof_test_util.add_node(proof_log, [[0]], True, True)
    proof_test_util.add_node(proof_log, [[0]], True, True)
    reasons, nodes = proof_analysis.find_reasons(proof_log)
    self.assertEqual(nodes, [1, 2, 0])
    self.assertEqual(reasons, [(1, 0, [0]), (2, 0, [0]), (0, 0, [])])

  def test_multi_root_reorder2_has_simple_proof(self):
    proof_log = proof_test_util.new_log(num_proofs=1)
    proof_test_util.add_node(proof_log, [[]], True)
    proof_test_util.add_node(proof_log, [[0]], True, True)
    proof_test_util.add_node(proof_log, [[1]], True, True)
    reasons, nodes = proof_analysis.find_reasons(proof_log)
    self.assertEqual(nodes, [1, 2, 0])
    self.assertEqual(reasons, [(1, 0, [0]), (2, 0, [1]), (0, 0, [])])

  def test_root_invalid_simple_proof(self):
    proof_log = proof_test_util.new_log(num_proofs=0)
    proof_test_util.add_node(proof_log, [[1]], True, True)
    proof_test_util.add_node(proof_log, [], False)
    self.assertIsNone(proof_analysis.find_reasons(proof_log))

  def test_root_is_leaf(self):
    proof_log = proof_test_util.new_log(num_proofs=1)
    proof_test_util.add_node(proof_log, [[]], True, True)
    reasons, nodes = proof_analysis.find_reasons(proof_log)
    self.assertEqual(nodes, [0])
    self.assertEqual(reasons, [(0, 0, [])])

  def test_root_has_simple_proof(self):
    proof_log = proof_test_util.new_log(num_proofs=1)
    proof_test_util.add_node(proof_log, [[1]], True, True)
    proof_test_util.add_node(proof_log, [[]], True)
    reasons, nodes = proof_analysis.find_reasons(proof_log)
    self.assertEqual(nodes, [0, 1])
    self.assertEqual(reasons, [(0, 0, [1]), (1, 0, [])])

  def test_root_has_simple_proof_order2(self):
    proof_log = proof_test_util.new_log(num_proofs=1)
    proof_test_util.add_node(proof_log, [[]], True)
    proof_test_util.add_node(proof_log, [[0]], True, True)
    reasons, nodes = proof_analysis.find_reasons(proof_log)
    self.assertEqual(nodes, [1, 0])
    self.assertEqual(reasons, [(1, 0, [0]), (0, 0, [])])

  def test_root_has_chain_ignores_unclosed(self):
    proof_log = proof_test_util.new_log(num_proofs=1)
    proof_test_util.add_node(proof_log, [[1]], True, True)
    proof_test_util.add_node(proof_log, [[2]], True)
    proof_test_util.add_node(proof_log, [[5], [4]], True)
    proof_test_util.add_node(proof_log, [[]], True)
    proof_test_util.add_node(proof_log, [[]], True)
    proof_test_util.add_node(proof_log, [], True)
    reasons, nodes = proof_analysis.find_reasons(proof_log)
    self.assertEqual(nodes, [0, 1, 2, 4])
    self.assertEqual(reasons, [(0, 0, [1]), (1, 0, [2]), (2, 1, [4]),
                               (4, 0, [])])

  def test_root_has_chain_ignores_loop(self):
    proof_log = proof_test_util.new_log(num_proofs=1)
    proof_test_util.add_node(proof_log, [[1]], True, True)
    proof_test_util.add_node(proof_log, [[2]], True)
    proof_test_util.add_node(proof_log, [[1], [4]], True)
    proof_test_util.add_node(proof_log, [[]], True)
    proof_test_util.add_node(proof_log, [[]], True)
    proof_test_util.add_node(proof_log, [], True)
    reasons, nodes = proof_analysis.find_reasons(proof_log)
    self.assertEqual(nodes, [0, 1, 2, 4])
    self.assertEqual(reasons, [(0, 0, [1]), (1, 0, [2]), (2, 1, [4]),
                               (4, 0, [])])

  def test_root_has_chain_ignores_loop_order2(self):
    proof_log = proof_test_util.new_log(num_proofs=1)
    proof_test_util.add_node(proof_log, [], True)
    proof_test_util.add_node(proof_log, [[2]], True, True)
    proof_test_util.add_node(proof_log, [[3]], True)
    proof_test_util.add_node(proof_log, [[2], [5]], True)
    proof_test_util.add_node(proof_log, [[]], True)
    proof_test_util.add_node(proof_log, [[]], True)
    reasons, nodes = proof_analysis.find_reasons(proof_log)
    self.assertEqual(nodes, [1, 2, 3, 5])
    self.assertEqual(reasons, [(1, 0, [2]), (2, 0, [3]), (3, 1, [5]),
                               (5, 0, [])])

  def test_extract_proof(self):
    proof_log = proof_test_util.new_log(num_proofs=1)
    proof_test_util.add_node(proof_log, [], True)
    proof_test_util.add_node(proof_log, [[2]], True, True)
    proof_test_util.add_node(proof_log, [[3]], True)
    proof_test_util.add_node(proof_log, [[2], [5]], True)
    proof_test_util.add_node(proof_log, [[]], True)
    proof_test_util.add_node(proof_log, [[]], True)
    output_log = proof_analysis.extract_proof(proof_log)
    self.assertEqual(len(output_log.nodes), 4)
    for i, j in enumerate([1, 2, 3, 5]):
      self.assertEqual(output_log.nodes[i].goal.conclusion,
                       proof_log.nodes[j].goal.conclusion)
    self.assertFalse(
        proof_analysis.check_extracted_proof_assumptions(output_log))

  def test_extract_proof_removes_cycles(self):
    proof_log = proof_test_util.new_log(num_proofs=1)
    proof_test_util.add_node(proof_log, [[2], [1]], True, True)
    proof_test_util.add_node(proof_log, [[3], [0], [2]], True)
    proof_test_util.add_node(proof_log, [[0], [2]], True)
    proof_test_util.add_node(proof_log, [[]], True)
    output_log = proof_analysis.extract_proof(proof_log)
    self.assertEqual(len(output_log.nodes), 3)
    for i, j in enumerate([0, 1, 3]):
      self.assertEqual(output_log.nodes[i].goal.conclusion,
                       proof_log.nodes[j].goal.conclusion)
    self.assertFalse(
        proof_analysis.check_extracted_proof_assumptions(output_log))

  def test_extract_proof_preserves_root_goal(self):
    proof_log = proof_test_util.new_log(num_proofs=1)
    proof_test_util.add_node(proof_log, [[]], proved=True)
    proof_test_util.add_node(proof_log, [[0]], proved=True, root_goal=True)
    output_log = proof_analysis.extract_proof(proof_log)
    self.assertEqual(len(output_log.nodes), 2)
    for i, j in enumerate([1, 0]):
      self.assertEqual(output_log.nodes[i].goal.conclusion,
                       proof_log.nodes[j].goal.conclusion)
    self.assertTrue(output_log.nodes[0].root_goal)
    self.assertFalse(
        proof_analysis.check_extracted_proof_assumptions(output_log))

  def test_extract_proof_no_root_goal_raises(self):
    proof_log = proof_test_util.new_log(num_proofs=1)
    proof_test_util.add_node(proof_log, [[]], proved=True)
    # This should be the root goal; but is a common legacy misconfiguration.
    proof_test_util.add_node(proof_log, [[0]], proved=True, root_goal=False)
    self.assertIsNone(proof_analysis.extract_proof(proof_log))


if __name__ == '__main__':
  tf.test.main()
