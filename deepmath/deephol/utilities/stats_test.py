"""Tests for deepmath.deephol.utilities.stats."""

from __future__ import absolute_import
from __future__ import division
# Import Type Annotations
from __future__ import print_function

import tensorflow as tf
from typing import List

from deepmath.deephol import deephol_pb2
from deepmath.deephol import theorem_fingerprint
from deepmath.deephol.utilities import deephol_stat_pb2
from deepmath.deephol.utilities import stats
from deepmath.proof_assistant import proof_assistant_pb2


def make_theorem(i: int, theorem: bool = True) -> proof_assistant_pb2.Theorem:
  """Make a theorem or goal with content generated from the integer.

  Args:
    i: Index to uniquely identify the theorem.
    theorem: If true, then it will be tagged as a theorem, otherwise a goal.

  Returns:
    The generated proof_assistant_pb2.Theorem object.
  """
  if theorem:
    tag = proof_assistant_pb2.Theorem.THEOREM
  else:
    tag = proof_assistant_pb2.Theorem.GOAL
  return proof_assistant_pb2.Theorem(
      hypotheses=['h'], conclusion='c%d' % i, tag=tag)


def add_node(log: deephol_pb2.ProofLog,
             proofs: List[List[int]],
             proved: bool = True,
             theorem: bool = False,
             prediction_time=37):
  """Add a new mock node to the proof log.

  Args:
    log: The proof log to be extended.
    proofs: Lists of lists of node indices representing the proof attempts
      (TacticApplications).
    proved: If true, the node is to be marked as true.
    theorem: If true, the node is marked as a theorem, otherwise a subgoal.
    prediction_time: The time the action generator used to generate predictions.
  """
  node_index = len(log.nodes)
  status = proof_assistant_pb2.Theorem.UNKNOWN
  if proved:
    status = deephol_pb2.ProofNode.PROVED

  def mk_tactic_application(proof):
    return deephol_pb2.TacticApplication(
        subgoals=[make_theorem(i) for i in proof],
        closed=True,
        result=deephol_pb2.TacticApplication.SUCCESS)

  proofs = [mk_tactic_application(proof) for proof in proofs]
  log.nodes.add(
      goal=make_theorem(node_index, theorem=theorem),
      status=status,
      proofs=proofs,
      action_generation_time_millisec=prediction_time)


def new_log(num_proofs: int = 1, time_spent: int = 10000):
  """Create a new proof log."""
  return deephol_pb2.ProofLog(
      num_proofs=num_proofs,
      prover_options=deephol_pb2.ProverOptions(),
      time_spent=time_spent)


class StatsTest(tf.test.TestCase):

  def test_proof_log_stats_empty(self):
    proof_log = deephol_pb2.ProofLog()
    s = stats.proof_log_stats(proof_log)
    self.assertEqual(s.num_theorems_attempted, 0)
    self.assertEqual(s.num_theorems_proved, 0)
    self.assertEqual(s.num_theorems_with_bad_proof, 0)
    self.assertEqual(s.num_nodes, 0)
    self.assertEqual(s.total_prediction_time, 0)
    self.assertEmpty(s.node_prediction_time_histogram.h)
    self.assertEqual(s.node_prediction_time_histogram.h[42], 0)
    self.assertEmpty(s.tapp_stat.time_spent_per_tapp_result)
    self.assertEmpty(s.tapp_stat.time_spent_per_tactic)
    self.assertEmpty(s.tapp_stat.total_tactic_applications_per_tactic)
    self.assertEmpty(s.tapp_stat.successful_tactic_applications_per_tactic)

  def test_proof_log_stats_root_only(self):
    proof_log = new_log(num_proofs=0)
    add_node(proof_log, [], False, True)
    s = stats.proof_log_stats(proof_log)
    self.assertEqual(s.num_theorems_attempted, 1)
    self.assertEqual(s.num_theorems_proved, 0)
    self.assertEqual(s.num_theorems_with_bad_proof, 0)
    self.assertEqual(s.num_nodes, 1)
    self.assertEqual(s.time_spent_milliseconds, 10000)
    self.assertEqual(s.theorem_fingerprint,
                     theorem_fingerprint.Fingerprint(proof_log.nodes[0].goal))

  def test_proof_with_loop(self):
    proof_log = new_log(num_proofs=1)
    add_node(proof_log, [], True)
    add_node(proof_log, [[2]], True, True)
    add_node(proof_log, [[3]], True)
    add_node(proof_log, [[2], [5]], True)
    add_node(proof_log, [[]], True)
    add_node(proof_log, [[]], True)
    s = stats.proof_log_stats(proof_log)
    self.assertEqual(s.num_theorems_attempted, 1)
    self.assertEqual(s.num_theorems_proved, 1)
    self.assertEqual(s.num_theorems_with_bad_proof, 0)
    self.assertEqual(s.num_nodes, 6)
    self.assertEqual(list(s.closed_node_indices), list(range(6)))
    self.assertEqual(s.time_spent_milliseconds, 10000)
    self.assertEqual(s.theorem_fingerprint,
                     theorem_fingerprint.Fingerprint(proof_log.nodes[1].goal))

  def test_proof_merge_stat_with_empty_stat(self):
    aggregate_stat = deephol_stat_pb2.ProofAggregateStat()
    proof_stat = deephol_stat_pb2.ProofStat(
        num_theorems_attempted=4,
        num_theorems_proved=2,
        num_theorems_with_bad_proof=1,
        num_nodes=10,
        reduced_node_indices=[1, 2, 3],
        closed_node_indices=[1, 2, 3, 4],
        time_spent_milliseconds=10000)
    stats.merge_stat(aggregate_stat, proof_stat)
    self.assertEqual(aggregate_stat.num_theorems_attempted, 4)
    self.assertEqual(aggregate_stat.num_theorems_proved, 2)
    self.assertEqual(aggregate_stat.num_theorems_with_bad_proof, 1)
    self.assertEqual(aggregate_stat.num_nodes, 10)
    self.assertEqual(aggregate_stat.num_reduced_nodes, 3)
    self.assertEqual(aggregate_stat.num_closed_nodes, 4)
    self.assertEqual(aggregate_stat.time_spent_milliseconds, 10000)
    self.assertNotEmpty(aggregate_stat.num_reduced_nodes_distribution)
    # 10000 ~~ 2**13
    self.assertEqual(aggregate_stat.proof_time_histogram.h[13], 1)
    self.assertEqual(aggregate_stat.proof_time_histogram_proved.h[13], 1)
    self.assertEmpty(aggregate_stat.proof_time_histogram_failed.h)
    self.assertNotEmpty(aggregate_stat.proof_prediction_time_histogram.h)

  def test_proof_merge_stat_with_stat(self):
    aggregate_stat = deephol_stat_pb2.ProofAggregateStat(
        num_theorems_attempted=5,
        num_theorems_proved=6,
        num_theorems_with_bad_proof=7,
        num_nodes=20,
        num_reduced_nodes=30,
        num_closed_nodes=40,
        time_spent_milliseconds=30000)
    proof_stat = deephol_stat_pb2.ProofStat(
        num_theorems_attempted=4,
        num_theorems_proved=2,
        num_theorems_with_bad_proof=1,
        num_nodes=10,
        reduced_node_indices=[1, 2, 3],
        closed_node_indices=[1, 2, 3, 4],
        time_spent_milliseconds=10000)
    stats.merge_stat(aggregate_stat, proof_stat)
    self.assertEqual(aggregate_stat.num_theorems_attempted, 9)
    self.assertEqual(aggregate_stat.num_theorems_proved, 8)
    self.assertEqual(aggregate_stat.num_theorems_with_bad_proof, 8)
    self.assertEqual(aggregate_stat.num_nodes, 30)
    self.assertEqual(aggregate_stat.num_reduced_nodes, 33)
    self.assertEqual(aggregate_stat.num_closed_nodes, 44)
    self.assertEqual(aggregate_stat.time_spent_milliseconds, 40000)

  def test_merge_aggregate_stat(self):
    stat1 = deephol_stat_pb2.ProofAggregateStat(
        num_theorems_attempted=5,
        num_theorems_proved=6,
        num_theorems_with_bad_proof=7,
        num_nodes=20,
        num_reduced_nodes=30,
        num_closed_nodes=40,
        time_spent_milliseconds=30000)
    stat2 = deephol_stat_pb2.ProofAggregateStat(
        num_theorems_attempted=6,
        num_theorems_proved=7,
        num_theorems_with_bad_proof=8,
        num_nodes=21,
        num_reduced_nodes=31,
        num_closed_nodes=41,
        time_spent_milliseconds=30001)
    stats.merge_aggregate_stat(stat1, stat2)
    self.assertEqual(stat1.num_theorems_attempted, 11)
    self.assertEqual(stat1.num_theorems_proved, 13)
    self.assertEqual(stat1.num_theorems_with_bad_proof, 15)
    self.assertEqual(stat1.num_nodes, 41)
    self.assertEqual(stat1.num_reduced_nodes, 61)
    self.assertEqual(stat1.num_closed_nodes, 81)
    self.assertEqual(stat1.time_spent_milliseconds, 60001)

  def test_aggregate_stats(self):
    stat1 = deephol_stat_pb2.ProofStat(
        num_theorems_attempted=4,
        num_theorems_proved=2,
        num_theorems_with_bad_proof=1,
        num_nodes=10,
        reduced_node_indices=[1, 2, 3],
        closed_node_indices=[1, 2, 3, 4],
        time_spent_milliseconds=10000)
    stat2 = deephol_stat_pb2.ProofStat(
        num_theorems_attempted=5,
        num_theorems_proved=3,
        num_theorems_with_bad_proof=2,
        num_nodes=11,
        reduced_node_indices=[1, 2, 3, 4],
        closed_node_indices=[1, 2, 3, 4, 5],
        time_spent_milliseconds=10001)
    aggregate_stat = stats.aggregate_stats([stat1, stat2])
    self.assertEqual(aggregate_stat.num_theorems_attempted, 9)
    self.assertEqual(aggregate_stat.num_theorems_proved, 5)
    self.assertEqual(aggregate_stat.num_theorems_with_bad_proof, 3)
    self.assertEqual(aggregate_stat.num_nodes, 21)
    self.assertEqual(aggregate_stat.num_reduced_nodes, 7)
    self.assertEqual(aggregate_stat.num_closed_nodes, 9)
    self.assertEqual(aggregate_stat.time_spent_milliseconds, 20001)

  def test_stat_to_string(self):
    stat = deephol_stat_pb2.ProofStat(
        num_theorems_attempted=4,
        num_theorems_proved=2,
        num_theorems_with_bad_proof=1,
        num_nodes=10,
        reduced_node_indices=[1, 2, 3],
        closed_node_indices=[1, 2, 3, 4],
        time_spent_milliseconds=10000)
    isinstance(stats.stat_to_string(stat), str)

  def test_aggregate_stat_to_string(self):
    s = deephol_stat_pb2.ProofAggregateStat(
        num_theorems_attempted=5,
        num_theorems_proved=6,
        num_theorems_with_bad_proof=7,
        num_nodes=20,
        num_reduced_nodes=30,
        num_closed_nodes=40,
        time_spent_milliseconds=30000)
    isinstance(stats.aggregate_stat_to_string(s), str)


if __name__ == '__main__':
  tf.test.main()
