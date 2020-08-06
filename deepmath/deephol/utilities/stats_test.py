"""Tests for deepmath.deephol.utilities.stats."""
import os
import tensorflow.compat.v1 as tf
from deepmath.deephol import deephol_pb2
from deepmath.deephol import theorem_fingerprint
from deepmath.public import recordio_util
from deepmath.deephol.utilities import deephol_stat_pb2
from deepmath.deephol.utilities import proof_test_util
from deepmath.deephol.utilities import stats
from deepmath.proof_assistant import proof_assistant_pb2


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
    proof_log = proof_test_util.new_log(num_proofs=0)
    proof_test_util.add_node(proof_log, [], False, True)
    s = stats.proof_log_stats(proof_log)
    self.assertEqual(s.num_theorems_attempted, 1)
    self.assertEqual(s.num_theorems_proved, 0)
    self.assertEqual(s.num_theorems_with_bad_proof, 0)
    self.assertEqual(s.num_nodes, 1)
    self.assertEqual(s.time_spent_milliseconds, 10000)
    fp = theorem_fingerprint.Fingerprint(proof_log.nodes[0].goal)
    self.assertSequenceEqual(list(s.attempted_theorems.items()), [(fp, 1)])
    self.assertEmpty(list(s.proven_theorems.keys()))

  def test_proof_with_loop(self):
    proof_log = proof_test_util.new_log(num_proofs=1)
    proof_test_util.add_node(proof_log, [], True)
    proof_test_util.add_node(proof_log, [[2]], True, True)
    proof_test_util.add_node(proof_log, [[3]], True)
    proof_test_util.add_node(proof_log, [[2], [5]], True)
    proof_test_util.add_node(proof_log, [[]], True)
    proof_test_util.add_node(proof_log, [[]], True)
    s = stats.proof_log_stats(proof_log)
    self.assertEqual(s.num_theorems_attempted, 1)
    self.assertEqual(s.num_theorems_proved, 1)
    self.assertEqual(s.num_theorems_with_bad_proof, 0)
    self.assertEqual(s.num_nodes, 6)
    self.assertEqual(list(s.closed_node_indices), list(range(6)))
    self.assertEqual(s.time_spent_milliseconds, 10000)
    self.assertSequenceEqual(
        list(s.proven_theorems.keys()),
        [theorem_fingerprint.Fingerprint(proof_log.nodes[1].goal)])

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

  def test_library_tag_aggregate_stat_to_string(self):
    s = deephol_stat_pb2.ProofAggregateStat(
        num_theorems_attempted=5,
        num_theorems_proved=6,
        num_theorems_with_bad_proof=7,
        num_nodes=20,
        num_reduced_nodes=30,
        num_closed_nodes=40,
        time_spent_milliseconds=30000)

    ltpas = deephol_stat_pb2.LabeledAggregateStats(labeled_stats={
        '<overall_statistics>': s,
        'arbitraryTag': s
    })
    isinstance(stats.labeled_agg_stat_to_string(ltpas), str)

  def test_search_stats(self):
    search_stats = deephol_pb2.SearchStatistics(
        search_depth=2, total_expansions=42, mcts_path_values=[1.0])
    proof_log = deephol_pb2.ProofLog(search_statistics=search_stats)
    stat1 = stats.proof_log_stats(proof_log)
    aggr1 = deephol_stat_pb2.LabeledAggregateStats()
    stats.merge_into_labeled_aggregate_stat(aggr1, stat1)

    stat2 = stats.proof_log_stats(proof_log)
    aggr2 = deephol_stat_pb2.LabeledAggregateStats()
    stats.merge_into_labeled_aggregate_stat(aggr2, stat2)

    stats.merge_labeled_stats(aggr1, aggr2)
    merged_stats = aggr1.labeled_stats[stats.OVERALL_LABEL]
    self.assertEqual(merged_stats.search_statistics.search_depths.total_value,
                     4.0)
    self.assertEqual(
        merged_stats.search_statistics.total_expansions.total_value, 84.0)
    self.assertEqual(
        merged_stats.search_statistics.mcts_root_values.total_value, 2.0)

  def test_mean_mcts_value_diffs(self):
    node = deephol_pb2.ProofNode(
        goal=proof_assistant_pb2.Theorem(conclusion='asdf'), root_goal=True)
    search_stats = deephol_pb2.SearchStatistics(
        mcts_path_values_difference=[.3])
    proof_log = deephol_pb2.ProofLog(
        nodes=[node], search_statistics=search_stats)
    stat1 = stats.proof_log_stats(proof_log)
    aggr1 = deephol_stat_pb2.LabeledAggregateStats()
    stats.merge_into_labeled_aggregate_stat(aggr1, stat1)
    self.assertAlmostEqual(
        list(aggr1.labeled_stats[
            stats.OVERALL_LABEL].mean_mcts_value_diffs.items())[0][1], .3)

    stat2 = stats.proof_log_stats(proof_log)
    aggr2 = deephol_stat_pb2.LabeledAggregateStats()
    stats.merge_into_labeled_aggregate_stat(aggr2, stat2)

    stats.merge_labeled_stats(aggr1, aggr2)
    merged_stats = aggr1.labeled_stats[stats.OVERALL_LABEL]
    self.assertGreater(
        list(merged_stats.mean_mcts_value_diffs.items())[0][1], .4)

  def test_mean_mcts_value_diffs_write_and_read(self):
    """Test if field mean_mcts_value_diffs is preserved through write and read."""
    node = deephol_pb2.ProofNode(
        goal=proof_assistant_pb2.Theorem(conclusion='asdf'), root_goal=True)
    search_stats = deephol_pb2.SearchStatistics(
        mcts_path_values_difference=[.3])
    proof_log = deephol_pb2.ProofLog(
        nodes=[node], search_statistics=search_stats)
    stat1 = stats.proof_log_stats(proof_log)
    aggr1 = deephol_stat_pb2.LabeledAggregateStats()
    stats.merge_into_labeled_aggregate_stat(aggr1, stat1)

    filename = os.path.join(self.create_tempdir().full_path, 'tmp_stats')
    recordio_util.write_protos_to_recordio(filename, [aggr1])
    stats_from_disk = list(
        recordio_util.read_protos_from_recordio(
            filename, deephol_stat_pb2.LabeledAggregateStats))[0]
    self.assertLen(
        stats_from_disk.labeled_stats[
            stats.OVERALL_LABEL].mean_mcts_value_diffs.items(), 1)
    self.assertAlmostEqual(
        list(stats_from_disk.labeled_stats[
            stats.OVERALL_LABEL].mean_mcts_value_diffs.items())[0][1], .3)


if __name__ == '__main__':
  tf.test.main()
