"""Statistics about various DeepHOL objects, especially ProofLogs.

In the first iteration this library can be used for generating statistics
over ProofLogs.
"""

import collections
import itertools
import math
from typing import List, Text, Tuple
import tensorflow.compat.v1 as tf
from deepmath.deephol import deephol_pb2
from deepmath.deephol import io_util
from deepmath.deephol import theorem_fingerprint
from deepmath.deephol.utilities import deephol_stat_pb2
from deepmath.deephol.utilities import proof_analysis

FLAGS = tf.flags.FLAGS
OVERALL_LABEL = '<overall_statistics>'
LONG_THEOREM_LEN = 20  # Log fingerprints with proofs longer than this.

tf.flags.DEFINE_boolean(
    'stats_per_library_tag', True,
    'Aggregate proof statistics by library tag (default True).')


def add_value_to_log_scale_histogram(hist: deephol_stat_pb2.LogScaleHistogram,
                                     value):
  assert value >= 0
  log_scale_key = int(math.floor(math.log(value + 1, 2)))
  if log_scale_key not in hist.h:
    hist.h[log_scale_key] = 0
  hist.h[log_scale_key] += 1
  hist.total_value += value
  hist.num_values_added += 1


def merge_log_scale_histograms(target, source):
  for key, value in source.h.items():
    if key not in target.h:
      target.h[key] = 0
    target.h[key] += value
  target.total_value += source.total_value
  target.num_values_added += source.num_values_added


def merge_basic_histograms(target, source):
  for key, value in source.items():
    target[key] += value


def tactic_time_stats(stat: deephol_stat_pb2.TacticTimeStat,
                      tapp: deephol_pb2.TacticApplication):
  """Update a TacticTimeStat by a single TacticApplication."""
  stat.total_time += tapp.time_spent
  add_value_to_log_scale_histogram(stat.total_distribution, tapp.time_spent)
  if tapp.result == deephol_pb2.TacticApplication.SUCCESS:
    add_value_to_log_scale_histogram(stat.success_distribution, tapp.time_spent)
  if tapp.result == deephol_pb2.TacticApplication.ERROR:
    add_value_to_log_scale_histogram(stat.failed_distribution, tapp.time_spent)
  if tapp.result == deephol_pb2.TacticApplication.UNCHANGED:
    add_value_to_log_scale_histogram(stat.unchanged_distribution,
                                     tapp.time_spent)


def merge_tactic_time_stats(target: deephol_stat_pb2.TacticTimeStat,
                            source: deephol_stat_pb2.TacticTimeStat) -> None:
  """Combines statistics over two non-intersecting subsets of data.

  The sums are calculated for all the counters and the hystograms from both the
  target and the source and written into the target.

  Args:
    target: A source of stats to combine and also the destination where the
      combined stast are written.
    source: Another source of stats to combine.
  """
  target.total_time += source.total_time
  merge_log_scale_histograms(target.total_distribution,
                             source.total_distribution)
  merge_log_scale_histograms(target.success_distribution,
                             source.success_distribution)
  merge_log_scale_histograms(target.failed_distribution,
                             source.failed_distribution)
  merge_log_scale_histograms(target.unchanged_distribution,
                             source.unchanged_distribution)


def tactic_application_stats(stat: deephol_stat_pb2.TacticApplicationStat,
                             tapp: deephol_pb2.TacticApplication):
  """Updates TacticApplicationStat by a single TacticApplication."""
  stat.time_spent_per_tapp_result[deephol_pb2.TacticApplication.Result.Name(
      tapp.result)] += tapp.time_spent
  if tapp.tactic:
    tactic = tapp.tactic.split()[0]
  else:
    tactic = 'not_available'
  stat.time_spent_per_tactic[tactic] += tapp.time_spent
  stat.total_tactic_applications_per_tactic[tactic] += 1

  stat.time_per_rank[tapp.rank] += tapp.time_spent
  stat.total_per_rank[tapp.rank] += 1
  score = int(tapp.score)
  score = max(-20, score)
  stat.time_per_score[score] += tapp.time_spent
  stat.total_per_score[score] += 1

  if tapp.result == deephol_pb2.TacticApplication.SUCCESS:
    stat.successful_tactic_applications_per_tactic[tactic] += 1
    stat.success_per_rank[tapp.rank] += 1
    stat.success_per_score[score] += 1
    if not tapp.subgoals:
      stat.closing_tactic_applications_per_tactic[tactic] += 1
  if tapp.result == deephol_pb2.TacticApplication.UNCHANGED:
    stat.unchanged_tactic_applications_per_tactic[tactic] += 1
    stat.unchanged_per_rank[tapp.rank] += 1
    stat.unchanged_per_score[score] += 1
  if tapp.result == deephol_pb2.TacticApplication.ERROR:
    stat.failed_tactic_applications_per_tactic[tactic] += 1
    stat.failed_per_rank[tapp.rank] += 1
    stat.failed_per_score[score] += 1
  if tactic == 'ASM_MESON_TAC':
    tactic_time_stats(stat.meson_stat, tapp)
  if tactic == 'REWRITE_TAC':
    tactic_time_stats(stat.rewrite_stat, tapp)
  if tactic == 'SIMP_TAC':
    tactic_time_stats(stat.simp_stat, tapp)
  tactic_time_stats(stat.overall_stat, tapp)
  if tapp.closed:
    stat.closed_applications_per_tactic[tactic] += 1
    stat.closed_per_rank[tapp.rank] += 1
    stat.closed_per_score[score] += 1


def merge_proof_tapp_stats(target: deephol_stat_pb2.TacticApplicationStat,
                           source: deephol_stat_pb2.TacticApplicationStat):
  """Merges ProofStep-level statistics."""
  merge_basic_histograms(target.time_spent_per_tapp_result,
                         source.time_spent_per_tapp_result)
  merge_basic_histograms(target.time_spent_per_tactic,
                         source.time_spent_per_tactic)
  merge_basic_histograms(target.total_tactic_applications_per_tactic,
                         source.total_tactic_applications_per_tactic)
  merge_basic_histograms(target.successful_tactic_applications_per_tactic,
                         source.successful_tactic_applications_per_tactic)
  merge_basic_histograms(target.unchanged_tactic_applications_per_tactic,
                         source.unchanged_tactic_applications_per_tactic)
  merge_basic_histograms(target.failed_tactic_applications_per_tactic,
                         source.failed_tactic_applications_per_tactic)
  merge_basic_histograms(target.unknown_tactic_applications_per_tactic,
                         source.unknown_tactic_applications_per_tactic)
  merge_basic_histograms(target.closing_tactic_applications_per_tactic,
                         source.closing_tactic_applications_per_tactic)
  merge_basic_histograms(target.closed_applications_per_tactic,
                         source.closed_applications_per_tactic)
  merge_tactic_time_stats(target.meson_stat, source.meson_stat)
  merge_tactic_time_stats(target.rewrite_stat, source.rewrite_stat)
  merge_tactic_time_stats(target.simp_stat, source.simp_stat)
  merge_tactic_time_stats(target.overall_stat, source.overall_stat)

  merge_basic_histograms(target.time_per_rank, source.time_per_rank)
  merge_basic_histograms(target.total_per_rank, source.total_per_rank)
  merge_basic_histograms(target.success_per_rank, source.success_per_rank)
  merge_basic_histograms(target.failed_per_rank, source.failed_per_rank)
  merge_basic_histograms(target.unchanged_per_rank, source.unchanged_per_rank)
  merge_basic_histograms(target.closed_per_rank, source.closed_per_rank)

  merge_basic_histograms(target.time_per_score, source.time_per_score)
  merge_basic_histograms(target.total_per_score, source.total_per_score)
  merge_basic_histograms(target.success_per_score, source.success_per_score)
  merge_basic_histograms(target.failed_per_score, source.failed_per_score)
  merge_basic_histograms(target.unchanged_per_score, source.unchanged_per_score)
  merge_basic_histograms(target.closed_per_score, source.closed_per_score)


def add_to_histogram(histogram: deephol_stat_pb2.Histogram, value):
  """Add a single value to a histogram."""
  assert histogram.HasField('max_value') and histogram.max_value
  assert histogram.HasField('num_buckets') and histogram.num_buckets
  if not (value >= 0.0 and value <= histogram.max_value):
    histogram.num_values_out_of_range += 1
    return
  histogram.total_value += value
  histogram.num_values_added += 1
  key = int((value * histogram.num_buckets) // histogram.max_value)
  assert key >= 0
  assert key <= histogram.num_buckets
  if key not in histogram.buckets:
    histogram.buckets[key] = 0
  histogram.buckets[key] += 1


def merge_histograms(target: deephol_stat_pb2.Histogram,
                     source: deephol_stat_pb2.Histogram):
  """Merges histograms; initializes target, if necessary."""
  if not source.num_values_added:
    return
  if not target.num_buckets:
    target.num_buckets = source.num_buckets
  if target.num_buckets != source.num_buckets:
    raise ValueError('Error in computing statistics: num_buckets is %d vs %d' %
                     (target.num_buckets, source.num_buckets))
  if not target.max_value:
    target.max_value = source.max_value
  target.total_value += source.total_value
  target.num_values_added += source.num_values_added
  target.num_values_out_of_range += source.num_values_out_of_range
  if target.max_value != source.max_value:
    tf.logging.error('Statistics: max_value is %f vs %f' %
                     (target.max_value, source.max_value))
    # treating all source values as out of range
    for key in source.buckets:
      target.num_values_out_of_range += source.buckets[key]
    return
  for key in source.buckets:
    if key not in target.buckets:
      target.buckets[key] = 0
    target.buckets[key] += source.buckets[key]


def populate_pruning_statistics(proof_log: deephol_pb2.ProofLog,
                                pruning: deephol_stat_pb2.PruningStatistics,
                                successful_proof: bool):
  """Populates the PruningStatistics proto given a proof_log."""
  # initialize histograms
  pruning.pruned_parameters_num.CopyFrom(
      deephol_stat_pb2.Histogram(max_value=40, num_buckets=20))
  pruning.pruned_steps_num.CopyFrom(
      deephol_stat_pb2.Histogram(max_value=20, num_buckets=20))
  pruning.strong_pruning_successful.CopyFrom(
      deephol_stat_pb2.Histogram(max_value=20, num_buckets=20))

  if proof_log.prover_options.prune_proof:
    nodes = proof_log.extracted_proof
  else:
    nodes = proof_log.nodes
  for node in nodes:
    node_pruning_time_ms = 0
    for tapp in node.proofs:
      if tapp.parameters and tapp.pruning_time_spent_ms:
        node_pruning_time_ms += tapp.pruning_time_spent_ms
        for p in tapp.parameters:
          if p.parameter_type == deephol_pb2.Tactic.THEOREM_LIST:
            add_to_histogram(pruning.pruned_parameters_num,
                             len(p.hard_negative_theorems))
        add_to_histogram(pruning.strong_pruning_successful,
                         tapp.strong_pruning_successful_num)

    add_value_to_log_scale_histogram(pruning.proof_node_pruning_ms,
                                     node_pruning_time_ms)

  add_value_to_log_scale_histogram(pruning.proof_log_pruning_ms,
                                   proof_log.pruning_time_ms)
  if successful_proof:
    add_to_histogram(pruning.pruned_steps_num, proof_log.pruned_steps_num)


def populate_search_statistics(
    log_stats: deephol_pb2.SearchStatistics,
    agg_stats: deephol_stat_pb2.AggregateSearchStatistics,
    successful_proof: bool):
  """From deephol_pb2.SearchStatistics to deephol_stat_pb2.AggregateSearchStatistics."""
  default_search_stats = deephol_stat_pb2.AggregateSearchStatistics(
      search_depths=deephol_stat_pb2.Histogram(max_value=20, num_buckets=20),
      total_expansions=deephol_stat_pb2.Histogram(
          max_value=4000, num_buckets=20),
      failed_expansions=deephol_stat_pb2.Histogram(
          max_value=10000, num_buckets=20),
      search_states=deephol_stat_pb2.Histogram(max_value=1000, num_buckets=20),
      total_prediction_times_ms=deephol_stat_pb2.LogScaleHistogram(),
      mcts_root_values=deephol_stat_pb2.Histogram(
          max_value=1.0, num_buckets=20),
      mcts_path_target_values=deephol_stat_pb2.Histogram(
          max_value=1.0, num_buckets=20),
      mcts_path_values_difference=deephol_stat_pb2.Histogram(
          max_value=1.0, num_buckets=20),
      mcts_initial_root_values_difference=deephol_stat_pb2.Histogram(
          max_value=1.0, num_buckets=20),
      mcts_path_values_squared_difference=deephol_stat_pb2.Histogram(
          max_value=1.0, num_buckets=20),
      mcts_values_all_states=deephol_stat_pb2.Histogram(
          max_value=1.0, num_buckets=20),
      mcts_initial_root_values=deephol_stat_pb2.Histogram(
          max_value=1.0, num_buckets=20),
      policy_kl_divergences=deephol_stat_pb2.Histogram(
          max_value=1.0, num_buckets=20),
      mcts_initial_root_values_closed=deephol_stat_pb2.Histogram(
          max_value=1.0, num_buckets=20),
      mcts_initial_root_values_open=deephol_stat_pb2.Histogram(
          max_value=1.0, num_buckets=20))
  agg_stats.CopyFrom(default_search_stats)
  add_to_histogram(agg_stats.search_depths, log_stats.search_depth)
  add_to_histogram(agg_stats.total_expansions, log_stats.total_expansions)
  add_to_histogram(agg_stats.failed_expansions, log_stats.failed_expansions)
  add_to_histogram(agg_stats.search_states, log_stats.num_search_states)
  agg_stats.aggregate_total_prediction_time_ms = int(
      log_stats.total_prediction_time_sec * 1000)
  add_value_to_log_scale_histogram(
      agg_stats.total_prediction_times_ms,
      int(log_stats.total_prediction_time_sec * 1000))
  for idx, value in enumerate(log_stats.mcts_path_values):
    add_to_histogram(agg_stats.mcts_root_values, value)
    if idx == 0:
      add_to_histogram(agg_stats.mcts_initial_root_values, value)
      if successful_proof:
        add_to_histogram(agg_stats.mcts_initial_root_values_closed, value)
      else:
        add_to_histogram(agg_stats.mcts_initial_root_values_open, value)
  for value in log_stats.mcts_path_target_values:
    add_to_histogram(agg_stats.mcts_path_target_values, value)
  for diff in log_stats.mcts_path_values_difference:
    add_to_histogram(agg_stats.mcts_path_values_difference, diff)
  if log_stats.mcts_path_values_difference:
    add_to_histogram(agg_stats.mcts_initial_root_values_difference,
                     log_stats.mcts_path_values_difference[0])
  for diff in log_stats.mcts_path_values_squared_difference:
    add_to_histogram(agg_stats.mcts_path_values_squared_difference, diff)
  for value in log_stats.mcts_values_all_states:
    add_to_histogram(agg_stats.mcts_values_all_states, value)

  for kl_divergence in log_stats.policy_kl_divergences:
    add_to_histogram(agg_stats.policy_kl_divergences, kl_divergence)

  agg_stats.goals_per_search_state.CopyFrom(log_stats.goals_per_search_state)
  agg_stats.assumptions_per_goal.CopyFrom(log_stats.assumptions_per_goal)


def merge_pruning_statistics(target: deephol_stat_pb2.PruningStatistics,
                             source: deephol_stat_pb2.PruningStatistics):
  merge_log_scale_histograms(target.proof_log_pruning_ms,
                             source.proof_log_pruning_ms)
  merge_log_scale_histograms(target.proof_node_pruning_ms,
                             source.proof_node_pruning_ms)
  merge_histograms(target.pruned_steps_num, source.pruned_steps_num)
  merge_histograms(target.pruned_parameters_num, source.pruned_parameters_num)
  merge_histograms(target.strong_pruning_successful,
                   source.strong_pruning_successful)


def merge_search_statistics(target: deephol_stat_pb2.AggregateSearchStatistics,
                            source: deephol_stat_pb2.AggregateSearchStatistics):
  """Merge two deephol_stat_pb2.AggregateSearchStatistics."""
  merge_histograms(target.search_depths, source.search_depths)
  merge_histograms(target.total_expansions, source.total_expansions)
  merge_histograms(target.failed_expansions, source.failed_expansions)
  merge_histograms(target.search_states, source.search_states)
  target.aggregate_total_prediction_time_ms += source.aggregate_total_prediction_time_ms
  merge_log_scale_histograms(target.total_prediction_times_ms,
                             source.total_prediction_times_ms)
  merge_histograms(target.mcts_root_values, source.mcts_root_values)
  merge_histograms(target.mcts_initial_root_values,
                   source.mcts_initial_root_values)
  merge_histograms(target.policy_kl_divergences, source.policy_kl_divergences)
  merge_histograms(target.mcts_initial_root_values_closed,
                   source.mcts_initial_root_values_closed)
  merge_histograms(target.mcts_initial_root_values_open,
                   source.mcts_initial_root_values_open)
  merge_histograms(target.mcts_path_target_values,
                   source.mcts_path_target_values)
  merge_histograms(target.mcts_path_values_difference,
                   source.mcts_path_values_difference)
  merge_histograms(target.mcts_initial_root_values_difference,
                   source.mcts_initial_root_values_difference)
  merge_histograms(target.mcts_path_values_squared_difference,
                   source.mcts_path_values_squared_difference)
  merge_histograms(target.mcts_values_all_states, source.mcts_values_all_states)
  merge_histograms(target.goals_per_search_state, source.goals_per_search_state)
  merge_histograms(target.assumptions_per_goal, source.assumptions_per_goal)


def proof_log_stats(
    proof_log: deephol_pb2.ProofLog) -> deephol_stat_pb2.ProofStat:
  """Create statistics for a single proof log."""
  stat = deephol_stat_pb2.ProofStat(
      num_theorems_attempted=0,
      num_theorems_proved=0,
      num_theorems_with_bad_proof=0,
      num_nodes=len(proof_log.nodes),
      reduced_node_indices=[],
      closed_node_indices=[],
      labels=proof_log_labels(proof_log))

  proof_log = io_util.fix_legacy_proof_log(proof_log)

  task_fingerprint = None
  if proof_log.HasField('prover_task'):
    task_fingerprint = theorem_fingerprint.TaskFingerprint(
        proof_log.prover_task)
  else:
    for node in proof_log.nodes:
      if node.root_goal:
        task_fingerprint = theorem_fingerprint.Fingerprint(node.goal)
  if task_fingerprint is not None:
    stat.num_theorems_attempted += 1
    stat.attempted_theorems[task_fingerprint] = 1
    if proof_log.num_proofs > 1:
      raise NotImplementedError
    if proof_log.num_proofs:
      stat.num_theorems_proved += 1
      stat.proven_theorems[task_fingerprint] = 1
    if proof_log.search_statistics.mcts_path_values_difference:
      diffs = list(proof_log.search_statistics.mcts_path_values_difference)
      if task_fingerprint not in stat.mean_mcts_value_diffs:
        stat.mean_mcts_value_diffs[task_fingerprint] = 0
      avg = sum(diffs) / len(diffs)
      stat.mean_mcts_value_diffs[task_fingerprint] += avg

  for i, node in enumerate(proof_log.nodes):
    if node.status == deephol_pb2.ProofNode.PROVED:
      stat.closed_node_indices.append(i)

  if stat.num_theorems_proved > 0:
    analysis_result = proof_analysis.find_reasons(proof_log)
    if analysis_result is None:
      stat.num_theorems_with_bad_proof += 1
      tf.logging.error('Found bad proof.')
    else:
      _, reduced_node_indices = analysis_result
      stat.reduced_node_indices.extend(reduced_node_indices)
      for node_idx in reduced_node_indices:
        reduced_node = proof_log.nodes[node_idx]
        successful_app = None
        for tactic_app in reduced_node.proofs:
          if tactic_app.closed:
            successful_app = tactic_app
            break
        if successful_app is not None:
          for param in successful_app.parameters:
            for theorem in param.theorems:
              fp = theorem_fingerprint.Fingerprint(theorem)
              if fp not in stat.premise_usages:
                stat.premise_usages[fp] = 0
              stat.premise_usages[fp] += 1

  if proof_log.HasField('time_spent'):
    stat.time_spent_milliseconds = proof_log.time_spent

  for node in proof_log.nodes:
    stat.total_prediction_time += node.action_generation_time_millisec
    add_value_to_log_scale_histogram(stat.node_prediction_time_histogram,
                                     node.action_generation_time_millisec)
    stat.total_embedding_time_ms += node.proof_state_emb_time_ms
    add_value_to_log_scale_histogram(stat.embedding_times_ms,
                                     node.proof_state_emb_time_ms)
    stat.total_theorem_score_time_ms += node.theorem_scores_time_ms
    add_value_to_log_scale_histogram(stat.theorem_scores_times_ms,
                                     node.theorem_scores_time_ms)
    add_value_to_log_scale_histogram(stat.assumptions_ranking_time_ms,
                                     node.assumptions_ranking_time_ms)
    add_value_to_log_scale_histogram(stat.heuristic_ranking_time_ms,
                                     node.heuristic_ranking_time_ms)
    for tapp in node.proofs:
      tactic_application_stats(stat.tapp_stat, tapp)

  populate_search_statistics(proof_log.search_statistics,
                             stat.search_statistics,
                             stat.num_theorems_proved > 0)
  populate_pruning_statistics(proof_log, stat.pruning,
                              stat.num_theorems_proved > 0)
  return stat


def merge_stat(target: deephol_stat_pb2.ProofAggregateStat,
               source: deephol_stat_pb2.ProofStat):
  """Update the aggregated statistics with the individual statistics.

  Args:
    target: Aggregeated statistics to be updated.
    source: Statistics of a single proof log.
  """
  target.num_theorems_attempted += source.num_theorems_attempted
  target.num_theorems_proved += source.num_theorems_proved
  target.num_theorems_with_bad_proof += source.num_theorems_with_bad_proof
  target.num_nodes += source.num_nodes
  target.num_reduced_nodes += len(source.reduced_node_indices)
  target.num_closed_nodes += len(source.closed_node_indices)
  target.time_spent_milliseconds += source.time_spent_milliseconds
  add_value_to_log_scale_histogram(target.proof_time_histogram,
                                   source.time_spent_milliseconds)
  if source.num_theorems_proved:
    success_hist = target.proof_time_histogram_proved
    target.num_reduced_nodes_distribution[len(source.reduced_node_indices)] += 1
  else:
    success_hist = target.proof_time_histogram_failed
  add_value_to_log_scale_histogram(success_hist, source.time_spent_milliseconds)
  merge_proof_tapp_stats(target.tapp_stat, source.tapp_stat)
  target.total_prediction_time += source.total_prediction_time
  add_value_to_log_scale_histogram(target.proof_prediction_time_histogram,
                                   source.total_prediction_time)
  merge_log_scale_histograms(target.node_prediction_time_histogram,
                             source.node_prediction_time_histogram)
  target.total_embedding_time_ms += source.total_embedding_time_ms
  merge_log_scale_histograms(target.embedding_times_ms,
                             source.embedding_times_ms)
  target.total_theorem_score_time_ms += source.total_theorem_score_time_ms
  merge_log_scale_histograms(target.theorem_scores_times_ms,
                             source.theorem_scores_times_ms)
  merge_log_scale_histograms(target.assumptions_ranking_time_ms,
                             source.assumptions_ranking_time_ms)
  merge_log_scale_histograms(target.heuristic_ranking_time_ms,
                             source.heuristic_ranking_time_ms)
  if source.HasField('time_spent_milliseconds') and source.num_theorems_proved:
    if source.num_theorems_proved != 1:
      tf.logging.error('More than one proof in single proof log; Cactus plot '
                       'will be incaccurate.')
    target.proof_closed_after_millis.append(source.time_spent_milliseconds)
  for fingerprint in source.attempted_theorems:
    target.attempted_theorems[fingerprint] += source.attempted_theorems[
        fingerprint]
  for fingerprint in source.proven_theorems:
    target.proven_theorems[fingerprint] += source.proven_theorems[fingerprint]
    if len(source.reduced_node_indices) >= LONG_THEOREM_LEN:
      target.long_proofs_fp[fingerprint] = len(source.reduced_node_indices)
  for fingerprint in source.mean_mcts_value_diffs:
    target.mean_mcts_value_diffs[fingerprint] += source.mean_mcts_value_diffs[
        fingerprint]

  for key, value in source.premise_usages.items():
    if key not in target.premise_usages:
      target.premise_usages[key] = 0
    target.premise_usages[key] += value

  merge_search_statistics(target.search_statistics, source.search_statistics)
  merge_pruning_statistics(target.pruning, source.pruning)


def merge_aggregate_stat(target: deephol_stat_pb2.ProofAggregateStat,
                         source: deephol_stat_pb2.ProofAggregateStat):
  """Merge two aggregated statistics.

  Args:
    target: Aggregeated statistics to be updated.
    source: Statistics to be merged in.
  """
  target.num_theorems_attempted += source.num_theorems_attempted
  target.num_theorems_proved += source.num_theorems_proved
  target.num_theorems_with_bad_proof += source.num_theorems_with_bad_proof
  target.num_nodes += source.num_nodes
  target.num_reduced_nodes += source.num_reduced_nodes
  target.num_closed_nodes += source.num_closed_nodes
  target.time_spent_milliseconds += source.time_spent_milliseconds
  merge_log_scale_histograms(target.proof_time_histogram,
                             source.proof_time_histogram)
  merge_log_scale_histograms(target.proof_time_histogram_proved,
                             source.proof_time_histogram_proved)
  merge_log_scale_histograms(target.proof_time_histogram_failed,
                             source.proof_time_histogram_failed)
  merge_proof_tapp_stats(target.tapp_stat, source.tapp_stat)
  merge_basic_histograms(target.num_reduced_nodes_distribution,
                         source.num_reduced_nodes_distribution)
  target.total_prediction_time += source.total_prediction_time
  merge_log_scale_histograms(target.proof_prediction_time_histogram,
                             source.proof_prediction_time_histogram)
  merge_log_scale_histograms(target.node_prediction_time_histogram,
                             source.node_prediction_time_histogram)
  target.total_embedding_time_ms += source.total_embedding_time_ms
  merge_log_scale_histograms(target.embedding_times_ms,
                             source.embedding_times_ms)
  target.total_theorem_score_time_ms += source.total_theorem_score_time_ms
  merge_log_scale_histograms(target.theorem_scores_times_ms,
                             source.theorem_scores_times_ms)
  merge_log_scale_histograms(target.assumptions_ranking_time_ms,
                             source.assumptions_ranking_time_ms)
  merge_log_scale_histograms(target.heuristic_ranking_time_ms,
                             source.heuristic_ranking_time_ms)

  target.proof_closed_after_millis.extend(source.proof_closed_after_millis)
  for fingerprint in source.attempted_theorems:
    target.attempted_theorems[fingerprint] += source.attempted_theorems[
        fingerprint]
  for fingerprint in source.proven_theorems:
    target.proven_theorems[fingerprint] += source.proven_theorems[fingerprint]
  for fingerprint in source.long_proofs_fp:
    target.long_proofs_fp[fingerprint] = source.long_proofs_fp[fingerprint]
  for fingerprint in source.mean_mcts_value_diffs:
    target.mean_mcts_value_diffs[fingerprint] += source.mean_mcts_value_diffs[
        fingerprint]

  for key, value in source.premise_usages.items():
    if key not in target.premise_usages:
      target.premise_usages[key] = 0
    target.premise_usages[key] += value

  merge_search_statistics(target.search_statistics, source.search_statistics)
  merge_pruning_statistics(target.pruning, source.pruning)


def merge_labeled_stats(
    target: deephol_stat_pb2.LabeledAggregateStats,
    source: deephol_stat_pb2.LabeledAggregateStats,
) -> None:
  """Merge two aggregated statistics.

  Args:
    target: Aggregated statistics to be updated.
    source: Statistics to be merged in.
  """

  for tag, aggstat in source.labeled_stats.items():
    if tag in target.labeled_stats:
      merge_aggregate_stat(target.labeled_stats[tag], aggstat)
    else:
      target.labeled_stats[tag].CopyFrom(aggstat)


def merge_into_labeled_aggregate_stat(
    target: deephol_stat_pb2.LabeledAggregateStats,
    source: deephol_stat_pb2.ProofStat
) -> deephol_stat_pb2.LabeledAggregateStats:
  for label in source.labels:
    if label in target.labeled_stats:
      merge_stat(target.labeled_stats[label], source)
    else:
      target.labeled_stats[label].CopyFrom(aggregate_stats([source]))
  return target


def aggregate_stats(
    stats: List[deephol_stat_pb2.ProofStat]
) -> deephol_stat_pb2.ProofAggregateStat:
  """Merge a list of proof log statistics.

  Args:
     stats: List of individual proof log statistics.

  Returns:
     Aggregated proof statistics.
  """
  result = deephol_stat_pb2.ProofAggregateStat()
  for stat in stats:
    merge_stat(result, stat)
  return result


def stat_to_string(stat: deephol_stat_pb2.ProofStat) -> Text:
  """Return a short summary of the stat."""
  status = 'failed'
  if stat.num_theorems_proved:
    status = 'proved'
  return '%.2f secs spent in %d/%d/%d nodes : %s' % (
      stat.time_spent_milliseconds / 1000.0, len(stat.reduced_node_indices),
      len(stat.closed_node_indices), stat.num_nodes, status)


def aggregate_stat_to_string(s: deephol_stat_pb2.ProofAggregateStat) -> Text:
  """Return a short summary of the aggregated statistics."""
  return ('Proofs: %d/%d - [%d] nodes: %d/%d/%d time: %.1f s. (%.1f days) '
          'prediction time: %.1f s') % (
              s.num_theorems_proved, s.num_theorems_attempted,
              s.num_theorems_with_bad_proof, s.num_reduced_nodes,
              s.num_closed_nodes, s.num_nodes,
              s.time_spent_milliseconds / 1000.0, s.time_spent_milliseconds /
              (1000.0 * 24 * 60 * 60), s.total_prediction_time / 1000.0)


def labeled_agg_stat_to_string(
    s: deephol_stat_pb2.LabeledAggregateStats) -> Text:
  """Summarize labeled aggregated statistics, broken down by tag."""

  overall = s.labeled_stats.pop(OVERALL_LABEL, None)
  if overall:
    total = 'Overall stats:\n%s\n' % aggregate_stat_to_string(overall)
  else:
    total = ''

  by_tag = [
      '\nLabel: %s\n%s' % (k, aggregate_stat_to_string(v))
      for k, v in s.labeled_stats.items()
  ]
  return total + '\n'.join(by_tag)


def copy_default(proto_map, default=0):
  # Using proto maps with default values causes the proto to change when a
  # non-existant key returns a default value. This causes the following error:
  # "Detected DoFn input element mutation which is not allowed."
  # By copying from proto map to defaultdict, we avoid this issue.
  copy_dict = collections.defaultdict(lambda: default)
  copy_dict.update(dict(proto_map))
  return copy_dict


def log_scale_histograms_to_string(histograms: List[Tuple[
    Text, deephol_stat_pb2.LogScaleHistogram]],
                                   scale=1) -> Text:
  """Converts histogram into string; missing values are interpreted as 0."""
  histograms = [(s, copy_default(hist.h)) for s, hist in histograms]
  header = ['ms    '.ljust(6)] + [s.rjust(12) for s, _ in histograms]
  lines = [''.join(header)]
  entry_nums = [len(hist) for _, hist in histograms]
  found_non_zero_entries = False  # suppresses initial rows that are all zeros
  index = 0
  while any(entry_nums):
    entries = []
    for hist_idx, (_, hist) in enumerate(histograms):
      if index in hist:
        entry_nums[hist_idx] -= 1
      entries.append(hist.get(index, 0) * scale)
    if found_non_zero_entries or any(entries):
      found_non_zero_entries = True
      lines.append(('2**%d:' % index).ljust(6) +
                   ''.join([str(e).rjust(12) for e in entries]))
    index += 1
  return '\n'.join(lines)


def average_of_basic_histogram(histogram) -> float:
  aggregate = 0
  num_items = 0
  for key, value in histogram.items():
    num_items += value
    aggregate += key * value
  if num_items:
    return aggregate / num_items
  else:
    return float('nan')


def histograms_to_string(maps_map, scale=1) -> Text:
  """Turns a list of names protobuf maps into a nice table."""
  column_size = 14
  lines = []
  all_keys = list(
      set(itertools.chain.from_iterable([m.keys() for _, m in maps_map])))
  all_keys.sort()
  key_lengths = list(map(len, map(str, all_keys)))
  key_lengths.append(8)
  longest_key = max(key_lengths) + 1
  header = [''.ljust(longest_key)] + [s.rjust(column_size) for s, _ in maps_map]
  lines.append(''.join(header))

  def nice_string(v):
    if isinstance(v, float):
      res = '%.3f' % v
    else:
      res = str(v)
    return res.rjust(column_size)

  for key in all_keys:
    row = [nice_string(m[key] * scale) for _, m in maps_map]
    lines.append(str(key).ljust(longest_key) + ''.join(row))
  return str('\n'.join(lines))


def histogram_to_string(hist: deephol_stat_pb2.Histogram) -> Text:
  """"Print out average and bucket values for histogram."""
  lines = []
  if not hist.num_values_added or not hist.num_buckets or not hist.max_value:
    lines.append('num_values_added: %d and num_buckets: %d and max_value %d' %
                 (hist.num_values_added, hist.num_buckets, hist.max_value))
  else:
    if hist.num_values_out_of_range:
      lines.append('VALUES OUT OF RANGE: %d' % hist.num_values_out_of_range)
    lines.append('Average: %f' % (hist.total_value / hist.num_values_added))
    for bucket_idx in range(hist.num_buckets + 1):
      bucket_range_min = hist.max_value * bucket_idx / hist.num_buckets
      bucket_range_min = ('%.2f' % bucket_range_min).ljust(10)
      bucket_range_max = hist.max_value * (bucket_idx + 1) / hist.num_buckets
      if bucket_idx == hist.num_buckets:
        bucket_range_max = hist.max_value
      bucket_range_max = ('< %.2f' % bucket_range_max).rjust(10)
      if bucket_idx in hist.buckets:
        value = hist.buckets[bucket_idx]
      else:
        value = 0
      value = ('%d' % value).rjust(10)
      lines.append('%s-%s:%s' % (bucket_range_min, bucket_range_max, value))
  return '\n'.join(lines)


def average_by(time, applications):
  avg = {}
  for key, app_num in applications.items():
    if app_num > 0 and key in time:
      avg[key] = time[key] / app_num
    else:
      avg[key] = 0.
  return avg


def basic_statistics(s: deephol_stat_pb2.ProofAggregateStat) -> Text:
  """Prints high-level aggregate statistics of a set of proof logs."""
  proven_theorems = len(list(s.proven_theorems.keys()))
  attempted_theorems = len(list(s.attempted_theorems.keys()))
  if attempted_theorems:
    perc_proven = (proven_theorems * 100.0) / attempted_theorems
  else:
    perc_proven = float('nan')

  elems = [
      'Unique theorems proven: %d' % proven_theorems,
      'Percentage: %.2f' % perc_proven,
      'Proofs attempts: %d  (unique theorems: %d)' %
      (s.num_theorems_attempted, attempted_theorems),
      'Search Time: %s seconds' % (float(s.time_spent_milliseconds) / 1000.0)
  ]
  return '   '.join(elems)


def num_tactic_applications(s: deephol_stat_pb2.TacticApplicationStat) -> int:
  apps = [value for _, value in s.total_tactic_applications_per_tactic.items()]
  return sum(apps)


def num_successful_tactic_applications(
    s: deephol_stat_pb2.TacticApplicationStat) -> int:
  apps = [
      value for _, value in s.successful_tactic_applications_per_tactic.items()
  ]
  return sum(apps)


def num_failed_tactic_applications(
    s: deephol_stat_pb2.TacticApplicationStat) -> int:
  apps = [value for _, value in s.failed_tactic_applications_per_tactic.items()]
  return sum(apps)


def num_unchanged_tactic_applications(
    s: deephol_stat_pb2.TacticApplicationStat) -> int:
  apps = [
      value for _, value in s.unchanged_tactic_applications_per_tactic.items()
  ]
  return sum(apps)


def detailed_statistics(
    all_aggregate_stats: deephol_stat_pb2.LabeledAggregateStats) -> Text:
  """Converts histograms into pretty tables."""

  s = all_aggregate_stats.labeled_stats[OVERALL_LABEL]

  lines = []
  lines.append(basic_statistics(s))
  lines.append('')

  for tag in all_aggregate_stats.labeled_stats.keys():
    if tag == OVERALL_LABEL:
      continue
    lines.append('Stats for %s:' % tag)
    lines.append(basic_statistics(all_aggregate_stats.labeled_stats[tag]))
    lines.append('')

  lines.append(
      'How long are the proofs?\n%s' %
      histograms_to_string([('#proofs', s.num_reduced_nodes_distribution)]))
  lines.append('Average length of proofs: %f\n' %
               average_of_basic_histogram(s.num_reduced_nodes_distribution))

  lines.append(
      'Time spent in tactic applications:  %f seconds  (%f%% of total search time)\n'
      % (float(s.tapp_stat.overall_stat.total_time) / 1000.0,
         float(s.tapp_stat.overall_stat.total_time) * 100 /
         float(s.time_spent_milliseconds + 1)))

  lines.append(
      'How much prediction time was spent in the value network?\n'
      'Total (seconds): %d  (%f%% of total search time)\n%s\n' %
      (float(s.search_statistics.aggregate_total_prediction_time_ms) / 1000.0,
       s.search_statistics.aggregate_total_prediction_time_ms * 100.0 /
       (s.time_spent_milliseconds + 1),
       log_scale_histograms_to_string(
           [('per proof', s.search_statistics.total_prediction_times_ms)])))

  lines.append(
      'How much tactic prediction time did proofs take?\n'
      'Total (seconds): %d (%f%% of total search time)\n%s\n' %
      (float(s.total_prediction_time) / 1000.0, float(s.total_prediction_time) *
       100 / float(s.time_spent_milliseconds + 1),
       log_scale_histograms_to_string(
           [('per proof', s.proof_prediction_time_histogram),
            ('per node', s.node_prediction_time_histogram)])))
  if s.HasField('total_embedding_time_ms') and s.total_prediction_time:
    lines.append(
        'Time to compute goal embeddings:\nTotal (seconds): %d  (%.2f%% of '
        'prediction time)\n%s\n' %
        (float(s.total_embedding_time_ms / 1000.0),
         s.total_embedding_time_ms * 100.0 / s.total_prediction_time,
         log_scale_histograms_to_string([('per node', s.embedding_times_ms)])))
  if s.HasField('total_theorem_score_time_ms') and s.total_prediction_time:
    lines.append(
        'Time to score theorems:\nTotal (seconds): %d  (%.2f%% of '
        'prediction time)\n%s\n' %
        (float(s.total_theorem_score_time_ms) / 1000.0,
         s.total_theorem_score_time_ms * 100.0 / s.total_prediction_time,
         log_scale_histograms_to_string([('per node', s.theorem_scores_times_ms)
                                        ])))
  if s.assumptions_ranking_time_ms.num_values_added and s.total_prediction_time:
    lines.append('Time to score assumptions (contained in time to score '
                 'theorems):\nTotal (seconds): %d  (%.2f%% of prediction '
                 'time)\n%s\n' %
                 (float(s.assumptions_ranking_time_ms.total_value) / 1000.0,
                  s.assumptions_ranking_time_ms.total_value * 100.0 /
                  s.total_prediction_time,
                  log_scale_histograms_to_string(
                      [('per node', s.assumptions_ranking_time_ms)])))
  if s.heuristic_ranking_time_ms.num_values_added and s.total_prediction_time:
    lines.append('Time to score premises with DeepHOL zero heuristic '
                 '(contained in time to score theorems):\nTotal '
                 '(seconds): %d  (%.2f%% of prediction time)\n%s\n' %
                 (float(s.heuristic_ranking_time_ms.total_value) / 1000.0,
                  s.heuristic_ranking_time_ms.total_value * 100.0 /
                  s.total_prediction_time,
                  log_scale_histograms_to_string(
                      [('per node', s.heuristic_ranking_time_ms)])))

  if s.HasField('pruning'):
    lines.append('\nTime to prune theorems:\nTotal (seconds): %f\n%s\n' %
                 (float(s.pruning.proof_log_pruning_ms.total_value) / 1000.0,
                  log_scale_histograms_to_string(
                      [('per proof', s.pruning.proof_log_pruning_ms),
                       ('per node', s.pruning.proof_node_pruning_ms)])))
    lines.append('Number of steps pruned per successful proof:\n%s' %
                 histogram_to_string(s.pruning.pruned_steps_num))
    lines.append(
        'Number of premises pruned per step that has a THEOREM_LIST:\n%s' %
        histogram_to_string(s.pruning.pruned_parameters_num))
    lines.append(
        'Number of premises pruned through proof-level (strong) pruning:\n%s' %
        histogram_to_string(s.pruning.strong_pruning_successful))

  if s.HasField('proof_time_histogram'):
    lines.append(
        '\nHow long did the proof attempts take?\nTotal (seconds): %d\n%s\n' %
        (float(s.proof_time_histogram.total_value) / 1000.0,
         log_scale_histograms_to_string([('per proof', s.proof_time_histogram)
                                        ])))

  lines.append('Proof nodes encountered during search: %d' % s.num_nodes)
  if s.num_theorems_attempted:
    lines.append(
        'Average number of proof nodes encountered during search: %f\n' %
        (s.num_nodes / s.num_theorems_attempted))

  lines.append('Histogram of assumptions per goal:\n%s\n' %
               histogram_to_string(s.search_statistics.assumptions_per_goal))

  lines.append('How deep are the MCTS searches?\n%s\n' %
               histogram_to_string(s.search_statistics.search_depths))

  lines.append('How many MCTS expansions per proof search?\n%s\n' %
               histogram_to_string(s.search_statistics.total_expansions))

  lines.append('How many MCTS failed expansions per proof search?\n%s\n' %
               histogram_to_string(s.search_statistics.failed_expansions))

  lines.append('How many search states per proof search?\n%s\n' %
               histogram_to_string(s.search_statistics.search_states))

  lines.append(
      'Distribution of predicted values at the MCTS search roots:\n%s\n' %
      histogram_to_string(s.search_statistics.mcts_root_values))

  lines.append('Distribution of target values at the MCTS search roots:\n%s\n' %
               histogram_to_string(s.search_statistics.mcts_path_target_values))

  lines.append(
      'Distribution of differences between predicted and target '
      'values at the MCTS search roots:\n%s\n' %
      histogram_to_string(s.search_statistics.mcts_path_values_difference))

  lines.append('Distribution of differences between predicted and target '
               'values at the initial MCTS search root:\n%s\n' %
               histogram_to_string(
                   s.search_statistics.mcts_initial_root_values_difference))

  lines.append('Distribution of predicted values of all search states:\n%s\n' %
               histogram_to_string(s.search_statistics.mcts_values_all_states))

  lines.append(
      'Distribution of values at the initial MCTS search roots:\n%s\n' %
      histogram_to_string(s.search_statistics.mcts_initial_root_values))

  lines.append(
      'Distribution of values at closed initial MCTS search roots:\n%s\n' %
      histogram_to_string(s.search_statistics.mcts_initial_root_values_closed))

  lines.append(
      'Distribution of values at unclosed initial MCTS search roots:\n%s\n' %
      histogram_to_string(s.search_statistics.mcts_initial_root_values_open))

  lines.append('Distribution of KL divergences at the MCTS roots:\n%s\n' %
               histogram_to_string(s.search_statistics.policy_kl_divergences))

  lines.append('Histogram of number of goals per search state:\n%s\n' %
               histogram_to_string(s.search_statistics.goals_per_search_state))

  lines.append('How much tactic application time did proofs take?\n%s\n' %
               (log_scale_histograms_to_string(
                   [('success', s.proof_time_histogram_proved),
                    ('failed', s.proof_time_histogram_failed)])))

  lines.append(
      'How much time was spent per TacticApplication.Result?\n%s\n' %
      (histograms_to_string(
          [('seconds', s.tapp_stat.time_spent_per_tapp_result)], scale=.001)))

  avg_tapp_time = average_by(s.tapp_stat.time_spent_per_tactic,
                             s.tapp_stat.total_tactic_applications_per_tactic)
  lines.append(
      'How much time was spent per tactic (seconds)?\n%s\n' %
      (histograms_to_string([('seconds', s.tapp_stat.time_spent_per_tactic),
                             ('avg s', avg_tapp_time)],
                            scale=.001)))

  total_tactic_applications_per_tactic = copy_default(
      s.tapp_stat.total_tactic_applications_per_tactic)
  successful_tactic_applications_per_tactic = copy_default(
      s.tapp_stat.successful_tactic_applications_per_tactic)
  failed_tactic_applications_per_tactic = copy_default(
      s.tapp_stat.failed_tactic_applications_per_tactic)
  unchanged_tactic_applications_per_tactic = copy_default(
      s.tapp_stat.unchanged_tactic_applications_per_tactic)
  unknown_tactic_applications_per_tactic = copy_default(
      s.tapp_stat.unknown_tactic_applications_per_tactic)
  closing_tactic_applications_per_tactic = copy_default(
      s.tapp_stat.closing_tactic_applications_per_tactic)
  closed_applications_per_tactic = copy_default(
      s.tapp_stat.closed_applications_per_tactic)

  lines.append('Outcome of tactic applications per tactic:\n%s\n' %
               (histograms_to_string([
                   ('total', total_tactic_applications_per_tactic),
                   ('success', successful_tactic_applications_per_tactic),
                   ('failed', failed_tactic_applications_per_tactic),
                   ('unchanged', unchanged_tactic_applications_per_tactic),
                   ('unknown', unknown_tactic_applications_per_tactic),
                   ('closing', closing_tactic_applications_per_tactic),
                   ('in closed', closed_applications_per_tactic),
                   ('success frac',
                    average_by(successful_tactic_applications_per_tactic,
                               total_tactic_applications_per_tactic)),
               ])))

  lines.append(('Distribution of invocation times of ASM_MESON_TAC in '
                'milliseconds:\n%s\n') % (log_scale_histograms_to_string([
                    ('total', s.tapp_stat.meson_stat.total_distribution),
                    ('successful', s.tapp_stat.meson_stat.success_distribution),
                    ('failed', s.tapp_stat.meson_stat.failed_distribution)
                ])))

  lines.append(
      ('Distribution of invocation times of REWRITE_TAC in '
       'milliseconds:\n%s\n') % (log_scale_histograms_to_string(
           [('total', s.tapp_stat.rewrite_stat.total_distribution),
            ('successful', s.tapp_stat.rewrite_stat.success_distribution),
            ('failed', s.tapp_stat.rewrite_stat.failed_distribution),
            ('unchanged', s.tapp_stat.rewrite_stat.unchanged_distribution)])))

  lines.append(('Distribution of invocation times of SIMP_TAC in '
                'milliseconds:\n%s\n') % (log_scale_histograms_to_string([
                    ('total', s.tapp_stat.simp_stat.total_distribution),
                    ('successful', s.tapp_stat.simp_stat.success_distribution),
                    ('failed', s.tapp_stat.simp_stat.failed_distribution),
                    ('unchanged', s.tapp_stat.simp_stat.unchanged_distribution)
                ])))

  lines.append(
      ('Distribution of invocation times of all the tactics combined in '
       'milliseconds:\n%s\n') % (log_scale_histograms_to_string(
           [('total', s.tapp_stat.overall_stat.total_distribution),
            ('successful', s.tapp_stat.overall_stat.success_distribution),
            ('failed', s.tapp_stat.overall_stat.failed_distribution),
            ('unchanged', s.tapp_stat.overall_stat.unchanged_distribution)])))

  total_per_rank = copy_default(s.tapp_stat.total_per_rank)
  time_per_rank = copy_default(s.tapp_stat.time_per_rank)
  lines.append('How much time was spent per rank (seconds)?\n%s\n' %
               (histograms_to_string(
                   [('seconds', time_per_rank),
                    ('avg s', average_by(time_per_rank, total_per_rank))],
                   scale=.001)))

  total_per_score = copy_default(s.tapp_stat.total_per_score)
  time_per_score = copy_default(s.tapp_stat.time_per_score)
  lines.append(
      'How much time was spent per score (seconds, cutoff -20)?\n%s\n' %
      (histograms_to_string(
          [('seconds', time_per_score),
           ('avg s', average_by(time_per_score, total_per_score))],
          scale=.001)))

  success_per_rank = copy_default(s.tapp_stat.success_per_rank)
  failed_per_rank = copy_default(s.tapp_stat.failed_per_rank)
  unchanged_per_rank = copy_default(s.tapp_stat.unchanged_per_rank)
  closed_per_rank = copy_default(s.tapp_stat.closed_per_rank)
  lines.append('Applications per rank:\n%s\n' % (histograms_to_string(
      [('total', total_per_rank), ('success', success_per_rank),
       ('failed', failed_per_rank), ('unchanged', unchanged_per_rank),
       ('closed', closed_per_rank),
       ('closed frac', average_by(closed_per_rank, total_per_rank))])))

  success_per_score = copy_default(s.tapp_stat.success_per_score)
  failed_per_score = copy_default(s.tapp_stat.failed_per_score)
  unchanged_per_score = copy_default(s.tapp_stat.unchanged_per_score)
  closed_per_score = copy_default(s.tapp_stat.closed_per_score)
  lines.append(
      'Applications per score (cutoff -20):\n%s\n' % (histograms_to_string(
          [('total', total_per_score), ('success', success_per_score),
           ('failed', failed_per_score), ('unchanged', unchanged_per_score),
           ('closed', closed_per_score),
           ('closed frac', average_by(closed_per_score, total_per_score))])))

  premise_usages = list(s.premise_usages.items())
  premise_usages.sort(key=lambda x: x[1])
  lines.append('Most commonly used premises:')
  # Print the 10 most used premises in descending order
  for fingerprint, usages in premise_usages[:-10:-1]:
    lines.append('  %d: %d' % (fingerprint, usages))

  lines.append('\nFingerprints of goals where proof length is >%d' %
               LONG_THEOREM_LEN)
  for fingerprint, proof_length in s.long_proofs_fp.items():
    lines.append('  %d: %d' % (fingerprint, proof_length))

  return '\n'.join(lines)


def proof_log_labels(proof_log: deephol_pb2.ProofLog) -> List[Text]:
  """Determines the buckets this proof log is aggregated with for statistics."""
  labels = [OVERALL_LABEL]
  if FLAGS.stats_per_library_tag:
    labels.extend(proof_log.theorem_in_database.library_tag)
    for node in proof_log.nodes:
      labels.extend(node.goal.library_tag)
  return list(sorted(set(labels)))  # removes duplicates
