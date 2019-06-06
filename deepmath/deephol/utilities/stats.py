"""Statistics about various DeepHOL objects, especially ProofLogs.

In the first iteration this library can be used for generating statistics
over ProofLogs.
"""

from __future__ import absolute_import
from __future__ import division
# Import Type Annotations
from __future__ import print_function

import itertools
import math
import tensorflow as tf
from typing import List, Text, Tuple
from deepmath.deephol import deephol_pb2
from deepmath.deephol import theorem_fingerprint
from deepmath.deephol.utilities import deephol_stat_pb2
from deepmath.deephol.utilities import proof_analysis
from deepmath.proof_assistant import proof_assistant_pb2


def add_value_to_log_scale_histogram(hist: deephol_stat_pb2.LogScaleHistogram,
                                     key: int, value: int):
  assert key >= 0
  log_scale_key = int(math.floor(math.log(key + 1, 2)))
  if log_scale_key not in hist.h:
    hist.h[log_scale_key] = 0
  hist.h[log_scale_key] += value


def merge_log_scale_histograms(target, source):
  for key, value in source.h.items():
    add_value_to_log_scale_histogram(target, 2**key, value)


def merge_histograms(target, source):
  for key, value in source.items():
    target[key] += value


def tactic_time_stats(stat: deephol_stat_pb2.TacticTimeStat,
                      tapp: deephol_pb2.TacticApplication):
  """Update a TacticTimeStat by a single TacticApplication."""
  stat.total_time += tapp.time_spent
  add_value_to_log_scale_histogram(stat.total_distribution, tapp.time_spent, 1)
  if tapp.result == deephol_pb2.TacticApplication.SUCCESS:
    add_value_to_log_scale_histogram(stat.success_distribution, tapp.time_spent,
                                     1)
  if tapp.result == deephol_pb2.TacticApplication.ERROR:
    add_value_to_log_scale_histogram(stat.failed_distribution, tapp.time_spent,
                                     1)
  if tapp.result == deephol_pb2.TacticApplication.UNCHANGED:
    add_value_to_log_scale_histogram(stat.unchanged_distribution,
                                     tapp.time_spent, 1)


def merge_tactic_time_stats(target: deephol_stat_pb2.TacticTimeStat,
                            source: deephol_stat_pb2.TacticTimeStat):
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
  if tapp.closed:
    stat.closed_applications_per_tactic[tactic] += 1
    stat.closed_per_rank[tapp.rank] += 1
    stat.closed_per_score[score] += 1


def merge_proof_tapp_stats(target: deephol_stat_pb2.TacticApplicationStat,
                           source: deephol_stat_pb2.TacticApplicationStat):
  """Merges ProofStep-level statistics."""
  merge_histograms(target.time_spent_per_tapp_result,
                   source.time_spent_per_tapp_result)
  merge_histograms(target.time_spent_per_tactic, source.time_spent_per_tactic)
  merge_histograms(target.total_tactic_applications_per_tactic,
                   source.total_tactic_applications_per_tactic)
  merge_histograms(target.successful_tactic_applications_per_tactic,
                   source.successful_tactic_applications_per_tactic)
  merge_histograms(target.unchanged_tactic_applications_per_tactic,
                   source.unchanged_tactic_applications_per_tactic)
  merge_histograms(target.failed_tactic_applications_per_tactic,
                   source.failed_tactic_applications_per_tactic)
  merge_histograms(target.unknown_tactic_applications_per_tactic,
                   source.unknown_tactic_applications_per_tactic)
  merge_histograms(target.closing_tactic_applications_per_tactic,
                   source.closing_tactic_applications_per_tactic)
  merge_histograms(target.closed_applications_per_tactic,
                   source.closed_applications_per_tactic)
  merge_tactic_time_stats(target.meson_stat, source.meson_stat)
  merge_tactic_time_stats(target.rewrite_stat, source.rewrite_stat)
  merge_tactic_time_stats(target.simp_stat, source.simp_stat)

  merge_histograms(target.time_per_rank, source.time_per_rank)
  merge_histograms(target.total_per_rank, source.total_per_rank)
  merge_histograms(target.success_per_rank, source.success_per_rank)
  merge_histograms(target.failed_per_rank, source.failed_per_rank)
  merge_histograms(target.unchanged_per_rank, source.unchanged_per_rank)
  merge_histograms(target.closed_per_rank, source.closed_per_rank)

  merge_histograms(target.time_per_score, source.time_per_score)
  merge_histograms(target.total_per_score, source.total_per_score)
  merge_histograms(target.success_per_score, source.success_per_score)
  merge_histograms(target.failed_per_score, source.failed_per_score)
  merge_histograms(target.unchanged_per_score, source.unchanged_per_score)
  merge_histograms(target.closed_per_score, source.closed_per_score)


def proof_log_stats(proof_log: deephol_pb2.ProofLog
                   ) -> deephol_stat_pb2.ProofStat:
  """Create statistics for a single proof log."""
  num_attempted = 0
  num_proved = 0
  num_bad_proofs = 0
  closed_node_indices = []
  fingerprint = None
  reduced_node_indices = []
  for i, node in enumerate(proof_log.nodes):
    if node.goal.tag == proof_assistant_pb2.Theorem.THEOREM:
      num_attempted += 1
      if node.status == deephol_pb2.ProofNode.PROVED:
        num_proved += 1
      if fingerprint is None:
        fingerprint = theorem_fingerprint.Fingerprint(node.goal)
    if node.status == deephol_pb2.ProofNode.PROVED:
      closed_node_indices.append(i)
  if num_proved > 0:
    analysis_result = proof_analysis.find_reasons(proof_log)
    if analysis_result is None:
      num_bad_proofs += 1
      num_proved = 0
    else:
      _, reduced_node_indices = analysis_result
  stat = deephol_stat_pb2.ProofStat(
      num_theorems_attempted=num_attempted,
      num_theorems_proved=num_proved,
      num_theorems_with_bad_proof=num_bad_proofs,
      num_nodes=len(proof_log.nodes),
      reduced_node_indices=reduced_node_indices,
      closed_node_indices=closed_node_indices)
  if proof_log.HasField('time_spent'):
    stat.time_spent_milliseconds = proof_log.time_spent
  if num_attempted == 1 and fingerprint is not None:
    stat.theorem_fingerprint = fingerprint

  for node in proof_log.nodes:
    stat.total_prediction_time += node.action_generation_time_millisec
    add_value_to_log_scale_histogram(stat.node_prediction_time_histogram,
                                     node.action_generation_time_millisec, 1)
    for tapp in node.proofs:
      tactic_application_stats(stat.tapp_stat, tapp)

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
                                   source.time_spent_milliseconds, 1)
  if source.num_theorems_proved:
    success_hist = target.proof_time_histogram_proved
    target.num_reduced_nodes_distribution[len(source.reduced_node_indices)] += 1
  else:
    success_hist = target.proof_time_histogram_failed
  add_value_to_log_scale_histogram(success_hist, source.time_spent_milliseconds,
                                   1)
  merge_proof_tapp_stats(target.tapp_stat, source.tapp_stat)
  target.total_prediction_time += source.total_prediction_time
  add_value_to_log_scale_histogram(target.proof_prediction_time_histogram,
                                   source.total_prediction_time, 1)
  merge_log_scale_histograms(target.node_prediction_time_histogram,
                             source.node_prediction_time_histogram)
  if source.HasField('time_spent_milliseconds') and source.num_theorems_proved:
    if source.num_theorems_proved != 1:
      tf.logging.error('More than one proof in single proof log; Cactus plot '
                       'will be incaccurate.')
    target.proof_closed_after_millis.append(source.time_spent_milliseconds)


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
  merge_histograms(target.num_reduced_nodes_distribution,
                   source.num_reduced_nodes_distribution)
  target.total_prediction_time += source.total_prediction_time
  merge_log_scale_histograms(target.proof_prediction_time_histogram,
                             source.proof_prediction_time_histogram)
  merge_log_scale_histograms(target.node_prediction_time_histogram,
                             source.node_prediction_time_histogram)
  target.proof_closed_after_millis.extend(source.proof_closed_after_millis)


def aggregate_stats(stats: List[deephol_stat_pb2.ProofStat]
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


def log_scale_histograms_to_string(
    histograms: List[Tuple[Text, deephol_stat_pb2.LogScaleHistogram]],
    scale=1) -> Text:
  """Converts histogram into string; missing values are interpreted as 0."""
  header = ['ms    '.ljust(6)] + [s.rjust(12) for s, _ in histograms]
  lines = [''.join(header)]
  entry_nums = [len(hist.h) for _, hist in histograms]
  found_non_zero_entries = False  # suppresses initial rows that are all zeros
  index = 0
  while any(entry_nums):
    entries = []
    for hist_idx, (_, hist) in enumerate(histograms):
      if index in hist.h:
        entry_nums[hist_idx] -= 1
      entries.append(hist.h.get(index, 0) * scale)
    if found_non_zero_entries or any(entries):
      found_non_zero_entries = True
      lines.append(('2**%d:' % index).ljust(6) +
                   ''.join([str(e).rjust(12) for e in entries]))
    index += 1
  return '\n'.join(lines)


def histograms_to_string(maps_map, scale=1) -> Text:
  """Turns a list of names protobuf maps into a nice table."""
  column_size = 12
  all_keys = list(
      set(itertools.chain.from_iterable([m.keys() for _, m in maps_map])))
  all_keys.sort()
  key_lengths = list(map(len, map(str, all_keys)))
  key_lengths.append(8)
  longest_key = max(key_lengths) + 1
  header = [''.ljust(longest_key)] + [s.rjust(column_size) for s, _ in maps_map]
  lines = [''.join(header)]
  for key in all_keys:

    def nice_string(v):
      if isinstance(v, float):
        res = '%.3f' % v
      else:
        res = str(v)
      return res.rjust(column_size)

    entries = [nice_string(m[key] * scale) for _, m in maps_map]
    lines.append(str(key).ljust(longest_key) + ''.join(entries))
  return str('\n'.join(lines))


def average_by(time, applications):
  return {
      key: time[key] / app_num
      for key, app_num in applications.items()
      if app_num > 0
  }


def detailed_statistics(s: deephol_stat_pb2.ProofAggregateStat) -> Text:
  """Converts histograms into pretty tables."""

  res = 'Histograms: \n'
  res += 'How much tactic application time did proofs take?\n%s\n\n' % (
      log_scale_histograms_to_string(
          [('success', s.proof_time_histogram_proved),
           ('failed', s.proof_time_histogram_failed)]))

  res += 'How much prediction time did proofs take?\n%s\n\n' % (
      log_scale_histograms_to_string(
          [('per proof', s.proof_prediction_time_histogram),
           ('per node', s.node_prediction_time_histogram)]))

  res += 'Histograms: \nHow long are the proofs?\n%s\n\n' % (
      histograms_to_string([('#proofs', s.num_reduced_nodes_distribution)]))

  res += 'How much time was spent per TacticApplication.Result?\n%s\n\n' % (
      histograms_to_string(
          [('seconds', s.tapp_stat.time_spent_per_tapp_result)], scale=.001))

  avg_tapp_time = average_by(s.tapp_stat.time_spent_per_tactic,
                             s.tapp_stat.total_tactic_applications_per_tactic)
  res += 'How much time was spent per tactic (seconds)?\n%s\n\n' % (
      histograms_to_string([('seconds', s.tapp_stat.time_spent_per_tactic),
                            ('avg s', avg_tapp_time)],
                           scale=.001))

  res += 'Outcome of tactic applications per tactic:\n%s\n\n' % (
      histograms_to_string([
          ('total', s.tapp_stat.total_tactic_applications_per_tactic),
          ('success', s.tapp_stat.successful_tactic_applications_per_tactic),
          ('failed', s.tapp_stat.failed_tactic_applications_per_tactic),
          ('unchanged', s.tapp_stat.unchanged_tactic_applications_per_tactic),
          ('unknown', s.tapp_stat.unknown_tactic_applications_per_tactic),
          ('closing', s.tapp_stat.closing_tactic_applications_per_tactic),
          ('in closed', s.tapp_stat.closed_applications_per_tactic),
          ('success %',
           average_by(s.tapp_stat.successful_tactic_applications_per_tactic,
                      s.tapp_stat.total_tactic_applications_per_tactic)),
      ]))

  res += ('Distribution of invocation times of ASM_MESON_TAC in '
          'milliseconds:\n%s\n\n') % (
              log_scale_histograms_to_string(
                  [('total', s.tapp_stat.meson_stat.total_distribution),
                   ('successful', s.tapp_stat.meson_stat.success_distribution),
                   ('failed', s.tapp_stat.meson_stat.failed_distribution)]))

  res += ('Distribution of invocation times of REWRITE_TAC in '
          'milliseconds:\n%s\n\n') % (
              log_scale_histograms_to_string([
                  ('total', s.tapp_stat.rewrite_stat.total_distribution),
                  ('successful', s.tapp_stat.rewrite_stat.success_distribution),
                  ('failed', s.tapp_stat.rewrite_stat.failed_distribution),
                  ('unchanged', s.tapp_stat.rewrite_stat.unchanged_distribution)
              ]))

  res += ('Distribution of invocation times of SIMP_TAC in '
          'milliseconds:\n%s\n\n') % (
              log_scale_histograms_to_string([
                  ('total', s.tapp_stat.simp_stat.total_distribution),
                  ('successful', s.tapp_stat.simp_stat.success_distribution),
                  ('failed', s.tapp_stat.simp_stat.failed_distribution),
                  ('unchanged', s.tapp_stat.simp_stat.unchanged_distribution)
              ]))

  res += 'How much time was spent per rank (seconds)?\n%s\n\n' % (
      histograms_to_string(
          [('seconds', s.tapp_stat.time_per_rank),
           ('avg s',
            average_by(s.tapp_stat.time_per_rank, s.tapp_stat.total_per_rank))],
          scale=.001))

  res += 'How much time was spent per score (seconds, cutoff -20)?\n%s\n\n' % (
      histograms_to_string([('seconds', s.tapp_stat.time_per_score),
                            ('avg s',
                             average_by(s.tapp_stat.time_per_score,
                                        s.tapp_stat.total_per_score))],
                           scale=.001))

  res += 'Applications per rank:\n%s\n\n' % (
      histograms_to_string([('total', s.tapp_stat.total_per_rank),
                            ('success', s.tapp_stat.success_per_rank),
                            ('failed', s.tapp_stat.failed_per_rank),
                            ('unchanged', s.tapp_stat.unchanged_per_rank),
                            ('closed', s.tapp_stat.closed_per_rank),
                            ('closed %',
                             average_by(s.tapp_stat.closed_per_rank,
                                        s.tapp_stat.total_per_rank))]))

  res += 'Applications per score (cutoff -20):\n%s\n\n' % (
      histograms_to_string([('total', s.tapp_stat.total_per_score),
                            ('success', s.tapp_stat.success_per_score),
                            ('failed', s.tapp_stat.failed_per_score),
                            ('unchanged', s.tapp_stat.unchanged_per_score),
                            ('closed', s.tapp_stat.closed_per_score),
                            ('closed %',
                             average_by(s.tapp_stat.closed_per_score,
                                        s.tapp_stat.total_per_score))]))
  return res
