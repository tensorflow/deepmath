r""""DeepHOL large scale reporting in Apache Beam."""

from __future__ import absolute_import
from __future__ import division
# Import Type Annotations
from __future__ import print_function
import io
import os
import apache_beam as beam
from apache_beam.metrics import Metrics
import matplotlib.pyplot as plot
import tensorflow as tf
from tf import gfile
from typing import List
from typing import Text
from google.protobuf import text_format
from deepmath.deephol import deephol_pb2
from deepmath.deephol import io_util
from deepmath.deephol.deephol_loop.missing import recordio
from deepmath.deephol.deephol_loop.missing import runner
from deepmath.deephol.utilities import deephol_stat_pb2
from deepmath.deephol.utilities import stats

STATS_BASENAME = 'proof_stats'
AGGREGATE_STAT_BASENAME = 'aggregate_stat'
PROVEN_GOALS_BASENAME = 'proven_goals_fps'
OPEN_GOALS_BASENAME = 'open_goals_fps'
PROVEN_STATS_BASENAME = 'proven_stats'
PRETTY_STATS_BASENAME = 'pretty_stats'
CACTUS_PLOT_FILE_NAME = 'cactus.pdf'
CACTUS_DATA_FILE_NAME = 'cactus.dat'


class StatDoFn(beam.DoFn):
  """Beam DoFn for statistics generation."""

  def __init__(self):
    self.processed_counter = Metrics.counter(self.__class__, 'processed')
    self.proven_counter = Metrics.counter(self.__class__, 'proven')
    self.attempted_counter = Metrics.counter(self.__class__, 'attempted')
    self.nodes_counter = Metrics.counter(self.__class__, 'nodes')

  def process(self, proof_log: deephol_pb2.ProofLog
             ) -> List[deephol_stat_pb2.ProofStat]:
    self.processed_counter.inc()
    s = stats.proof_log_stats(proof_log)
    self.proven_counter.inc(s.num_theorems_proved)
    self.attempted_counter.inc(s.num_theorems_attempted)
    self.nodes_counter.inc(s.num_nodes)
    return [s]


class AggregateStatsFn(beam.CombineFn):
  """Beam CombineFn for statistics aggregation."""

  def create_accumulator(self):
    return deephol_stat_pb2.ProofAggregateStat()

  def add_input(self, target, source):
    stats.merge_stat(target, source)
    return target

  def merge_accumulators(self, aggregate_stats):
    result = deephol_stat_pb2.ProofAggregateStat()
    for s in aggregate_stats:
      stats.merge_aggregate_stat(result, s)
    return result

  def extract_output(self, result):
    return result


class UniqueFn(beam.CombineFn):
  """De-duping combinator for Beam."""

  def create_accumulator(self):
    return set()

  def add_input(self, target, source):
    target.add(source)
    return target

  def merge_accumulators(self, sets):
    result = set()
    for s in sets:
      result.update(s)
    return result

  def extract_output(self, result):
    return '\n'.join([str(x) for x in result])


def proven_or_open(proof_stat):
  if proof_stat.num_theorems_proved > 0:
    yield beam.pvalue.TaggedOutput('proven',
                                   '%d' % proof_stat.theorem_fingerprint)
  else:
    yield beam.pvalue.TaggedOutput('open',
                                   '%d' % proof_stat.theorem_fingerprint)


def make_proof_logs_collection(root, proof_logs: Text):
  return (root | 'Create' >> recordio.ReadFromRecordIO(
      proof_logs, beam.coders.ProtoCoder(deephol_pb2.ProofLog)))


def reporting_pipeline(proof_logs_collection, stats_out: Text,
                       aggregate_stats: Text, proven_goals: Text,
                       open_goals: Text):
  """A pipeline reporting aggregate statistics and proved theorems.

  Args:
    proof_logs_collection: beam collection of proof logs.
    stats_out: Filename for outputting per proof statistics.
    aggregate_stats: Filename for storing aggregated statistics
    proven_goals: Filename for the fingerprint of proven goals.
    open_goals: Filename for the fingerprint of open goals.

  Returns:
    A beam pipeline for writing statistics.
  """
  proof_stats = (proof_logs_collection | 'Stats' >> beam.ParDo(StatDoFn()))
  _ = proof_stats | 'WriteStats' >> recordio.WriteToRecordIO(
      file_path_prefix=stats_out,
      coder=beam.coders.ProtoCoder(deephol_stat_pb2.ProofStat))
  _ = (
      proof_stats
      | 'AggregateStats' >> beam.CombineGlobally(AggregateStatsFn())
      | 'MapProtoToString' >> beam.Map(text_format.MessageToString)
      | 'WriteAggregates' >> beam.io.WriteToText(aggregate_stats, '.pbtxt'))
  results = proof_stats | (
      'ProvenOrOpen' >> beam.FlatMap(proven_or_open).with_outputs())
  _ = (
      results.proven
      | 'UniqueProven' >> beam.CombineGlobally(UniqueFn())
      | 'WriteProven' >> beam.io.WriteToText(proven_goals, '.txt'))
  _ = (
      results.open
      | 'UniqueOpen' >> beam.CombineGlobally(UniqueFn())
      | 'WriteOpen' >> beam.io.WriteToText(open_goals, '.txt'))


def file_lines_set(fname):
  with gfile.Open(fname) as f:
    return set([line.rstrip() for line in f])


class ReportingPipeline(object):
  """Top level class to manage a reporting pipeline."""

  def __init__(self, out_dir: Text):
    self.out_dir = out_dir
    gfile.MakeDirs(out_dir)
    self.proof_stats_filename = os.path.join(out_dir, STATS_BASENAME)
    self.aggregate_stat_filename = os.path.join(out_dir,
                                                AGGREGATE_STAT_BASENAME)
    self.proven_goals_filename = os.path.join(out_dir, PROVEN_GOALS_BASENAME)
    self.open_goals_filename = os.path.join(out_dir, OPEN_GOALS_BASENAME)
    self.proven_stats_filename = os.path.join(out_dir, PROVEN_STATS_BASENAME)
    self.pretty_stats_filename = os.path.join(out_dir, PRETTY_STATS_BASENAME)
    self.cactus_plot_filename = os.path.join(out_dir, CACTUS_PLOT_FILE_NAME)
    self.cactus_data_filename = os.path.join(out_dir, CACTUS_DATA_FILE_NAME)

  def setup_pipeline(self, proof_logs_collection):
    reporting_pipeline(proof_logs_collection, self.proof_stats_filename,
                       self.aggregate_stat_filename, self.proven_goals_filename,
                       self.open_goals_filename)

  def write_final_stats(self):
    """Log and write final aggregated statistics to file system."""
    fname = self.aggregate_stat_filename + '-00000-of-00001.pbtxt'
    aggregate_stat = io_util.load_text_proto(
        fname, deephol_stat_pb2.ProofAggregateStat, 'aggregate statistics')
    if aggregate_stat is None:
      tf.logging.warning('Could not read aggregate statistics "%s"', fname)
      return
    tf.logging.info('Stats:\n%s',
                    stats.aggregate_stat_to_string(aggregate_stat))
    open_goals = file_lines_set(self.open_goals_filename +
                                '-00000-of-00001.txt')
    proven_goals = file_lines_set(self.proven_goals_filename +
                                  '-00000-of-00001.txt')
    never_proven = open_goals - proven_goals
    num_open_goals = len(never_proven)
    num_proven_goals = len(proven_goals)
    tf.logging.info('Open goals: %d', num_open_goals)
    tf.logging.info('Proved goals: %d', num_proven_goals)
    perc_proven = 100.0 * num_proven_goals / float(num_open_goals +
                                                   num_proven_goals)
    tf.logging.info('Percentage proven: %.2f', perc_proven)
    with gfile.Open(self.proven_stats_filename, 'w') as f:
      f.write('%d %d %.2f\n' % (num_open_goals, num_proven_goals, perc_proven))
    with gfile.Open(self.pretty_stats_filename, 'w') as f:
      f.write('%s\n' % stats.detailed_statistics(aggregate_stat))

    # Write cactus plot
    if aggregate_stat.proof_closed_after_millis:
      cactus_data = list(aggregate_stat.proof_closed_after_millis)
      cactus_data.sort()
      with gfile.Open(self.cactus_data_filename, 'w') as f:
        f.write('\n'.join(map(str, cactus_data)))
      fig = plot.figure()
      plot.xlabel('Number of proofs closed')
      plot.ylabel('Wall clock time in s')
      plot.plot([ms * .001 for ms in cactus_data])  # convert to seconds
      buf = io.BytesIO()
      fig.savefig(buf, format='pdf', bbox_inches='tight')
      with gfile.Open(self.cactus_plot_filename, 'wb') as f:
        f.write(buf.getvalue())

  def run_pipeline(self, proof_logs: Text):

    def pipeline(root):
      proof_logs_collection = make_proof_logs_collection(root, proof_logs)
      self.setup_pipeline(proof_logs_collection)

    runner.Runner().run(pipeline).wait_until_finish()
    self.write_final_stats()
    tf.logging.info('Finished reporting.')
