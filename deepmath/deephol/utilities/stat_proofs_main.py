r"""Print some statistics and optionally a short summary about each proof.

This is just a preliminary executable to examine the proofs produced by the
prover.

Example invocation:
  bazel run :stat_proofs_main -- --alsologtostderr \
    --proof_logs=/path/to/prooflogs.textpbs
"""
import tensorflow.compat.v1 as tf
from deepmath.deephol import io_util
from deepmath.deephol.utilities import deephol_stat_pb2
from deepmath.deephol.utilities import stats

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string(
    'proof_logs', None, 'Required: a pattern desribing the input proof logs. '
    'Can be a comma separated list of globs or recordio '
    'patterns.')

tf.flags.DEFINE_boolean(
    'verbose', None, 'If true, then individual proof statistics are reported')


def main(argv):
  tf.logging.info('Computing proof statistics!')
  if len(argv) > 1:
    raise ValueError('Too many command-line arguments.')
  assert FLAGS.proof_logs
  stat_list = [
      stats.proof_log_stats(log)
      for log in io_util.read_proof_logs(FLAGS.proof_logs)
  ]
  if not stat_list:
    tf.logging.info('Empty stats list.')
    return

  tf.logging.info('Aggregating statistics')
  aggregate_stat = stats.aggregate_stats(stat_list)

  if FLAGS.verbose:
    for stat in sorted(stat_list, key=lambda s: s.num_nodes):
      tf.logging.info('%s', stats.stat_to_string(stat))
    labeled_aggregate = deephol_stat_pb2.LabeledAggregateStats()
    labeled_aggregate.labeled_stats[stats.OVERALL_LABEL] = aggregate_stat
    tf.logging.info(stats.detailed_statistics(labeled_aggregate))

  overall = deephol_stat_pb2.LabeledAggregateStats()

  for stat in stat_list:
    for label in stat.labels:
      if label in overall.labeled_stats:
        stats.merge_stat(overall.labeled_stats[label], stat)
      else:
        overall.labeled_stats[label].CopyFrom(stats.aggregate_stats([stat]))

  tf.logging.info(stats.labeled_agg_stat_to_string(overall))


if __name__ == '__main__':
  tf.app.run(main)
