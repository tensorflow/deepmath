r"""Print some statistics and optionally a short summary about each proof.

This is just a preliminary executable to examine the proofs produced by the
prover.

Example invocation:
  bazel run :stat_proofs_main -- --alsologtostderr \
    --proof_logs=/path/to/prooflogs.textpbs
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from deepmath.deephol import deephol_pb2
from deepmath.deephol import io_util
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
      for log in io_util.read_protos(FLAGS.proof_logs, deephol_pb2.ProofLog)
  ]
  if not stat_list:
    tf.logging.info('Empty stats list.')
    return
  tf.logging.info('Aggregating statistics')
  aggregate_stat = stats.aggregate_stats(stat_list)
  if FLAGS.verbose:
    for stat in sorted(stat_list, key=lambda s: s.num_nodes):
      tf.logging.info('%s', stats.stat_to_string(stat))
    tf.logging.info(stats.detailed_statistics(aggregate_stat))
  tf.logging.info('Aggregated statistics:')
  tf.logging.info(stats.aggregate_stat_to_string(aggregate_stat))


if __name__ == '__main__':
  tf.app.run(main)
