"""Simple runner for the prover.

Iterate over the tasks sequentially.

This runner can run the prover on a set of tasks without the overhead of
starting a distributed job.
"""

from __future__ import absolute_import
from __future__ import division
# Import Type Annotations
from __future__ import print_function

import tensorflow as tf
from typing import List
from typing import Text
from deepmath.public import build_data
from deepmath.deephol import deephol_pb2
from deepmath.deephol import io_util
from deepmath.deephol import prover
from deepmath.deephol.utilities import stats
from deepmath.proof_assistant import proof_assistant_pb2

FLAGS = tf.flags.FLAGS


def program_started():
  pass


def compute_stats(output):
  """Compute aggregate statistics given prooflog file."""
  tf.logging.info('Computing aggregate statistics from %s', output)
  stat_list = [
      stats.proof_log_stats(log)
      for log in io_util.read_protos(output, deephol_pb2.ProofLog)
  ]
  if not stat_list:
    tf.logging.info('Empty stats list.')
    return
  aggregate_stat = stats.aggregate_stats(stat_list)
  tf.logging.info('Aggregated statistics:')
  tf.logging.info(stats.aggregate_stat_to_string(aggregate_stat))


def run_pipeline(prover_tasks: List[proof_assistant_pb2.ProverTask],
                 prover_options: deephol_pb2.ProverOptions, path_output: Text):
  """Iterate over all prover tasks and store them in the specified file."""
  if FLAGS.output.split('.')[-1] != 'textpbs':
    tf.logging.warn('Output file should end in ".textpbs"')

  prover.cache_embeddings(prover_options)
  this_prover = prover.create_prover(prover_options)
  proof_logs = []
  for task in prover_tasks:
    proof_log = this_prover.prove(task)
    proof_log.build_data = build_data.BuildData()
    proof_logs.append(proof_log)
  if path_output:
    tf.logging.info('Writing %d proof logs as text proto to %s',
                    len(proof_logs), path_output)
    io_util.write_text_protos(path_output, proof_logs)

  tf.logging.info('Proving complete!')
  compute_stats(FLAGS.output)
