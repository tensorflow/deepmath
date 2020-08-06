r""""DeepHOL non-distributed prover.

Usage examples:
  bazel run -c opt :main -- --alsologtostderr \
    --prover_options=prover_options.pbtxt \
    --output=${HOME}/deephol_out/proof_log.textpbs \
    --splits=validation
"""
import tensorflow.compat.v1 as tf
from deepmath.deephol import prover_flags
from deepmath.deephol import prover_runner


def main(argv):
  prover_runner.program_started()
  if len(argv) > 1:
    raise Exception('Too many command-line arguments.')
  prover_configurations = prover_flags.process_prover_flags()
  if len(prover_configurations) != 1:
    tf.logging.fatal('Need exactly one prover options file; given %s',
                     tf.flags.FLAGS.prover_options)
  prover_tasks, options, _ = prover_configurations[0]
  prover_runner.run_pipeline(prover_tasks, options, tf.flags.FLAGS.output)


if __name__ == '__main__':
  tf.app.run(main)
