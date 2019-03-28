r""""DeepHOL non-distributed prover.

Usage examples:
  bazel run -c opt :main -- --alsologtostderr \
    --prover_options=prover_options.pbtxt \
    --output=${HOME}/deephol_out/proofs.textpbs \
"""

from __future__ import absolute_import
from __future__ import division
# Import Type Annotations
from __future__ import print_function

import tensorflow as tf

from deepmath.deephol import prover_flags
from deepmath.deephol import prover_runner


def main(argv):
  prover_runner.program_started()
  if len(argv) > 1:
    raise Exception('Too many command-line arguments.')
  prover_runner.run_pipeline(*prover_flags.process_prover_flags())


if __name__ == '__main__':
  tf.app.run(main)
