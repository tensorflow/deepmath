r"""Calls proof_checker_lib to convert proofs to OCaml.

Processes multiple proof logs at a time, but at most one proof per theorem.

  bazel run :proof_checker -- --alsologtostderr \
    --proof_logs=path/to/proof_logs.textpbs \
    --theorem_database=path/to/theorem_database.textpb \
    --out_file=path/to/ocaml_file.ml
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from deepmath.deephol import deephol_pb2
from deepmath.deephol import io_util
from deepmath.deephol.utilities import proof_checker_lib

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('theorem_database', None, 'Path to theorem database.')
tf.flags.DEFINE_string('proof_logs', None, 'Path of proofs to check.')
tf.flags.DEFINE_string('out_file', None, 'Path to write OCaml file to.')


def main(argv):
  if len(argv) > 1:
    raise ValueError('Too many command-line arguments.')
  if FLAGS.theorem_database is None:
    tf.logging.fatal('--theorem_database flag is required.')
  if FLAGS.proof_logs is None:
    tf.logging.fatal('--proof_logs flag is required.')

  theorem_db = io_util.load_theorem_database_from_file(FLAGS.theorem_database)
  proof_logs = io_util.read_protos(FLAGS.proof_logs, deephol_pb2.ProofLog)
  ocaml_code = proof_checker_lib.verify(proof_logs, theorem_db)
  with tf.gfile.Open(FLAGS.out_file, 'w') as f:
    f.write(ocaml_code)
  tf.logging.info('Proof checker successful.')


if __name__ == '__main__':
  tf.app.run(main)
