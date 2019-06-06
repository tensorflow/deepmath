"""Running the loop.

An exacutable runs parts or whole of the theorem proving loop pipeline.
"""
from __future__ import absolute_import
from __future__ import division
# Import Type Annotations
from __future__ import print_function
from tensorflow import app
from tensorflow import flags
from tensorflow import logging
from deepmath.deephol import io_util
from deepmath.deephol.deephol_loop import loop_meta
from deepmath.deephol.deephol_loop import loop_pb2
from deepmath.deephol.deephol_loop import loop_pipeline
from deepmath.deephol.deephol_loop.missing import runner

FLAGS = flags.FLAGS

flags.DEFINE_string('root', None, 'Root directory containing all the loops.')

flags.DEFINE_string('config', '', 'Configuration_file for the loop')

flags.DEFINE_integer('rounds', 10, 'Number of rounds to perform.')

flags.DEFINE_string('initial_examples', None,
                    'Initial examples to copy over during setup.')


def main(argv):
  runner.program_started()
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  assert FLAGS.root is not None, 'Required flag --root is missing.'
  config = io_util.load_text_proto(FLAGS.config, loop_pb2.LoopConfig)
  if (not FLAGS.rounds and not FLAGS.initial_examples and
      not config.inherited_proof_logs):
    logging.fatal('Loop setup requires either initial examples '
                  'or inherited proof logs')
  controller_fingerprint = loop_meta.create_fingerprint()
  meta = loop_meta.LoopMeta(FLAGS.root, config, controller_fingerprint, False)
  assert meta.status, 'Could not read status'
  loop = loop_pipeline.LoopPipeline(meta, config)
  if not FLAGS.rounds:
    logging.info('Setting up loop...')
    loop.setup_examples(FLAGS.initial_examples)
  else:
    for _ in xrange(FLAGS.rounds):
      loop.perform_round(FLAGS.initial_examples)


if __name__ == '__main__':
  app.run(main)
