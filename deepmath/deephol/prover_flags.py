"""Processing of prover flags and inputs."""

from __future__ import absolute_import
from __future__ import division
# Import Type Annotations
from __future__ import print_function
import re
import time
import tensorflow as tf
from google.protobuf import text_format
from deepmath.deephol import deephol_pb2
from deepmath.deephol import io_util
from deepmath.deephol import prover_util
from deepmath.proof_assistant import proof_assistant_pb2

FLAGS = tf.flags.FLAGS
HIST_AVG = deephol_pb2.ProverOptions.HIST_AVG
HIST_CONV = deephol_pb2.ProverOptions.HIST_CONV
HIST_ATT = deephol_pb2.ProverOptions.HIST_ATT

# Note that the flags below override might override some of the fields in
# prover_options, should they be defined.
# This is a required parameter. The prover will fail if it is not provided
# with prover_options.
tf.flags.DEFINE_string(
    'prover_options', None,
    'Required: path to file containing the ProverOptions proto')

tf.flags.DEFINE_string(
    'task_list', None,
    'Optional ProverTaskList text protobuf to specify the theorem proving '
    'tasks. If not supplied then a task list is generated automatically from '
    'the theorem library. The filtering for training_split is in effect for '
    'the goals in task_list as well.')

tf.flags.DEFINE_string(
    'tasks', None,
    'Optional multi-line ProverTask text protobuf or recordio to specify the '
    'theorem proving tasks. Either this or task list or tasks_by_fingerprint '
    'must be specified, otherwise the tasks are generated automatically from '
    'the theorem library. The filtering for training_split is in effect for '
    'the goals in the read tasks as well.')

tf.flags.DEFINE_string(
    'tasks_by_fingerprint', None,
    'Optional comma-separated list of fingerprints of theorems in the theorem '
    'database. No filtering by training_split in place.')

tf.flags.DEFINE_string(
    'splits', None,
    'Specifies which examples to run on. Either "all" or comma separated list '
    'of {"training, "testing" and "validation"} This setting overrides the '
    'related setting in the ProverOptions protobuf.')

tf.flags.DEFINE_string(
    'libraries', None,
    'Specifies which examples to run on. Either "all" or comma separated list '
    'of library tags. This setting overrides the related setting in the '
    'ProverOptions protobuf.')

tf.flags.DEFINE_integer(
    'timeout_seconds', None,
    'Override the timeout/task specified in the prover options.')

tf.flags.DEFINE_integer('max_theorem_parameters', None,
                        'Override max_theorem_parameters in prover options.')

# Must be specified from the command line. Not stored in ProverOptions.
tf.flags.DEFINE_string('output', None, 'Path where proof logs are saved.')


def _verify_prover_options(prover_options: deephol_pb2.ProverOptions) -> None:
  """Asserts some (incomplete) consistency requirements over prover_options."""
  for field_name in [
      'path_tactics', 'path_tactics_replace', 'path_theorem_database',
      'path_model_prefix'
  ]:
    if not prover_options.HasField(field_name):
      tf.logging.fatal('Missing field "%s" in ProverOptions', field_name)
  if prover_options.prover not in ['nobacktrack', 'bfs']:
    tf.logging.fatal('Unsupported proof strategy: "%s"', prover_options.prover)

  history_dependent = [HIST_AVG, HIST_CONV, HIST_ATT]
  if prover_options.model_architecture in history_dependent:
    if not prover_options.path_emb_model_prefix:
      tf.logging.fatal(
          'History dependent model %s requires embeddings checkpoint '
          'path_emb_model_prefix.',
          deephol_pb2.ProverOptions.ModelArchitecture.Name(
              prover_options.model_architecture))

    # Light assertions on file naming conventions for embedding consistency.
    # Embedding checkpoint number should be the end of the input file.
    emb_checkpoint_num = next(
        re.finditer(r'\d+$', prover_options.path_emb_model_prefix)).group(0)
    if emb_checkpoint_num not in prover_options.path_model_prefix:
      tf.logging.fatal(
          'Embeddings checkpoint number (%s) was not found '
          'in the path of predictions checkpoint (%s), indicating '
          'it was trained with different embeddings.', emb_checkpoint_num,
          prover_options.path_model_prefix)


def get_prover_options(prover_round_tag='manual',
                       prover_round=-1) -> deephol_pb2.ProverOptions:
  """Returns a ProverOptions proto based on FLAGS."""
  if not FLAGS.prover_options:
    tf.logging.fatal('Mandatory flag --prover_options is not specified.')
  if not tf.gfile.Exists(FLAGS.prover_options):
    tf.logging.fatal('Required prover options file "%s" does not exist.',
                     FLAGS.prover_options)
  prover_options = deephol_pb2.ProverOptions()
  if FLAGS.max_theorem_parameters is not None:
    tf.logging.warning(
        'Overring max_theorem_parameters in prover options to %d.',
        FLAGS.max_theorem_parameters)
    prover_options.action_generator_options.max_theorem_parameters = (
        FLAGS.max_theorem_parameters)
  with tf.gfile.Open(FLAGS.prover_options) as f:
    text_format.MergeLines(f, prover_options)
  if prover_options.builtin_library:
    tf.logging.warning('builtin_library is deprecated. Do not provide.')
    if str(prover_options.builtin_library) not in ['core']:
      tf.logging.fatal('Unsupported built in library: %s',
                       prover_options.builtin_library)
  if FLAGS.timeout_seconds is not None:
    prover_options.timeout_seconds = FLAGS.timeout_seconds
  if not FLAGS.output:
    tf.logging.fatal('Missing flag --output [recordio_pattern]')
    prover_options.prover_round = deephol_pb2.ProverRound(
        start_seconds=int(round(time.time())),
        tag=prover_round_tag,
        round=prover_round)
  _verify_prover_options(prover_options)

  # Log prover options.
  tf.logging.info('Using prover_options:\n %s', str(prover_options))
  return prover_options


def process_prover_flags():
  """Process the flags and return tasks, options and output path."""
  prover_options = get_prover_options()

  if FLAGS.splits:
    tf.logging.info(
        '--splits flag overrides prover options for split selection.')
    splits_to_prove = prover_util.translate_splits(FLAGS.splits)
  else:
    splits_to_prove = list(prover_options.splits_to_prove)
  if not splits_to_prove and not FLAGS.tasks_by_fingerprint:
    tf.logging.fatal('No split specification!')
  tf.logging.info(
      'Splits to prove: %s', ', '.join(
          map(proof_assistant_pb2.Theorem.Split.Name, splits_to_prove)))

  if FLAGS.libraries:
    tf.logging.info(
        '--libraries flag overrides prover options for library_tag selection.')
    if FLAGS.libraries == 'all':
      library_tags = set()
    else:
      library_tags = set([tag for tag in FLAGS.libraries.split(',')])
  else:
    library_tags = set(prover_options.library_tags)
  if not library_tags:
    tf.logging.info('Disregarding library tags.')
  else:
    tf.logging.info('Library tags to prove: %s', ', '.join(
        sorted(list(library_tags))))

  # Fail fast in case error in specifying tactics.
  _ = io_util.load_tactics_from_file(
      str(prover_options.path_tactics),
      str(prover_options.path_tactics_replace))
  theorem_db = io_util.load_theorem_database_from_file(
      str(prover_options.path_theorem_database))
  if not theorem_db.HasField('name'):
    theorem_db.name = 'default'  # Set a dummy name for backwards compatibility
    tf.logging.warning('Missing theorem database name is set to %s',
                       theorem_db.name)
  if FLAGS.task_list and FLAGS.tasks:
    tf.logging.fatal('Only one of --tasks or --task_list is allowed.')
  prover_tasks = prover_util.get_task_list(FLAGS.tasks, FLAGS.task_list,
                                           FLAGS.tasks_by_fingerprint,
                                           theorem_db, splits_to_prove,
                                           library_tags)
  # TODO(szegedy): Verify tasks that they all fit the theorem database(s)
  tf.logging.info('Number of prover tasks: %d', len(prover_tasks))
  return (prover_tasks, prover_options, FLAGS.output)
