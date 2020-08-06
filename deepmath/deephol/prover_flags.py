"""Processing of prover flags and inputs."""
from typing import List
import tensorflow.compat.v1 as tf
from deepmath.deephol import deephol_pb2
from deepmath.deephol import io_util
from deepmath.deephol import prover_util
from deepmath.public import prover_options_helper
from deepmath.proof_assistant import proof_assistant_pb2

load_training_options = prover_options_helper.load_training_options
FLAGS = tf.flags.FLAGS

# Note that the flags below override might override some of the fields in
# prover_options, should they be defined.
# This is a required parameter. The prover will fail if it is not provided
# with prover_options.
tf.flags.DEFINE_string(
    'prover_options', None,
    'Required: comma-separated list of paths to ProverOptions protos. When '
    'multiple prover options are specified, all prover runs are merged into a '
    'single beam job.')

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
    'tasks_by_fingerprint_file', None,
    'Optional text file containing fingerprints from theorem database, one-per-line.'
)

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

tf.flags.DEFINE_string(
    'run_with_default_prover_options', None,
    'Comma-separated list of paths to checkpoints to run with the default '
    'prover options.')


def verify_prover_options(prover_options: deephol_pb2.ProverOptions,
                          loop: bool) -> None:
  """Asserts some (incomplete) consistency requirements over prover_options."""
  required_fields = [
      'path_tactics', 'path_tactics_replace', 'path_theorem_database'
  ]
  if not loop:
    required_fields.append('path_model_prefix')
  for field_name in required_fields:
    if not prover_options.HasField(field_name):
      tf.logging.fatal('Missing field "%s" in ProverOptions', field_name)
  if prover_options.prover not in ['nobacktrack', 'bfs', 'mcts']:
    tf.logging.fatal('Unsupported proof strategy: "%s"', prover_options.prover)
  if prover_options.prover == 'bfs':
    if not prover_options.HasField('bfs_options'):
      tf.logging.fatal('Missing field bfs_options in prover options.')
    if (not prover_options.bfs_options.HasField('max_explored_nodes') and
        not prover_options.bfs_options.meta_options.HasField(
            'max_explored_nodes')):
      tf.logging.fatal('Missing field max_explored_nodes in bfs options.')


def process_prover_options(prover_options: deephol_pb2.ProverOptions):
  """Load the checkpoint and verify the consistency of the options."""
  if FLAGS.max_theorem_parameters is not None:
    tf.logging.warning(
        'Overring max_theorem_parameters in prover options to %d.',
        FLAGS.max_theorem_parameters)
    prover_options.action_generator_options.max_theorem_parameters = (
        FLAGS.max_theorem_parameters)
  if prover_options.builtin_library:
    tf.logging.warning('builtin_library is deprecated. Do not provide.')
    if str(prover_options.builtin_library) not in ['core']:
      tf.logging.fatal('Unsupported built in library: %s',
                       prover_options.builtin_library)
  if FLAGS.timeout_seconds is not None:
    prover_options.timeout_seconds = FLAGS.timeout_seconds

  load_training_options(prover_options)

  verify_prover_options(prover_options, loop=False)

  # Log prover options.
  tf.logging.info('Processed prover_options:\n %s', prover_options)


def get_prover_options() -> List[deephol_pb2.ProverOptions]:
  """Loads the prover options specified by the flags."""
  if not (FLAGS.prover_options or FLAGS.run_with_default_prover_options):
    tf.logging.fatal(
        '--prover_options or --run_with_default_prover_options required.')
  list_of_prover_options = []
  if FLAGS.prover_options:
    options_paths = FLAGS.prover_options.split(',')
    tf.logging.info('Reading the following prover options: %s', options_paths)
    for pattern in options_paths:
      paths = tf.gfile.Glob(pattern)
      if not paths:
        tf.logging.fatal('Required prover options file "%s" does not exist.',
                         pattern)
      for path in paths:
        prover_options = io_util.load_text_proto(path,
                                                 deephol_pb2.ProverOptions)
        process_prover_options(prover_options)
        list_of_prover_options.append(prover_options)

  if FLAGS.run_with_default_prover_options:
    model_paths = FLAGS.run_with_default_prover_options.split(',')
    for model_path in model_paths:
      prover_options = prover_options_helper.get_default_prover_options(
          model_path)
      process_prover_options(prover_options)
      list_of_prover_options.append(prover_options)
  if not list_of_prover_options:
    raise ValueError('0 prover options found after processing flags. Failing.')
  tf.logging.info('Number of prover options: %d', len(list_of_prover_options))

  return list_of_prover_options


def _unique_prover_names(options: List[deephol_pb2.ProverOptions]):
  """Make sure each run has a unique name, even when we forgot to specify one."""
  if len(options) == 1:
    if not options[0].HasField('prover_run_subdirectory'):
      options[0].prover_run_subdirectory = ''
    return
  for idx, o in enumerate(options):
    if not o.HasField('prover_run_subdirectory'):
      o.prover_run_subdirectory = '%05d' % idx
  name_set = set([o.prover_run_subdirectory for o in options])
  if len(name_set) < len(options):
    raise ValueError('Collisions in naming scheme for prover runs. Either give '
                     'unique values for prover_options.prover_run_subdirectory '
                     'or leave the field blank.')


def process_prover_flags():
  """Process the flags and return tasks, options and output path."""

  prover_options_list = get_prover_options()
  assert prover_options_list  # non-empty list of ProverOptions protos

  # Avoid loading files for each options file
  theorem_dbs = dict()
  tactic_paths = dict()

  prover_configurations = []
  _unique_prover_names(prover_options_list)
  for options in prover_options_list:
    if FLAGS.splits:
      tf.logging.info(
          '--splits flag overrides prover options for split selection.')
      splits_to_prove = prover_util.translate_splits(FLAGS.splits)
    else:
      splits_to_prove = list(options.splits_to_prove)
    if (not splits_to_prove and not FLAGS.tasks_by_fingerprint and
        not FLAGS.tasks_by_fingerprint_file):
      tf.logging.fatal('No split specification!')
    tf.logging.info(
        'Splits to prove: %s',
        ', '.join(map(proof_assistant_pb2.Theorem.Split.Name, splits_to_prove)))

    if FLAGS.libraries:
      tf.logging.info(
          '--libraries flag overrides prover options for library_tag selection.'
      )
      if FLAGS.libraries == 'all':
        library_tags = set()
      else:
        library_tags = set([tag for tag in FLAGS.libraries.split(',')])
    else:
      library_tags = set(options.library_tags)
    if not library_tags:
      tf.logging.info('Disregarding library tags.')
    else:
      tf.logging.info('Library tags to prove: %s',
                      ', '.join(sorted(list(library_tags))))

    # Load tactic files to fail fast in case error in specifying tactics.
    tactic_path = (str(options.path_tactics), str(options.path_tactics_replace))
    if tactic_path not in tactic_paths:
      tactic_paths[tactic_path] = io_util.load_tactics_from_file(*tactic_path)
    theorem_database_path = str(options.path_theorem_database)
    if theorem_database_path not in theorem_dbs:
      theorem_dbs[
          theorem_database_path] = io_util.load_theorem_database_from_file(
              theorem_database_path)
    theorem_db = theorem_dbs[theorem_database_path]
    if not theorem_db.HasField('name'):
      theorem_db.name = 'default'  # dummy name for backwards compatibility
      tf.logging.warning('Missing theorem database name is set to %s',
                         theorem_db.name)
    if FLAGS.task_list and FLAGS.tasks:
      tf.logging.fatal('Only one of --tasks or --task_list is allowed.')
    prover_tasks = prover_util.get_task_list(
        FLAGS.tasks,
        FLAGS.task_list,
        FLAGS.tasks_by_fingerprint,
        theorem_db,
        splits_to_prove,
        library_tags,
        fingerprint_file=FLAGS.tasks_by_fingerprint_file)
    # TODO(szegedy): Verify tasks that they all fit the theorem database(s)
    tf.logging.info('Number of prover tasks: %d', len(prover_tasks))
    prover_configurations.append(
        (prover_tasks, options, options.prover_run_subdirectory))
  return prover_configurations
