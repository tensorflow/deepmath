"""Simple runner for the prover.

Iterate over the tasks sequentially.

This runner can run the prover on a set of tasks without the overhead of
starting a distributed job.
"""
from typing import List
from typing import Text
import tensorflow.compat.v1 as tf
from deepmath.public import build_data
from deepmath.deephol import action_generator
from deepmath.deephol import deephol_pb2
from deepmath.deephol import embedding_store
from deepmath.deephol import io_util
from deepmath.deephol import mcts_prover
from deepmath.deephol import predictions
from deepmath.deephol import predictions_builder
from deepmath.deephol import proof_assistant_builder
from deepmath.deephol import prover
from deepmath.deephol import prover_environment
from deepmath.public import sequence_action_generator
from deepmath.deephol.utilities import stats
from deepmath.proof_assistant import proof_assistant_pb2

FLAGS = tf.flags.FLAGS


def program_started():
  pass


def compute_stats(output):
  """Compute aggregate statistics given prooflog file."""
  tf.logging.info('Computing aggregate statistics from %s', output)
  stat_list = [
      stats.proof_log_stats(log) for log in io_util.read_proof_logs(output)
  ]
  if not stat_list:
    tf.logging.info('Empty stats list.')
    return
  aggregate_stat = stats.aggregate_stats(stat_list)
  tf.logging.info('Aggregated statistics:')
  tf.logging.info(stats.aggregate_stat_to_string(aggregate_stat))


def get_predictor(
    options: deephol_pb2.ProverOptions) -> predictions.Predictions:
  """Returns appropriate predictor based on prover options."""
  return predictions_builder.build(options)


def cache_embeddings(options: deephol_pb2.ProverOptions):
  emb_path = str(options.theorem_embeddings)
  if options.HasField('theorem_embeddings') and not tf.gfile.Glob(emb_path):
    tf.logging.info(
        'theorem_embeddings file "%s" does not exist, computing & saving.',
        emb_path)
    emb_store = embedding_store.TheoremEmbeddingStore(get_predictor(options))
    emb_store.compute_embeddings_for_thms_from_db_file(
        str(options.path_theorem_database))
    emb_store.save_embeddings(emb_path)


def create_prover(options: deephol_pb2.ProverOptions,
                  theorem_database=None,
                  tactics=None,
                  predictor=None) -> prover.Prover:
  """Creates a Prover object, initializing all dependencies."""
  if theorem_database is None:
    theorem_database = io_util.load_theorem_database_from_file(
        str(options.path_theorem_database))
  if tactics is None:
    tactics = io_util.load_tactics_from_file(
        str(options.path_tactics), str(options.path_tactics_replace))
  if options.action_generator_options.asm_meson_no_params_only:
    tf.logging.warn('Note: Using Meson action generator with no parameters.')
    action_gen = action_generator.MesonActionGenerator()
  elif options.HasField('sequence_action_generator_options'):
    action_gen = sequence_action_generator.SequenceActionGenerator(
        theorem_database, tactics, options.sequence_action_generator_options,
        str(options.path_model_prefix))
  else:
    if predictor is None:
      predictor = get_predictor(options)
    emb_store = None
    if options.HasField('theorem_embeddings'):
      emb_store = embedding_store.TheoremEmbeddingStore(predictor)
      emb_store.read_embeddings(str(options.theorem_embeddings))
      if emb_store.thm_embeddings.shape[0] != len(theorem_database.theorems):
        raise ValueError('Using incorrect embedding database.')
    action_gen = action_generator.ActionGenerator(
        theorem_database, tactics, predictor, options.action_generator_options,
        options.model_architecture, emb_store)
  hol_wrapper = proof_assistant_builder.build(theorem_database)
  tf.logging.info('DeepHOL dependencies initialization complete.')
  if options.prover == 'bfs':
    return prover.BFSProver(options, hol_wrapper, action_gen, theorem_database)
  if options.prover == 'mcts':
    env = prover_environment.ProverEnvironment(options, hol_wrapper, action_gen,
                                               predictor)
    return mcts_prover.MCTSProver(options, hol_wrapper, theorem_database, env)
  raise ValueError('Unknown prover type: %s' % options.prover)


def run_pipeline(prover_tasks: List[proof_assistant_pb2.ProverTask],
                 prover_options: deephol_pb2.ProverOptions, path_output: Text):
  """Iterate over all prover tasks and store them in the specified file."""
  if FLAGS.output.split('.')[-1] != 'textpbs':
    tf.logging.warn('Output file should end in ".textpbs"')

  cache_embeddings(prover_options)
  this_prover = create_prover(prover_options)
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
