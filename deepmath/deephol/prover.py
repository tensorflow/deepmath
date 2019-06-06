"""DeepHOL prover."""

from __future__ import absolute_import
from __future__ import division
# Import Type Annotations
from __future__ import print_function
import random
import time
import tensorflow as tf
from typing import Optional, Text
from google.protobuf import text_format
# Import predictors.
from deepmath.deephol.public import proof_assistant
from deepmath.deephol import action_generator
from deepmath.deephol import deephol_pb2
from deepmath.deephol import embedding_store
from deepmath.deephol import holparam_predictor
from deepmath.deephol import io_util
from deepmath.deephol import predictions
from deepmath.deephol import proof_search_tree
from deepmath.deephol import prover_util
from deepmath.deephol import prune_lib
from deepmath.proof_assistant import proof_assistant_pb2
from deepmath.public import error
# Max number of tactics to attempt to apply per NoBacktrack proofs.
NO_BACKTRACK_SEARCH_NODES = 45


def _sample_from_interval(interval: deephol_pb2.IntegerInterval):
  return random.randint(interval.min_value, interval.max_value)


def _sample_bfs_options(prover_options: deephol_pb2.ProverOptions):
  """Sample parameters according the meta options."""
  if not prover_options.HasField('bfs_options'):
    return
  options = prover_options.bfs_options
  if options.HasField('meta_options'):
    meta_options = options.meta_options
    if meta_options.HasField('max_top_suggestions'):
      options.max_top_suggestions = _sample_from_interval(
          meta_options.max_top_suggestions)
    if meta_options.HasField('max_successful_branches'):
      options.max_successful_branches = _sample_from_interval(
          meta_options.max_successful_branches)
    if meta_options.HasField('max_explored_nodes'):
      options.max_explored_nodes = _sample_from_interval(
          meta_options.max_explored_nodes)
    if meta_options.HasField('min_successful_branches'):
      options.min_successful_branches = _sample_from_interval(
          meta_options.min_successful_branches)
    if meta_options.HasField('max_theorem_parameters'):
      prover_options.action_generator_options.max_theorem_parameters = (
          _sample_from_interval(meta_options.max_theorem_parameters))


def check_task(task: proof_assistant_pb2.ProverTask,
               prover_options: deephol_pb2.ProverOptions
              ) -> Optional[deephol_pb2.ProofLog]:
  """Check whether the task is valid or supported.

  If the task is not valid and supported, then it returns a ProofLog with the
  appropriate error message.

  Args:
    task: Prover task to be performed.
    prover_options: Prover options.

  Returns:
     None at success or a proof log with error message otherwise.
  """

  def make_empty_log(error_message: Text):
    return deephol_pb2.ProofLog(
        error_message=error_message,
        num_proofs=0,
        prover_options=prover_options)

  if not task.goals:
    return make_empty_log('Task has no theorems to prove')
  elif len(task.goals) > 1:
    return make_empty_log('Multiple theorems in one task are not supported '
                          'yet.')
  return None


class Prover(object):
  """Base class of the prover."""

  def __init__(self, prover_options, hol_wrapper, theorem_db, single_goal=True):
    if not single_goal:
      tf.logging.fatal('Only one goal per task is supported')
    self.prover_options = prover_options
    self.hol_wrapper = hol_wrapper
    self.accept_tasks = True
    self.error = None
    self.single_goal = single_goal
    self.start_time = time.time()
    # Timeout for each individual "prove" call in separation.
    self.timeout_seconds = self.prover_options.timeout_seconds
    self.pruner = None
    if self.prover_options.prune_theorem_parameters:
      self.pruner = prune_lib.ParameterPruning(
          theorem_db, hol_wrapper=hol_wrapper)

  def timed_out(self) -> bool:
    """Returns true if the prover has timed out."""
    return time.time() - self.start_time > self.timeout_seconds

  def prove_one(self, search_tree: proof_search_tree.ProofSearchTree,
                task: proof_assistant_pb2.ProverTask) -> Optional[Text]:
    """Prove a single-goal task.

    This method can assume an already initialized search tree with node 0
    being the sing goal specified in the task.

    Args:
      search_tree: The pre-initialized search tree.
      task: Task to be performed.

    Returns:
      Error message on error, None otherwise.
    """
    raise NotImplementedError('Must define this.')

  def prove(self, task: proof_assistant_pb2.ProverTask) -> deephol_pb2.ProofLog:
    """Top level prove method."""
    if not self.single_goal:
      tf.logging.fatal('Only one goal per task is supported')
    return self.prove_one_wrapper(task)

  def prove_one_wrapper(self, task: proof_assistant_pb2.ProverTask
                       ) -> deephol_pb2.ProofLog:
    """Wrapper of prove_one methods for single goal use cases.

    This wrapper handles, timeout, error management and can set the prover
    into a mode that does not accept tasks anymore.

    Args:
      task: ProverTask to be performed.

    Returns:
      A proof log of the task performed.
    """
    # Note that this changes the prover options in place.
    _sample_bfs_options(self.prover_options)
    log = check_task(task, self.prover_options)
    if log is not None:
      tf.logging.info('Task did not fit the prover.')
      return log
    goal_thm = task.goals[0]
    tree = proof_search_tree.ProofSearchTree(self.hol_wrapper, goal_thm)
    error_message = None
    if self.accept_tasks:
      try:
        self.start_time = time.time()
        tf.logging.info('Attempting task %s.',
                        text_format.MessageToString(task))
        error_message = self.prove_one(tree, task)
      except error.StatusNotOk as exception:
        tf.logging.error('Prover stopped accepting tasks due to "%s"',
                         exception.message)
        self.error = exception.message
        error_message = exception.message
        self.accept_tasks = False
    else:
      tf.logging.warning('Prover does not accept tasks anymore.')
      error_message = 'Prover stopped accepting tasks due to %s.' % self.error
    proof_log = tree.to_proto()
    if not self.accept_tasks:
      proof_log.rejected = True
    proof_log.time_spent = int(round((time.time() - self.start_time) * 1000.0))
    if tree.nodes[0].closed:
      proof_log.num_proofs = 1
    else:
      proof_log.num_proofs = 0
      proof_log.error_message = error_message or 'No proof.'
    proof_log.prover_options.CopyFrom(self.prover_options)
    proof_log.prover_task.CopyFrom(task)
    tf.logging.info('Pruning theorem nodes...')
    if self.pruner is not None:
      for node in proof_log.nodes:
        if node.status == deephol_pb2.ProofNode.PROVED:
          self.pruner.prune_closed_tactic_applications(node)
    return proof_log


class NoBacktrackProver(Prover):
  """Searches for a proof without backtracking for single-goal tasks."""

  def __init__(self, prover_options: deephol_pb2.ProverOptions, hol_wrapper,
               action_gen: action_generator.ActionGenerator,
               theorem_db: proof_assistant_pb2.TheoremDatabase):
    super(NoBacktrackProver, self).__init__(
        prover_options, hol_wrapper, theorem_db, single_goal=True)
    self.action_gen = action_gen

  def prove_one(self, tree: proof_search_tree.ProofSearchTree,
                task: proof_assistant_pb2.ProverTask) -> Optional[Text]:
    """Searches for a proof without backtracking.

    Args:
      tree: Search tree with a single goal node to be proved.
      task: ProverTask to be performed.

    Returns:
      None on success and error message on failure.
    """
    root = tree.nodes[0]
    budget = NO_BACKTRACK_SEARCH_NODES
    cur_index = 0
    while not root.closed and not self.timed_out():
      if cur_index >= len(tree.nodes):
        # This situation can happen only if the tactics succeed, but end up
        # reconstructing an earlier node.
        return 'NoBacktrack: Loop.'
      node = tree.nodes[cur_index]
      cur_index += 1
      prover_util.try_tactics(node, budget, 0, 1, task.premise_set,
                              self.action_gen,
                              self.prover_options.tactic_timeout_ms)
      if not node.successful_attempts:
        return ('NoBacktrack: No successful tactic applications within '
                'limit %d' % budget)
      else:
        if len(node.successful_attempts) != 1:
          tf.logging.info('%d successful attempts.',
                          len(node.successful_attempts))
          for tac_app in node.successful_attempts:
            tf.logging.info('attempt: %s', tac_app.tactic)
        assert len(node.successful_attempts) == 1
        budget -= len(node.failed_attempts) + 1
    if not root.closed:
      if self.timed_out():
        return 'Timed out.'
      else:
        return 'NoBacktrack: Could not find proof.'


class BFSProver(Prover):
  """A BFS prover for single-goal tasks."""

  def __init__(self, prover_options: deephol_pb2.ProverOptions, hol_wrapper,
               action_gen: action_generator.ActionGenerator,
               theorem_db: proof_assistant_pb2.TheoremDatabase):
    super(BFSProver, self).__init__(
        prover_options, hol_wrapper, theorem_db, single_goal=True)
    self.action_gen = action_gen
    self.options = prover_options.bfs_options

  def prove_one(self, tree: proof_search_tree.ProofSearchTree,
                task: proof_assistant_pb2.ProverTask) -> Optional[Text]:
    """Searches for a proof via BFS.

    Args:
      tree: Search tree with a single goal node to be proved.
      task: ProverTask to be performed.

    Returns:
      None on success and error message on failure.
    """
    root = tree.nodes[0]
    nodes_explored = 0
    # Note that adding new node to the tree might re-enable previous nodes
    # for tactic applications, if they were marked to be ignored by
    # failing sibling nodes.
    tree.cur_index = 0
    while not self.timed_out() and not root.closed and not root.failed and (
        nodes_explored < self.options.max_explored_nodes):
      if tree.cur_index >= len(tree.nodes):
        return 'BFS: All nodes are failed or ignored.'
      node = tree.nodes[tree.cur_index]
      tree.cur_index += 1
      if node.ignore or node.failed or node.closed or node.processed:
        continue
      nodes_explored += 1
      # Note that the following function might change tree.cur_index
      # (if a node that was ignored suddenly becomes subgoal of a new
      # tactic application).
      prover_util.try_tactics(node, self.options.max_top_suggestions,
                              self.options.min_successful_branches,
                              self.options.max_successful_branches,
                              task.premise_set, self.action_gen,
                              self.prover_options.tactic_timeout_ms)
    root_status = ' '.join([
        p[0] for p in [('closed', root.closed), ('failed', root.failed)] if p[1]
    ])
    tf.logging.info('Timeout: %s root status: %s explored: %d',
                    str(self.timed_out()), root_status, nodes_explored)
    if self.timed_out():
      return 'BFS: Timeout.'
    elif root.failed:
      return 'BFS: Root Failed.'
    elif nodes_explored >= self.options.max_explored_nodes and not root.closed:
      return 'BFS: Node limit reached.'


def get_predictor(options: deephol_pb2.ProverOptions
                 ) -> predictions.Predictions:
  """Returns appropriate predictor based on prover options."""
  model_arch = options.model_architecture
  if model_arch == deephol_pb2.ProverOptions.PAIR_DEFAULT:
    return holparam_predictor.HolparamPredictor(str(options.path_model_prefix))
  if model_arch == deephol_pb2.ProverOptions.PARAMETERS_CONDITIONED_ON_TAC:
    return holparam_predictor.TacDependentPredictor(
        str(options.path_model_prefix))
  if model_arch == deephol_pb2.ProverOptions.GNN_GOAL:
    raise NotImplementedError('GNN_GOAL not implemented for %s' %
        str(options.path_model_prefix))
  if (model_arch == deephol_pb2.ProverOptions.HIST_AVG or
      model_arch == deephol_pb2.ProverOptions.HIST_CONV or
      model_arch == deephol_pb2.ProverOptions.HIST_ATT):
    raise NotImplementedError(
        'History-dependent model %s is not supported in the prover.' %
        model_arch)

  raise AttributeError('Unknown model architecture in prover options: %s' %
                       model_arch)


def cache_embeddings(options: deephol_pb2.ProverOptions):
  emb_path = str(options.theorem_embeddings)
  if options.HasField('theorem_embeddings') and not tf.gfile.Exists(emb_path):
    tf.logging.info(
        'theorem_embeddings file "%s" does not exist, computing & saving.',
        emb_path)
    emb_store = embedding_store.TheoremEmbeddingStore(get_predictor(options))
    emb_store.compute_embeddings_for_thms_from_db_file(
        str(options.path_theorem_database))
    emb_store.save_embeddings(emb_path)


def create_prover(options: deephol_pb2.ProverOptions) -> Prover:
  """Creates a Prover object, initializing all dependencies."""
  theorem_database = io_util.load_theorem_database_from_file(
      str(options.path_theorem_database))
  tactics = io_util.load_tactics_from_file(
      str(options.path_tactics), str(options.path_tactics_replace))
  if options.action_generator_options.asm_meson_no_params_only:
    tf.logging.warn('Note: Using Meson action generator with no parameters.')
    action_gen = action_generator.MesonActionGenerator()
  else:
    predictor = get_predictor(options)
    emb_store = None
    if options.HasField('theorem_embeddings'):
      emb_store = embedding_store.TheoremEmbeddingStore(predictor)
      emb_store.read_embeddings(str(options.theorem_embeddings))
      assert emb_store.thm_embeddings.shape[0] == len(theorem_database.theorems)
    action_gen = action_generator.ActionGenerator(
        theorem_database, tactics, predictor, options.action_generator_options,
        options.model_architecture, emb_store)
  hol_wrapper = setup_prover(theorem_database)
  tf.logging.info('DeepHOL dependencies initialization complete.')
  if options.prover == 'bfs':
    return BFSProver(options, hol_wrapper, action_gen, theorem_database)
  return NoBacktrackProver(options, hol_wrapper, action_gen, theorem_database)


def setup_prover(theorem_database: proof_assistant_pb2.TheoremDatabase):
  """Starts up HOL and seeds it with given TheoremDatabase."""
  tf.logging.info('Setting up and registering theorems with proof assistant...')
  proof_assistant_obj = proof_assistant.ProofAssistant()
  for thm in theorem_database.theorems:
    response = proof_assistant_obj.RegisterTheorem(
        proof_assistant_pb2.RegisterTheoremRequest(theorem=thm))
    if response.HasField('error_msg') and response.error_msg:
      tf.logging.fatal('Registration failed for %d with: %s' %
                       (response.fingerprint, response.error_msg))
  tf.logging.info('Proof assistant setup done.')
  return proof_assistant_obj
