"""DeepHOL prover."""
import random
import time
from typing import Optional, Text
import tensorflow.compat.v1 as tf
from google.protobuf import text_format
from deepmath.deephol import action_generator
from deepmath.deephol import deephol_pb2
from deepmath.deephol import proof_search_tree
from deepmath.deephol import prover_util
from deepmath.deephol import prune_lib
from deepmath.proof_assistant import proof_assistant_pb2
from deepmath.public import error
# Max number of tactics to attempt to apply per NoBacktrack proofs.
NO_BACKTRACK_SEARCH_NODES = 45


def _sample_from_interval(interval: deephol_pb2.IntegerInterval):
  return random.randint(interval.min_value, interval.max_value)


def _sample_mcts_options(prover_options: deephol_pb2.ProverOptions):
  """Sample parameters according the meta options."""
  if not prover_options.HasField('mcts_options'):
    return
  options = prover_options.mcts_options
  if options.HasField('meta_options'):
    meta_options = options.meta_options
    if meta_options.HasField('max_theorem_parameters'):
      prover_options.action_generator_options.max_theorem_parameters = (
          _sample_from_interval(meta_options.max_theorem_parameters))


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


def check_task(
    task: proof_assistant_pb2.ProverTask,
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
    if self.prover_options.prune_theorem_parameters or self.prover_options.prune_proof:
      self.pruner = prune_lib.ParameterPruning(
          theorem_db, hol_wrapper=hol_wrapper)

  def timed_out(self) -> bool:
    """Returns true if the prover has timed out."""
    return time.time() - self.start_time > self.timeout_seconds

  def prove_one(self,
                task: proof_assistant_pb2.ProverTask) -> deephol_pb2.ProofLog:
    """Prove a single-goal task.

    This method can assume an already initialized search tree with node 0
    being the sing goal specified in the task.

    Args:
      task: Task to be performed.

    Returns:
      A proof log.
    """
    raise NotImplementedError('Must define this.')

  def prove(self, task: proof_assistant_pb2.ProverTask) -> deephol_pb2.ProofLog:
    """Top level prove method."""
    if not self.single_goal:
      tf.logging.fatal('Only one goal per task is supported')
    proof_log = self.prove_one_wrapper(task)
    if self.prover_options:
      proof_log.prover_options.CopyFrom(self.prover_options)
    return proof_log

  def time_spent(self) -> int:
    """Time spent since the last call to proof_one_wrapper in ms."""
    return int(round((time.time() - self.start_time) * 1000.0))

  def prove_one_wrapper(
      self, task: proof_assistant_pb2.ProverTask) -> deephol_pb2.ProofLog:
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
    _sample_mcts_options(self.prover_options)
    log = check_task(task, self.prover_options)
    if log is not None:
      tf.logging.info('Task did not fit the prover.')
      return log
    error_message = None
    if self.accept_tasks:
      try:
        self.start_time = time.time()
        tf.logging.info('Attempting task %s.',
                        text_format.MessageToString(task))
        proof_log = self.prove_one(task)
      except error.StatusNotOk as exception:
        tf.logging.error('Prover stopped accepting tasks due to "%s"',
                         exception.message)
        self.error = exception.message
        proof_log = deephol_pb2.ProofLog(error_message=exception.message)
        self.accept_tasks = False
      error_message = proof_log.error_message
    else:
      tf.logging.warning('Prover does not accept tasks anymore.')
      error_message = 'Prover stopped accepting tasks due to %s.' % self.error
    if not self.accept_tasks:
      proof_log.rejected = True
    proof_log.time_spent = self.time_spent()
    if (proof_log.nodes and
        proof_log.nodes[0].status == deephol_pb2.ProofNode.PROVED):
      proof_log.num_proofs = 1
    else:
      proof_log.num_proofs = 0
      proof_log.error_message = error_message or 'No proof.'
    proof_log.prover_options.CopyFrom(self.prover_options)
    proof_log.prover_task.CopyFrom(task)
    if self.prover_options.prune_proof:
      assert self.pruner is not None
      self.pruner.prune_proof_log(proof_log)
    elif self.prover_options.prune_theorem_parameters:
      assert self.pruner is not None
      self.pruner.prune_closed_tactic_applications(proof_log)
    return proof_log


class BFSProver(Prover):
  """A BFS prover for single-goal tasks."""

  def __init__(self, prover_options: deephol_pb2.ProverOptions, hol_wrapper,
               action_gen: action_generator.ActionGenerator,
               theorem_db: proof_assistant_pb2.TheoremDatabase):
    super(BFSProver, self).__init__(
        prover_options, hol_wrapper, theorem_db, single_goal=True)
    self.action_gen = action_gen
    self.options = prover_options.bfs_options

  def prove_one(self,
                task: proof_assistant_pb2.ProverTask) -> deephol_pb2.ProofLog:
    """Searches for a proof via BFS.

    Args:
      task: ProverTask to be performed.

    Returns:
      A proof log.
    """
    goal_thm = task.goals[0]
    tree = proof_search_tree.ProofSearchTree(self.hol_wrapper, goal_thm, task)
    root = tree.nodes[0]
    nodes_explored = 0
    # Note that adding new node to the tree might re-enable previous nodes
    # for tactic applications, if they were marked to be ignored by
    # failing sibling nodes.
    tree.cur_index = 0
    exit_message = None
    while not self.timed_out() and not root.closed and not root.failed and (
        nodes_explored < self.options.max_explored_nodes):
      if tree.cur_index >= len(tree.nodes):
        exit_message = 'BFS: All nodes are failed or ignored.'
        break
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
      exit_message = 'BFS: Timeout.'
    elif root.failed:
      exit_message = 'BFS: Root Failed.'
    elif nodes_explored >= self.options.max_explored_nodes and not root.closed:
      exit_message = 'BFS: Node limit reached.'

    proof_log = tree.to_proto()
    if exit_message is not None:
      proof_log.error_message = exit_message
    return proof_log
