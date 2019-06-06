"""Utility functions for the prover.

This module contains various utility functions that can be shared between
various theorem prover objects and other helper utilities.
"""

from __future__ import absolute_import
from __future__ import division
# Import Type Annotations
from __future__ import print_function
import time
import tensorflow as tf
from typing import Iterable, Iterator, List, Optional, Text
from google.protobuf import text_format
from deepmath.deephol import action_generator
from deepmath.deephol import deephol_pb2
from deepmath.deephol import io_util
from deepmath.deephol import proof_search_tree
from deepmath.deephol import theorem_fingerprint
from deepmath.proof_assistant import proof_assistant_pb2


def make_premise_set(theorem: proof_assistant_pb2.Theorem,
                     database_name) -> proof_assistant_pb2.PremiseSet:
  """Create a premise set for the preceding section of the database.

  Args:
    theorem: Theorem object that is supposed to be in the database.
    database_name: Name of the database in which the theorem is located in.

  Returns:
    PremiseSet specifying the section in the database.
  """
  return proof_assistant_pb2.PremiseSet(sections=[
      proof_assistant_pb2.DatabaseSection(
          database_name=database_name,
          before_premise=theorem_fingerprint.Fingerprint(theorem))
  ])


def make_prover_task_for_goal(goal: proof_assistant_pb2.Theorem,
                              theorem: proof_assistant_pb2.Theorem,
                              database_name) -> proof_assistant_pb2.ProverTask:
  """Create a new prover task for proving the goal using preceding theorems.

  Args:
    goal: Goal to be proved.
    theorem: The first theorem in the database that should not be used for
      proving this goal. All preceding theorems are allowed.
    database_name: Name of the database in which the theorem is located in.

  Returns:
    ProverTask for proving the theorem using all preceding theorems.
  """
  return proof_assistant_pb2.ProverTask(
      premise_set=make_premise_set(theorem, database_name), goals=[goal])


def make_prover_task(theorem: proof_assistant_pb2.Theorem,
                     database_name) -> proof_assistant_pb2.ProverTask:
  """Create a new prover task for proving the theorem using preceding theorems.

  Args:
    theorem: Theorem to be proved.
    database_name: Name of the database in which the theorem is located in.

  Returns:
    ProverTask for proving the theorem using all preceding theorems.
  """
  return make_prover_task_for_goal(theorem, theorem, database_name)


def is_thm_included(thm, splits, library_tags):
  """Decides whether the theorem is included in the selection.

  This function can be used for filtering for theorems belonging to
  the allowed splits and library tags.

  Args:
    thm: Theorem object to be decided for inclusion.
    splits: Of type List[proof_assistant_pb2.Theorem.Split], the list of
      training splits for which tasks should be generated for.
    library_tags: List of strings for the library tags to be processed. If
      empty, then all library tags are allowed.

  Returns:
    Boolean indicating whether the theorem is included in the selection.
  """
  return thm.training_split in splits and (
      (not library_tags or (set(thm.library_tag) & library_tags)))


def create_tasks_for_theorem_db(theorem_db: proof_assistant_pb2.TheoremDatabase,
                                allowed_splits, library_tags
                               ) -> List[proof_assistant_pb2.ProverTask]:
  """Creates the theorem proving tasks for each theorem in the theorem database.

  The tasks will have the premise sets of all the theorems in the database
  preceding the goal.

  Args:
    theorem_db: The theorem database for which the tasks should be generated.
    allowed_splits: Of type List[proof_assistant_pb2.Theorem.Split], the list of
      training splits for which tasks should be generated for.
    library_tags: List of strings for the library tags to be processed. If
      empty, then all library tags are allowed.

  Returns:
    A list of prover tasks for the specified theorems.
  """
  assert theorem_db.HasField('name') and theorem_db.name
  return [
      make_prover_task(thm, theorem_db.name)
      for thm in theorem_db.theorems
      if is_thm_included(thm, allowed_splits, library_tags)
  ]


def theorem_to_goal_proto(thm: proof_assistant_pb2.Theorem
                         ) -> proof_assistant_pb2.Theorem:
  return proof_assistant_pb2.Theorem(
      tag=proof_assistant_pb2.Theorem.GOAL,
      hypotheses=thm.hypotheses,
      conclusion=thm.conclusion)


def try_tactics(node: proof_search_tree.ProofSearchNode, max_tries: int,
                min_successes: int, max_successes: int,
                premise_set: proof_assistant_pb2.PremiseSet,
                action_gen: action_generator.ActionGenerator,
                tactic_timeout_ms: int) -> int:
  """Generate initial proof attempts by applying a set of tactics.

  Args:
    node: A goal or subgoal with one or multiple proof attempts.
    max_tries: Maximum number of tactics applications to be attempted. Includes
      both successful and failed attempts.
    min_successes: Don't stop below this number of successful tactic
      applications, even if max_tries is reached unless we are left with no more
      tactics.
    max_successes: Stop after this many successful tactic applications.
    premise_set: A premise set passed to the action generator to select tactic
      arguments from.
    action_gen: Generator for creating and score tactic applications.
    tactic_timeout_ms: Timeout per-tactic in milliseconds.

  Returns:
    The number of successful tactics applications.
  """
  assert not node.processed
  assert not node.action_generation_time_millisec
  assert not node.successful_attempts
  assert not node.failed_attempts
  node.closed = False
  start_time = time.time()
  suggestion_scores = action_gen.step(node, premise_set)
  node.action_generation_time_millisec = int(
      round(1000.0 * (time.time() - start_time)))
  tf.logging.info('Suggestions and scores: %s', str(suggestion_scores))
  if not suggestion_scores:
    return 0
  top_suggestions = sorted(suggestion_scores, key=lambda x: x[1], reverse=True)
  request = proof_assistant_pb2.ApplyTacticRequest(
      goal=theorem_to_goal_proto(node.goal), timeout_ms=tactic_timeout_ms)
  tf.logging.info('Attempting to apply tactics: %s', str(top_suggestions))
  node.processed = True
  while (
      top_suggestions and
      (len(node.successful_attempts) < min_successes or
       (len(node.successful_attempts) + len(node.failed_attempts) <= max_tries))
      and (len(node.successful_attempts) < max_successes)):
    top_suggestion, score = top_suggestions.pop(0)
    request.tactic = top_suggestion
    proof_search_tree.TacticApplication(node, node.successful_attempts,
                                        node.failed_attempts, node.tree,
                                        request, score)
  if not node.successful_attempts:
    node.failed = True
    # Set those nodes to ignore that have become useless due to this
    # node's failing.
    node.update_ignore()
  return len(node.successful_attempts)


def translate_splits(splits: str):
  """Translate a comma separated list of splits in to python set.

  Args:
    splits: String with comma separated list of split specifications.

  Returns:
    Python set of proof_assistant_pb2.Theorem.Split.
  """

  def translate(s):
    if s == 'testing':
      return proof_assistant_pb2.Theorem.TESTING
    elif s == 'training':
      return proof_assistant_pb2.Theorem.TRAINING
    elif s == 'validation':
      return proof_assistant_pb2.Theorem.VALIDATION
    tf.logging.fatal('Unknown split specification: %s', s)

  if splits == 'all':
    return translate_splits('training,testing,validation')
  return {translate(s) for s in splits.split(',')}


class ProverTaskGenerator(object):
  """Class for ProofTask generation from prooflogs.

     This function makes three important assumptions:
     - Each ProofLog has a single Theorem object and several goals.
     - Each theorem object occurs in the database, but the goals don't occur
       there necessarily.
     - One can use any preceding theorems from the database to prove subgoals
       for a theorem.
  """

  def __init__(self,
               theorem_db: proof_assistant_pb2.TheoremDatabase,
               splits,
               create_tasks_for_closed_goals: bool = False,
               create_tasks_for_open_goals: bool = False,
               create_tasks_for_theorems: bool = False,
               create_tasks_for_subgoals: bool = False):
    """Constructor.

    Several flags control for which subset of the nodes should we generate
    task for.

    Args:
      theorem_db: A single corresponding theorem database that matches the proof
        logs. Should have its name field set.
      splits: Allowed set of training splits. Must be of type
        Set[proof_assistant_pb2.Theorem.Split].
      create_tasks_for_closed_goals: Specifies whether we should generate tasks
        for those goals that are marked closed.
      create_tasks_for_open_goals: Specifies whether we should generate tasks
        for those goals whose status is open.
      create_tasks_for_theorems: Specifies whether we should generate tasks for
        top-level theorems.
      create_tasks_for_subgoals: Specifies whether we should generate tasks for
        internal goal nodes of a proof.
    """
    self.theorem_db_name = theorem_db.name
    self.splits = splits
    self.fingerprint_to_theorem = {
        theorem_fingerprint.Fingerprint(thm): thm for thm in theorem_db.theorems
    }
    self.create_tasks_for_closed_goals = create_tasks_for_closed_goals
    self.create_tasks_for_open_goals = create_tasks_for_open_goals
    self.create_tasks_for_theorems = create_tasks_for_theorems
    self.create_tasks_for_subgoals = create_tasks_for_subgoals
    self.errors = []
    self.count_errors = {}
    self.nodes_omitted = 0
    self.theorems_with_error = 0
    self.theorems_omitted = 0
    self.count_logs = 0
    self.open_theorems = 0
    self.open_subgoals = 0
    self.open_others = 0
    self.closed_theorems = 0
    self.closed_subgoals = 0
    self.closed_others = 0
    self.nodes_refuted = 0
    self.count_dupes = 0

  def node_stats(self):
    return (
        ('Closed thms/subgoals/others: %d + %d + %d = %d\n' %
         (self.closed_theorems, self.closed_subgoals, self.closed_others,
          self.closed_theorems + self.closed_subgoals + self.closed_others)) +
        ('Open   thms/subgoals/others: %d + %d + %d = %d\n' %
         (self.open_theorems, self.open_subgoals, self.open_others,
          self.open_theorems + self.open_subgoals + self.open_others)) +
        ('(Open + Closed) Theorems: %d   Subgoals: %d   Others: %d\n' %
         (self.open_theorems + self.closed_theorems, self.open_subgoals +
          self.closed_subgoals, self.open_others + self.closed_others)) +
        ('Refuted: %d/omitted: %d' % (self.nodes_refuted, self.nodes_omitted)))

  def error_report(self):
    reports = [
        '"%s": %d' % (msg, count) for msg, count in sorted(
            self.count_errors.items(), key=lambda item: item[0])
    ]
    return '\n'.join(reports)

  def emit_error(self, error_msg, *args):
    """Stores error messages for later processing."""
    self.errors.append('%s for log %d' % (error_msg % args, self.count_logs))
    self.count_errors[error_msg] = self.count_errors.get(error_msg, 0) + 1

  def flush_errors(self):
    """Returns and flushes the errors."""
    errors = self.errors
    self.errors = []
    return errors

  def create_tasks(self, proof_log: deephol_pb2.ProofLog
                  ) -> Iterator[proof_assistant_pb2.ProverTask]:
    """Creates a stream of task according to the configuration.

    Args:
      proof_log: Input proof log to be processed.

    Yields:
      Prover tasks created for the subgoals.
    """
    self.count_logs += 1
    top_theorem = proof_log.theorem_in_database
    if top_theorem is None:
      tf.logging.fatal(text_format.MessageToString(proof_log))
      self.emit_error('Top level theorem is not found in the proof_log.')
      return
    database_theorem = self.fingerprint_to_theorem.get(
        theorem_fingerprint.Fingerprint(top_theorem))
    if database_theorem is None:
      self.emit_error('Could not find theorem %d in database' %
                      theorem_fingerprint.Fingerprint(top_theorem))
      self.theorems_with_error += 1
      return
    if database_theorem.training_split not in self.splits:
      self.theorems_omitted += 1
      return
    for node in proof_log.nodes:
      process = False
      if node.status == deephol_pb2.ProofNode.PROVED:
        process = self.create_tasks_for_closed_goals
        if node.goal.tag == proof_assistant_pb2.Theorem.THEOREM:
          process = process and self.create_tasks_for_theorems
          self.closed_theorems += 1
        elif node.goal.tag == proof_assistant_pb2.Theorem.GOAL:
          process = process and self.create_tasks_for_subgoals
          self.closed_subgoals += 1
        else:
          self.closed_others += 1
      elif node.status == deephol_pb2.ProofNode.UNKNOWN:
        process = self.create_tasks_for_open_goals
        if node.goal.tag == proof_assistant_pb2.Theorem.THEOREM:
          process = process and self.create_tasks_for_theorems
          self.open_theorems += 1
        elif node.goal.tag == proof_assistant_pb2.Theorem.GOAL:
          self.open_subgoals += 1
          process = process and self.create_tasks_for_subgoals
        else:
          self.open_others += 1
      else:
        assert node.status == deephol_pb2.ProofNode.REFUTED, (
            'Unknown node status: %d' % node.status)
        self.nodes_refuted += 1
        process = False
      if process:
        task = make_prover_task_for_goal(node.goal, top_theorem,
                                         self.theorem_db_name)
        for goal in task.goals:
          goal.training_split = database_theorem.training_split
        yield task
      else:
        self.nodes_omitted += 1

  def create_tasks_from_iterator(
      self,
      proof_log_iterator: Iterable[deephol_pb2.ProofLog],
      dedupe: bool = True,
      verbosity: int = 1000) -> Iterator[proof_assistant_pb2.ProverTask]:
    """Iterate over proof logs to craeate optionally deduplicated tasks.

    This method iterates over proof logs, generates tasks, dedupes them
    and turns them into tasks.

    Args:
      proof_log_iterator: An iterator over proof logs.
      dedupe: Boolean flag to indicate whether duplicate tasks are to be
        removed. The first task trying to solve a goal is kept. The default
        value is true.
      verbosity: Specifies the frequency with which error messages are written.
        None or zero verbosity

    Yields:
      ProverTask objects.
    """
    fingerprints = set()
    self.count_dupes = 0
    for proof_log in proof_log_iterator:
      if verbosity and self.count_logs % verbosity == 0:
        tf.logging.info('Log %d errors: %d', self.count_logs, len(self.errors))
        tf.logging.info('Stats:\n%s', self.node_stats())
        if self.errors:
          tf.logging.info('Errors:\n%s', self.error_report())
      for task in self.create_tasks(proof_log):
        if dedupe:
          assert len(task.goals) == 1
          fp = theorem_fingerprint.Fingerprint(task.goals[0])
        if not (dedupe and (fp in fingerprints)):
          if dedupe:
            fingerprints.add(fp)
          yield task
        else:
          self.count_dupes += 1

  def create_task_list(self,
                       proof_log_iterator: Iterable[deephol_pb2.ProofLog],
                       dedupe: bool = True,
                       verbosity: int = 1000
                      ) -> proof_assistant_pb2.ProverTaskList:
    """Iterate over proof logs to create an optionally deduplicated task list.

    This method iterates over proof logs, generates tasks, dedupes them
    and turns them into a ProverTaskList.

    Args:
      proof_log_iterator: An iterator over proof logs.
      dedupe: Boolean flag to indicate whether duplicate tasks are to be
        removed. The first task trying to solve a goal is kept. The default
        value is true.
      verbosity: Specifies the frequency with which error messages are written.
        None or zero verbosity

    Returns:
      A ProverTaskList object with the generated tasks.
    """
    task_list = proof_assistant_pb2.ProverTaskList()
    task_list.tasks.extend(
        self.create_tasks_from_iterator(proof_log_iterator, dedupe, verbosity))
    return task_list


def get_task_list(prover_tasks_file: Optional[str],
                  prover_task_list_file: Optional[str],
                  tasks_by_fingerprint: Optional[Text],
                  theorem_db: Optional[proof_assistant_pb2.TheoremDatabase],
                  splits, library_tags) -> List[proof_assistant_pb2.ProverTask]:
  """Get a list of theorem from either sources.

  Whichever parameter is specified first is used as a source for the
  tasks.

  Args:
    prover_tasks_file: File name for a text file with multiple ProverTasks in
      each line or a recordio file depending on the extension.
    prover_task_list_file: File name for a text protobuf prover_tasks_file.
    tasks_by_fingerprint: Comma-separated list of fingerprints from the theorem
      database to generate a prover task for.
    theorem_db: TheoremDatabase object.
    splits: List of splits to be considered. The list will be filtered for the
      splits that are not specified.
    library_tags: List of strings for the library tags to be processed. If
      empty, then all library tags are allowed.

  Returns:
    List of tasks extracted.
  """
  if prover_tasks_file:
    tf.logging.info('Loading tasks from tasks file "%s".', prover_tasks_file)
    return [
        task for task in io_util.read_protos(prover_tasks_file,
                                             proof_assistant_pb2.ProverTask)
        if is_thm_included(task.goals[0], splits, library_tags)
    ]
  elif prover_task_list_file:
    tf.logging.info('Loading tasks from task list file "%s".',
                    prover_tasks_file)
    task_list = io_util.load_text_proto(prover_task_list_file,
                                        proof_assistant_pb2.ProverTaskList,
                                        'prover task list')
    return [
        task for task in task_list.tasks
        if is_thm_included(task.goals[0], splits, library_tags)
    ]
  elif tasks_by_fingerprint:
    if not theorem_db:
      tf.logging.fatal('Require a theorem database to create prover tasks from '
                       'fingerprints.')
    tf.logging.info('Generating task list for fingerprint(s) %s',
                    tasks_by_fingerprint)
    fingerprints = set([int(fp) for fp in tasks_by_fingerprint.split(',')])
    theorems = []
    for thm in theorem_db.theorems:
      fingerprint = theorem_fingerprint.Fingerprint(thm)
      if fingerprint in fingerprints:
        fingerprints.remove(fingerprint)
        theorems.append(thm)
    if fingerprints:
      tf.logging.error('Some fingerprints could not be found in theorem db: %s',
                       str(fingerprints))
    return [
        make_prover_task_for_goal(thm, thm, theorem_db.name) for thm in theorems
    ]
  else:
    tf.logging.info('Generating task list for theorem database.')
    return create_tasks_for_theorem_db(theorem_db, splits, library_tags)
