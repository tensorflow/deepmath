"""Infrastructure to maintain a proof search tree.

This file contains support for performing and logging proof searches.

Proof search tree has the following important features:
  - Can maintain multiple alternative proof branches.
  - Allows efficient subgoal-sharing between different proof branches.
  - Consistent, verified proof-search logging, independent of the
    search implementation.

A search algorithm should iterate on:
  1. Rank subgoals.
    1.1 Picking a (open) subgoal from the tree.
    1.2 Create a new SearchTreeNode for closing the subgoal
    1.3 Call try_tactics for the SearchTreeNode.
    1.4 Update the status and other metadata of the nodes whose status
        has changed.
    1.5 Rerank subgoals whose status might have changed.
    1.6 Go to 1.1
  2. Produce search log.
"""
import collections
import random
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple, Text
import tensorflow.compat.v1 as tf
from deepmath.deephol import deephol_pb2
from deepmath.deephol import predictions
from deepmath.deephol import tactic_utils
from deepmath.deephol import theorem_fingerprint
from deepmath.deephol import theorem_utils
from deepmath.deephol import to_sexpression
from deepmath.public import proof_assistant
from deepmath.deephol.utilities import normalization_lib
from deepmath.deephol.utilities import sexpression_parser
from deepmath.proof_assistant import proof_assistant_pb2
from deepmath.public import error

# To reference a SubGoal of a tactic application, we need the following two
# pieces of information:
# - The TacticApplication that generated this subgoal.
# - The index of the subgoal in the list of subgoals.
# Note that we can't use NamedTuple, since TacticApplication can't be
# referenced due to recursive definitions.
SubGoalRef = collections.namedtuple('SubGoalRef',
                                    ['tactic_application', 'subgoal_index'])


def _is_same_theorem(t1: proof_assistant_pb2.Theorem,
                     t2: proof_assistant_pb2.Theorem):
  assert t1.tag == proof_assistant_pb2.Theorem.THEOREM
  assert t2.tag == proof_assistant_pb2.Theorem.THEOREM
  assert not t1.assumptions
  assert not t2.assumptions
  return t1.conclusion == t2.conclusion and t1.hypotheses == t2.hypotheses


def _is_same_goal(g1: proof_assistant_pb2.Theorem,
                  g2: proof_assistant_pb2.Theorem):
  assert g1.tag == proof_assistant_pb2.Theorem.GOAL
  assert g2.tag == proof_assistant_pb2.Theorem.GOAL
  assert not g1.hypotheses
  assert not g2.hypotheses
  return (g1.conclusion == g2.conclusion and
          len(g1.assumptions) == len(g2.assumptions) and all([
              _is_same_theorem(a1, a2)
              for a1, a2 in zip(g1.assumptions, g2.assumptions)
          ]))


class ProofSearchTree(object):
  """Container object to represent the whole search tree.

  This object maintains:
    - A list of nodes, where the first node corresponds to the root goal.
      (Which should be in the theorem database now, for premise selection
       purposes).
    - A map of theorems to nodes in order to allow subgoal-sharing. It is
      unclear if this ever happens.
    - A pointer to the wrapper for the proof assistant.
    - A current index to iterate through the search tree in a BFS manner.
  """

  def add_node(self, goal: proof_assistant_pb2.Theorem,
               parent: Optional[SubGoalRef]) -> 'ProofSearchNode':
    """Append a new node to the tree."""

    for assum in goal.assumptions:
      if not assum.HasField('assumption_index'):
        raise ValueError('Assumption indices not set.')

    if goal.tag != proof_assistant_pb2.Theorem.GOAL:
      tf.logging.warning(
          'ProofSearchTree given goal with tag %d. Converting '
          'to goal.', goal.tag)
      goal = theorem_utils.theorem_to_goal(goal)

    goal_as_string = to_sexpression.convert_goal(goal, False)
    if goal_as_string in self.nodes_map:
      node = self.nodes[self.nodes_map[goal_as_string]]
      # Make sure that we really match everything exactly
      assert _is_same_goal(node.goal, goal)
      if parent is not None:
        node.parents.append(parent)
        # If the node was already ignored, remove its ignore flag if
        # there is something we can still do about this node.
        # Now the question remains: How would the prover enqueue all
        # nodes that might be helpful for closing this goal?
        # Note that this code might or might not remove the ignore
        # flag from this node and a lot of its descendants.
        # However, the descendants should have higher index than this
        # node, unless the node is involved in a loop in which case
        # this node can never be closed along that loop.
        node.remove_ignore()
        if not (node.ignore or node.closed or node.failed):
          if self.cur_index is None or self.cur_index > node.index:
            self.cur_index = node.index
      return node
    else:
      index = len(self.nodes)
      self.nodes_map[goal_as_string] = index
      node = ProofSearchNode(self, index, goal, parent)
      self.nodes.append(node)
      return node

  def __init__(self,
               proof_assistant_obj: Optional[proof_assistant.ProofAssistant],
               goal: proof_assistant_pb2.Theorem,
               prover_task: Optional[proof_assistant_pb2.ProverTask] = None):
    """Constructor for a proof search tree.

    Args:
      proof_assistant_obj: An interface to the proof assistant. Can be set to
        None, e.g. when deserializing a proof log into a proof search tree.
      goal: The root goal which is also used to limit the premise selection to
        preceding theorems. This is the first theorem in the theorem database
        that is not allowed to be used in the proof. For now, it is mandatory
        that the goal is in the theorem database. Later, we should relax this
        constraint.
      prover_task: ProverTask could be used, e.g. for target if forward proving.
    """
    self.proof_assistant = proof_assistant_obj
    self.nodes = []
    self.nodes_map = {}
    self.cur_index = None
    root = self.add_node(goal, None)
    assert root.index == 0
    self.prover_task = prover_task

  @property
  def targets(self) -> List[proof_assistant_pb2.Theorem]:
    """List of target goals to optionally reach.

    Returns:
      The target goals. Prover is trying to a subset of these targets.
    """
    if self.prover_task is None:
      return []
    targets = list(self.prover_task.targets)
    if len(targets) > 1:
      raise ValueError('Target goal stack must be empty or singleton.')
    return targets

  def within_targets(self, goal: proof_assistant_pb2.Theorem) -> bool:
    """Checks if goal is within the optional set of additional target goals."""
    return any(_is_same_goal(goal, target) for target in self.targets)

  def to_proto(self) -> deephol_pb2.ProofLog:
    """Serialize the proof search tree as a protobuf.

    Returns:
      A deephol_pb2.ProofLog protobuf representing the whole proof search tree.
    """
    proof_log = deephol_pb2.ProofLog()

    if self.prover_task is not None:
      proof_log.prover_task.CopyFrom(self.prover_task)

    for node in self.nodes:
      status = deephol_pb2.ProofNode.UNKNOWN
      if node.closed:
        status = deephol_pb2.ProofNode.PROVED
      node_log = proof_log.nodes.add(
          goal=node.goal,
          status=status,
          action_generation_time_millisec=node.action_generation_time_millisec,
          proof_state_emb_time_ms=node.proof_state_emb_time_ms,
          theorem_scores_time_ms=node.theorem_scores_time_ms,
          assumptions_ranking_time_ms=node.assumptions_ranking_time_ms,
          heuristic_ranking_time_ms=node.heuristic_ranking_time_ms,
          root_goal=(node.index == 0))
      for tapp in node.failed_attempts:
        tapp.add_to_node_proto(node_log)
      for tapp in node.successful_attempts:
        tapp.add_to_node_proto(node_log)
    return proof_log


class TacticApplication(object):
  """Result of tactic applications."""

  def __init__(
      self,
      parent,  # : ProofSearchNode,
      tactic: Text,
      parameters: Iterable[deephol_pb2.TacticParameter],
      score: float,
      rank: Optional[int],
      index: int,
      result: deephol_pb2.TacticApplication.Result,
      error_message: Optional[Text],
      time_spent: int,
      subgoals,  # : List[ProofSearchNode],
      closed: bool,
      failed: bool):
    """Constructor for the result of a tactic application.

    Args:
      parent: ProofSearchNode to which the tactic was applied to.
      tactic: Tactic string that was applied, NOT including the parameter list.
      parameters: List of tactic parameters, e.g. the premises.
      score: Score produced by the action generator.
      rank: Order of this tactic application against other tactic applications
        coming from the same proof search node, wrt action generator score.
      index: Index of the tactic application in either (successful or failed)
        list of proof attempts in the ProofSearchNode.
      result: Result of the tactic application.
      error_message: Error message (if any) from applying the tactic.
      time_spent: How much time in milliseconds was spent applying the tactic.
      subgoals: List of ProofSearchNodes corresponding to the subgoals of this
        tactic application.
      closed: True iff all the subgoals are closed.
      failed: True iff all the subgoals are failed to close. DEPRECATED.
    """
    self.parent = parent
    self.tactic = tactic
    self.parameters = list(parameters)
    self.score = score
    self.rank = rank
    self.index = index
    self.result = result
    self.error_message = error_message
    self.time_spent = time_spent
    self.subgoals = subgoals
    self.closed = closed
    self.failed = failed

  def update_closed(self) -> bool:
    """Update the "closed" property for the TacticApplication.

    It returns true if the application was a successful tactic
    application and all of the resulting subgoals are already marked as closed.
    Otherwise it checks all subgoals and marks the application if
    any of them is closed. Note that it is essential to test that
    self.result is SUCCESS, otherwise failed attempt would be marked
    as closed, which would be a grave mistake.

    Returns:
      True if the status of the application was success and all
      the subgoals are closed, otherwise false.
    """
    if self.result != deephol_pb2.TacticApplication.SUCCESS:
      return False
    if self.closed:
      return True

    if not self.subgoals_closed():
      return False

    self.closed = True
    # We are marking the associated node closed. Note that this is a recursive
    # call and might update more associated TacticApplications upstream.
    self.parent.mark_closed(self)
    return True

  def mark_failed(self):  # DEPRECATED
    """Mark this tactic-application failed if any of the subgoals has failed.

    Note that having "failed" is a soft condition, not a definitive one. Right
    now, the hard-coded behavior is to "fail" if no tactic could is applied
    in a way that changes the goal without producing error. However, this
    behavior might be overridden.
    """
    if self.failed:
      # Nothing to do, we are already marked as a failure.
      # Make sure that we have not marked this node closed. That would be a
      # contradiction.
      assert not self.closed
      return
    for subgoal in self.subgoals:
      if subgoal.failed:
        # We have found a failing subgoal.
        # Make sure that we have not marked this node closed. That would be a
        # contradiction.
        assert not self.closed
        # This tactic application is failed since we can't close it anymore.
        self.failed = True
        # Update the parent node to be failed if all of its tactic applications
        # have failed.
        self.parent.update_failed()
        # We would like to mark all non-failing sibling nodes and their
        # descendants useless if they have not chance of contributing to closing
        # any other goal.
        for sibling in self.subgoals:
          sibling.update_ignore()
        # Don't do duplicated work.
        return

  def subgoals_closed(self) -> bool:
    """Checks if all subgoals of this tactic are closed."""
    return all(node.closed for node in self.subgoals)

  def close_targets(self):
    """Closes the subgoals which reached targets."""
    tree = self.parent.tree
    for node in self.subgoals:
      if tree.within_targets(node.goal):
        node.closed = True

  def add_to_node_proto(self, node_proto: deephol_pb2.ProofNode):
    """Attempt to add a node to the proof search tree.

    Fails if tactic formatis invalid.
    Args:
      node_proto: Proof node which new tactic application is added to.
    """
    node_proto.proofs.add(
        tactic=self.tactic,
        parameters=self.parameters,
        subgoals=[sg.goal for sg in self.subgoals],
        result=self.result,
        error_message=self.error_message,
        time_spent=self.time_spent,
        closed=self.closed,
        score=self.score,
        rank=self.rank)


class ProofSearchNode(object):
  """Node in the proof tree, corresponding to one goal."""

  def __init__(self,
               tree: ProofSearchTree,
               index: int,
               goal: proof_assistant_pb2.Theorem,
               parent: Optional[SubGoalRef] = None):
    """Constructor for a Node within proof search.

    Each node represents a goal or subgoal with one or multiple proof attempts.
    Proof attempts are tactic_applications that can generate one or multiple
    subgoals.

    Args:
      tree: The ProofSearchTree object to which this node belongs.
      index: Index of this node in the list of nodes of the search tree.
      goal: Actual goal to be proved.
      parent: The source of this goal. If it is None, then it must be the root
        of the search tree. Otherwise it must be a SubGoalRef referring to the
        tactic application that created the proof search node.
    """
    self.tree = tree
    self.goal = goal
    if not self.goal.fingerprint:
      self.goal.fingerprint = theorem_fingerprint.Fingerprint(goal)
    self.normalized_goal = None
    self.index = index
    if parent is not None:
      self.parents = [parent]
    else:
      self.parents = []
    # The list of the successful tactic applications. Note that elements
    # of this list might refer to subtrees that are not or can't be closed
    # successfully.
    self.successful_attempts = []
    # The list of the failed tactic applications.
    self.failed_attempts = []
    # Here, we have three options:
    # - None: we have attempted no tactics yet
    # - False: the tree was expanded at this node, but it is not closed yet.
    # - True: We have at least one proof attempt that was successful.
    #         Note: Success means reaching a subset of the target goals.
    self.closed = None
    # This is a temporary marker: we say that a node has failed if
    # all of its proof attempts have failed.
    self.failed = None
    # This is a temporary marker: we set a node to be *ignored* if it is useless
    # to close, that is:
    # - it has at least one parent (that is: it is not the root node) and
    # - each of its parent is to set to ignore.
    #
    # A node should be ignored if it is not helpful to close any of its subgoals
    # anymore, bacause its parent is set to ignored due to
    # - its being hopeless to close (i.e. failed) or
    # - its being useless to close, since some of the ancestors participates
    #   only in tactic applications where at least one of the subgoals has
    #   failed.
    #
    # A node will be ignored if either
    # - All of its parents (participating tactic applications) are failed or
    #   ignored
    # - One of its sibling has failed.
    # The ignore flag is propagated from failed nodes in the following way:
    # - Once a node is set to failed, then all of its siblings are set to ignore
    # - All descendants of ignored nodes are marked as ignored as long as all
    #   their other parents are ignored or failed too.
    #
    #   Once a non-failed node becomes descendant of a non-ignored
    #   node again (as a shared node), then the ignored flags are removed for
    #   all of its non-failed descendants (that have a chance to close).
    self.ignore = False
    # Set to true if initial tactics are applied.
    self.processed = False
    # Total time or step function of the action generator.
    self.action_generation_time_millisec = None
    # Time to embed the proof state. Part of action_generation_time_millisec.
    self.proof_state_emb_time_ms = None
    # Time to compute theorem scores. Part of action_generation_time_millisec.
    self.theorem_scores_time_ms = None
    # Time to rank assumptions. Part of theorem_scores_time_ms.
    self.assumptions_ranking_time_ms = None
    # Time to compute the DeepHOL zero heuristic for similar premises.
    self.heuristic_ranking_time_ms = None

  def get_normalized_goal(self):
    if self.normalized_goal is None:
      self.normalized_goal = normalization_lib.normalize(self.goal)
    return self.normalized_goal

  def update_ignore(self):
    """Update the ignore flag on the node all descendants if warrented."""
    if self.ignore:
      # Don't do double work if the node is already ignored.
      return
    if not self.parents:
      # Never ignore the root node. It might fail, but never gets ignored.
      return
    for parent in self.parents:
      tac_app = parent.tactic_application
      if not (tac_app.failed or tac_app.parent.ignore):
        # This node might be useful for closing this tactic application.
        return
    # This node is useless as closing it will not help the final goal:
    # - The node is not the root node.
    # - Either the tactic application has already failed or
    # - the goal of each parent tactic application is already useless.
    self.ignore = True
    # Now, we need to update all the children in all current tactic applications
    for tac_app in self.successful_attempts:
      for subgoal in tac_app.subgoals:
        # Mark all subgoals in the proof attempts as ignore, since it is
        # useless to close this goal.
        subgoal.update_ignore()

  # If a node gets a new non-ignored parent, then we should mark it and
  # all its useful descendants non-ignored, unless all proof attempts
  # of the node have failed.
  def remove_ignore(self):
    """Clear the ignore flag on the node if warranted by the circumnstances.

    The use case of this function is to re-enable ignored nodes if they
    become interesting again, since they show up as result of tactic
    applications on newly expanded nodes. In this case, we might be forced
    to remove the ignore flag on the node if the node has become interesting
    again.
    """
    # We only remove ignore from a node if there is something interesting
    # we can do about this node.
    # If it is failed, then there is nothing we can do
    # If it is closed, then there is nothing to do
    # If it is not yet ignored, then we don't need to mark it not-ignored.
    if self.failed or self.closed or not self.ignore:
      return
    remove_ignore = False
    for parent in self.parents:
      tac_app = parent.tactic_application
      if not (tac_app.failed or tac_app.parent.ignore):
        remove_ignore = True
        break
    if remove_ignore:
      self.ignore = False
    else:
      if self.parents:
        # We are hopless anyways. There is no reason to close to
        return
      else:
        # The root should never be set to ignore.
        self.ignore = False
    might_close = False
    for tac_app in self.successful_attempts:
      if tac_app.closed:
        self.mark_closed(tac_app)
        break
      if not tac_app.failed:
        might_close = True
    if might_close and not self.closed:
      # Recursively remove the ignore flag in all descendants
      # that have a chance to close.
      for tac_app in self.successful_attempts:
        if not tac_app.failed:
          assert not tac_app.closed
          for subgoal in tac_app.subgoals:
            subgoal.remove_ignore()

  def mark_closed(self, tactic_application: TacticApplication):
    """Mark the proof search node closed.

    This function assumes that we have a fully closed subtree on one of
    the TacticApplications in successful_attempts. Assertion will fail
    if this is not the case.

    Args:
      tactic_application: The TacticApplication that has lead to a proof. This
        parameter is only used for verifying that the node is really closed.
        Assertions will fail if that's not the case.
    """
    assert self.closed is not None
    if self.closed:
      return
    # Check that the tactic_application belongs to this node.
    assert tactic_application.parent.index == self.index
    # Make sure that all subgoals are really closed.
    assert tactic_application.subgoals_closed()
    self.closed = True
    # Now, we don't want to close this goal again. We ignore it
    # for all further attempts.
    self.ignored = True
    # For all other non-closed tactic application, ignore the children
    # if they don't need to be closed as other subgoals.
    for tac_app in self.successful_attempts:
      if not tac_app.closed:
        for subgoal in tac_app.subgoals:
          # Mark all subgoals in the other proof attempts as ignore, they
          # have become useless for closing this goal.
          subgoal.update_ignore()
    # Note that update_closed does not necessarily close the
    # parent. Only, if all of their subgoals are closed.
    # In general, we have a recursive call, that might mark
    # all the closed (sub-)goals closed in the relevant part
    # of the search tree, if they got proved.
    for subgoal_ref in self.parents:
      subgoal_ref.tactic_application.update_closed()

  def update_failed(self):  # DEPRECATED
    """Update the not to be failed if there is no chance to close it."""
    if self.closed:
      # This node can't fail as it is already closed.
      return
    # Mark this node to be failed if all of its tactic applications have failed.
    for tac_app in self.successful_attempts:
      if not tac_app.failed:
        # We have a chance to close some of these subgoals
        return
    self.failed = True
    for subgoal_ref in self.parents:
      subgoal_ref.tactic_application.mark_failed()
    # Mark this node and its descendants to be ignored.
    self.update_ignore()

  def apply_tactic_legacy(self, tactic_application: Text, timeout_ms: int,
                          score: float) -> TacticApplication:
    """Legacy function that supports tactic strings including parameters."""
    tactic, parameters = tactic_utils.extract_tactic_and_parameters(
        self.goal, tactic_application)
    return self.apply_tactic(tactic, parameters, timeout_ms, score)

  def apply_tactic(self, tactic: Text,
                   parameters: List[deephol_pb2.TacticParameter],
                   timeout_ms: int, score: float) -> TacticApplication:
    """Wrapper around a proof assistant's ApplyTactic.

    This function is a wrapper around a proof assistant's ApplyTactic.
    TacticApplication objects are stored as elements in the
    {successful/failed}_attempts fields of ProofSearchNode. These represent
    starts of proof attempts for particular goals or subgoals.

    If tactic could not be applied, timed out or did not change the goal, then
    the application is added to self.failed_attempts and the index will refer
    to this list. The result field must be any value different from
    deephol_pb2.TacticApplication.SUCCESS in this case.

    If the tactic is applied successfully, then this application is added to
    self.successful_attempts and the index will refer to this list. The result
    field must contain deephol_pb2.TacticApplication.SUCCESS in this case.

    Args:
      tactic: Tactic string to be applied, NOT including the parameter list.
      parameters: parameter list, e.g. the premises of the tactic.
      timeout_ms: Timeout in milliseconds for ApplyTactic.
      score: Score produced by the action generator.

    Returns:
      Tactic application result after calling the proof assistant.
    """
    assert self.tree.proof_assistant is not None

    if self.closed is None:
      self.closed = False

    rank = len(self.failed_attempts) + len(self.successful_attempts)

    if self.goal.tag != proof_assistant_pb2.Theorem.GOAL:
      raise ValueError('Cannot apply tactic to Theorem with tag %s' %
                       str(self.goal.tag))

    tactic_application_string = tactic_utils.tactic_string(tactic, parameters)
    request = proof_assistant_pb2.ApplyTacticRequest(
        goal=self.goal, tactic=tactic_application_string, timeout_ms=timeout_ms)

    start_time = time.time()
    try:
      response = self.tree.proof_assistant.ApplyTactic(request)
      elapsed_msecs = int((time.time() - start_time) * 1000.0 + 0.5)
    except error.StatusNotOk as exception:
      elapsed_msecs = int((time.time() - start_time) * 1000.0 + 0.5)
      tf.logging.info('Tactic application failed: %s with error %s',
                      tactic_application_string, exception.message)
      # Sometimes, rarely, the prover gets into a situation in which it stops
      # communicating and eventually requests hang. However we
      # can bail out before that happen and can prevent the whole
      # program to hang for a long time.
      if (exception.message.startswith('Communication') and
          exception.message.endswith('failed.')):
        tf.logging.info(
            'Communication error with proof assistant on request:'
            '\n%s', request)
        raise exception
      application = TacticApplication(
          parent=self,
          tactic=tactic,
          parameters=parameters,
          score=score,
          rank=rank,
          index=len(self.failed_attempts),
          result=deephol_pb2.TacticApplication.ERROR,
          error_message=exception.message,
          time_spent=elapsed_msecs,
          subgoals=[],
          closed=False,
          failed=True)
      self.failed_attempts.append(application)
      return application

    if response.HasField('error'):
      tf.logging.info('Tactic application has error: %s, %s',
                      tactic_application_string, response.error)
      application = TacticApplication(
          parent=self,
          tactic=tactic,
          parameters=parameters,
          score=score,
          rank=rank,
          index=len(self.failed_attempts),
          result=deephol_pb2.TacticApplication.ERROR,
          error_message=response.error,
          time_spent=elapsed_msecs,
          subgoals=[],
          closed=False,
          failed=True)
      self.failed_attempts.append(application)
      return application

    assert response.HasField('goals')
    new_subgoals = list(response.goals.goals)
    for subgoal in new_subgoals:
      theorem_utils.convert_legacy_goal(subgoal)

    if len(new_subgoals) == 1 and _is_same_goal(request.goal, new_subgoals[0]):
      tf.logging.info('Tactic %s applied, but did not change subgoals.',
                      tactic_application_string)
      application = TacticApplication(
          parent=self,
          tactic=tactic,
          parameters=parameters,
          score=score,
          rank=rank,
          index=len(self.failed_attempts),
          result=deephol_pb2.TacticApplication.UNCHANGED,
          error_message=None,
          time_spent=elapsed_msecs,
          subgoals=[],
          closed=False,
          failed=True)
      self.failed_attempts.append(application)
      return application

    # Check whether new subgoals are properly parenthesized
    message = validate_parentheses(new_subgoals)
    if message is not None:
      application = TacticApplication(
          parent=self,
          tactic=tactic,
          parameters=parameters,
          score=score,
          rank=rank,
          index=len(self.failed_attempts),
          result=deephol_pb2.TacticApplication.ERROR,
          error_message=message,
          time_spent=elapsed_msecs,
          subgoals=[],
          closed=False,
          failed=True)
      self.failed_attempts.append(application)
      return application

    # We have a successful tactic application.
    application = TacticApplication(
        parent=self,
        tactic=tactic,
        parameters=parameters,
        score=score,
        rank=rank,
        index=len(self.successful_attempts),
        result=deephol_pb2.TacticApplication.SUCCESS,
        error_message=None,
        time_spent=elapsed_msecs,
        subgoals=[],
        closed=False,
        failed=False)
    for i, goal in enumerate(new_subgoals):
      if goal.tag != proof_assistant_pb2.Theorem.GOAL:
        raise ValueError('HOL Light response subgoal without GOAL tag.')
      subgoal = proof_assistant_pb2.Theorem()
      subgoal.CopyFrom(goal)
      subgoal.fingerprint = theorem_fingerprint.Fingerprint(subgoal)
      subgoal_ref = SubGoalRef(tactic_application=application, subgoal_index=i)
      application.subgoals.append(self.tree.add_node(subgoal, subgoal_ref))
    tf.logging.info('Tactic %s successfully applied.',
                    tactic_application_string)
    self.successful_attempts.append(application)
    application.close_targets()
    if application.subgoals_closed():
      assert application.update_closed()
    return application

  def deserialize_tactic_application(
      self, proto: deephol_pb2.TacticApplication) -> TacticApplication:
    """Deserializes proto to create a tactic application coming from this node.

    Args:
      proto: Tactic application proto to be deserialized.

    Returns:
      Deserialized tactic application.
    """
    if self.closed is None:
      self.closed = False
    failed = (proto.result != deephol_pb2.TacticApplication.SUCCESS)
    index = (
        len(self.failed_attempts) if failed else len(self.successful_attempts))

    # We do not compute the rank (i.e. order wrt action generator score).
    application = TacticApplication(
        parent=self,
        tactic=proto.tactic,
        parameters=proto.parameters,
        score=proto.score,
        rank=None,
        index=index,
        result=proto.result,
        error_message=proto.error_message,
        time_spent=proto.time_spent,
        subgoals=[],
        closed=False,
        failed=failed)

    if failed:
      if proto.subgoals:
        raise ValueError('Failed TacticApplication with subgoals.')
      self.failed_attempts.append(application)
    else:
      if application.error_message:
        raise ValueError(
            'TacticApplication with SUCCESS and error message: %s' %
            application.error_message)
      for i, goal in enumerate(proto.subgoals):
        if goal.tag != proof_assistant_pb2.Theorem.GOAL:
          raise ValueError('TacticApplication subgoal without GOAL tag.')
        subgoal = proof_assistant_pb2.Theorem()
        subgoal.CopyFrom(goal)
        subgoal.fingerprint = theorem_fingerprint.Fingerprint(subgoal)
        subgoal_ref = SubGoalRef(
            tactic_application=application, subgoal_index=i)
        application.subgoals.append(self.tree.add_node(subgoal, subgoal_ref))
      self.successful_attempts.append(application)
      application.close_targets()
      if application.subgoals_closed():
        assert application.update_closed()
    return application


def validate_parentheses(
    goals: List[proof_assistant_pb2.Theorem]) -> Optional[Text]:
  """Checks whether input goals are properly parenthesized.

  Args:
    goals: assumptions and conclusion of each goal are checked.

  Returns:
    Error message string, or None if all goals are properly parenthesized.
  """
  assert all([g.tag == proof_assistant_pb2.Theorem.GOAL for g in goals])
  for goal in goals:
    try:
      if not sexpression_parser.is_bare_word(goal.conclusion):
        sexpression_parser.validate_parens(goal.conclusion)
    except sexpression_parser.SExpParseError as exception:
      message = ('Received improperly parenthesized conclusion. '
                 'Exception message: %s' % str(exception))
      tf.logging.info(message)
      return message
    for assumption in goal.assumptions:
      try:
        if not sexpression_parser.is_bare_word(assumption.conclusion):
          sexpression_parser.validate_parens(assumption.conclusion)
      except sexpression_parser.SExpParseError as exception:
        message = ('Received improperly parenthesized conclusion. '
                   'Exception message: %s' % str(exception))
        tf.logging.info(message)
        return message
      for hypothesis in assumption.hypotheses:
        try:
          if not sexpression_parser.is_bare_word(hypothesis):
            sexpression_parser.validate_parens(hypothesis)
        except sexpression_parser.SExpParseError as exception:
          message = ('Received improperly parenthesized hypothesis. '
                     'Exception message: %s' % str(exception))
          tf.logging.info(message)
          return message
  return None


def check_tree_consistency(tree: ProofSearchTree):
  """Checks the consistency of the proof search tree.

  Verifies that the cross-reference indices are set correctly.
  It also checks that the "closed" flags are propagated correctly through
  the tree.

  Args:
    tree: Reference to the proof search tree to be checked.
  """
  for i, node in enumerate(tree.nodes):
    assert i == node.index, ('Inconsistent node index %d != %d' %
                             (node.index, i))
    assert tree is node.tree, ('Inconsistent tree for node %d' % i)
    if i == 0:
      assert not node.parents, 'Root node with parent'
    else:
      assert node.parents, ('Non-root node %d without parent' % i)
      for parent in node.parents:
        tapp = parent.tactic_application
        assert tapp.subgoals[parent.subgoal_index] is node
    for j, tapp in enumerate(node.failed_attempts):
      assert j == tapp.index, ('Index mismatch %d != %d' % (j, tapp.index))
      assert tapp.result != deephol_pb2.TacticApplication.UNKNOWN
      assert tapp.result != deephol_pb2.TacticApplication.SUCCESS
      assert not tapp.subgoals, ('Failed attempts with subgoals %d %d' % (i, j))
    closed = False
    for j, tapp in enumerate(node.successful_attempts):
      assert j == tapp.index, ('Inconsistent TacticApplication index %d != %d' %
                               (j, tapp.index))
      assert tapp.result != deephol_pb2.TacticApplication.UNKNOWN
      assert tapp.result == deephol_pb2.TacticApplication.SUCCESS
      assert not tapp.error_message, ('Successful attempt with error %s %d %d' %
                                      (tapp.error_message, i, j))
      all_goals_closed = True
      for goal in tapp.subgoals:
        if not goal.closed:
          all_goals_closed = False
      if all_goals_closed:
        assert tapp.closed, ('All subgoals closed for %d:%d but tapp is '
                             'is not closed' % (i, j))
        closed = True
      assert all_goals_closed == tapp.closed, ('Inconsistent closed mark '
                                               '%d:%d' % (i, j))
    if not node.failed_attempts and not node.successful_attempts:
      closed = None
    assert closed == node.closed, ('Inconsistent closed mark for node %d' % i)


def tree_from_proof_log(
    proof_log: deephol_pb2.ProofLog,
    forward_proving=False
) -> Tuple[ProofSearchTree, Dict[TacticApplication,
                                 deephol_pb2.TacticApplication]]:
  """Deserializes input proof log into a proof search tree.

  Args:
    proof_log: Proof log to be deserializes into a proof search tree.
    forward_proving: tag where whether or not we use forward proving: if we do,
      then ProofSearchTree is initialized with forward proving.

  Returns:
    1) Deserialized proof search tree.
    2) Dictionary mapping all the tactic applications in the proof search tree
       to their tactic application proto counterparts in the input proof log.
       The dictionary is computed due to our deserialization being lossy, e.g.,
       we cannot recover tactic parameters used in human proof logs that are
       unsupported in our current prover setup.
  """
  # TODO(vtoman): extend deserialization capabilities of tactic application so
  # it is lossless and we no longer need the application<->proto dictionary.
  tree = None
  application_to_proto_map = {}
  fingerprint_to_proof_log_node = {}
  stack = []
  visited_fingerprints = set()

  for proof_log_node in proof_log.nodes:
    fingerprint = theorem_fingerprint.Fingerprint(proof_log_node.goal)
    if fingerprint in fingerprint_to_proof_log_node:
      fingerprint_to_proof_log_node[fingerprint].append(proof_log_node)
    else:
      fingerprint_to_proof_log_node[fingerprint] = [proof_log_node]
    if proof_log_node.root_goal:
      if tree is not None:
        raise ValueError('Cannot convert proof log with multiple root goals to '
                         'a proof search tree.')
      if proof_log_node.goal.tag != proof_assistant_pb2.Theorem.GOAL:
        raise ValueError('Root ProofLog goal without GOAL tag.')
      root_goal = proof_assistant_pb2.Theorem()
      root_goal.CopyFrom(proof_log_node.goal)
      tree = ProofSearchTree(proof_assistant_obj=None, goal=root_goal)
      assert len(tree.nodes) == 1
      stack.append(tree.nodes[0])
      visited_fingerprints.add(fingerprint)
  if tree is None:
    raise ValueError('Cannot convert proof log with no root goal to a proof '
                     'search tree.')

  if proof_log.HasField('prover_task'):
    tree.prover_task = proof_assistant_pb2.ProverTask()
    tree.prover_task.CopyFrom(proof_log.prover_task)

  while stack:
    proof_search_node = stack.pop()
    fingerprint = theorem_fingerprint.Fingerprint(proof_search_node.goal)
    if fingerprint not in fingerprint_to_proof_log_node:
      raise ValueError('In ProofLog, there is TacticApplication with a subgoal '
                       'such that there is no ProofNode with such goal.')
    # We deduplicate proof log nodes with the same goal, but we retain
    # the tactic applications coming from all of the duplicates
    for proof_log_node in fingerprint_to_proof_log_node[fingerprint]:
      for tactic_application_proto in proof_log_node.proofs:
        tactic_application = proof_search_node.deserialize_tactic_application(
            tactic_application_proto)
        application_to_proto_map[tactic_application] = tactic_application_proto
        for new_search_node in tactic_application.subgoals:
          new_search_node_fingerprint = theorem_fingerprint.Fingerprint(
              new_search_node.goal)
          if new_search_node_fingerprint not in visited_fingerprints:
            stack.append(new_search_node)
            visited_fingerprints.add(new_search_node_fingerprint)

  if forward_proving:
    tree.forward_proving = forward_proving

  return tree, application_to_proto_map


def proof_state_from_proof_search_node(
    node: ProofSearchNode,
    history_bound: Optional[int] = None,
    visited_fingerprints: Optional[Set[int]] = None) -> predictions.ProofState:
  """Creates a Proof State object that will be passed to the predictor.

  The computed Proof State is a linked list of states where the top state
  corresponds to the input proof search node. We recursively process ancestors
  (up to the specified bound), collect their respective proof states,
  and link them together.

  Args:
    node: Node in the proof search tree.
    history_bound: How much (if any) history of the proof node is collected.
      'None' bound means all history is collected.
    visited_fingerprints: Internal tracker that maintains fingerprints of goals
      visited during proof state history construction. Call this function
      without specifying this argument, so the default None is used.

  Returns:
    Proof State corresponding to the input proof search node.
  """
  assert history_bound is None or history_bound >= 0
  if visited_fingerprints is None:
    visited_fingerprints = set()
  if not node.goal.HasField('fingerprint'):
    raise ValueError('Goal in ProofLogNode has no fingerprint.')
  goal_fp = node.goal.fingerprint
  assert goal_fp not in visited_fingerprints
  visited_fingerprints.add(goal_fp)

  previous_proof_state = None
  if (node.parents and node.index > 0 and
      (history_bound is None or len(visited_fingerprints) <= history_bound)):
    # A proof node may have multiple parents in the proof graph, we consider
    # the first one as the canonical one for the purpose of previous proof state
    parent_proof_node = node.parents[0].tactic_application.parent
    if parent_proof_node is None:
      raise ValueError('Ill-formed proof graph - proof node was created by '
                       'tactic application coming from None.')
    if not parent_proof_node.goal.HasField('fingerprint'):
      raise ValueError('Goal in ProofLogNode has no fingerprint.')
    parent_goal_fp = parent_proof_node.goal.fingerprint
    if parent_goal_fp not in visited_fingerprints:
      previous_proof_state = proof_state_from_proof_search_node(
          parent_proof_node, history_bound, visited_fingerprints)
  normalized_goal = node.get_normalized_goal()

  return predictions.ProofState(
      goal=normalized_goal,
      targets=node.tree.targets,
      previous_proof_state=previous_proof_state)


GoalTacticExample = Tuple[Tuple[ProofSearchNode, TacticApplication],
                          Optional[List[proof_assistant_pb2.Theorem]]]
GoalTacticExamples = Iterable[GoalTacticExample]


def tactic_application_examples(
    proof_tree: ProofSearchTree,
    process_all: bool = False,
) -> GoalTacticExamples:
  """Generate training examples for each tactic application in the search tree.

  Args:
    proof_tree: Search tree to generate edges for.
    process_all: Enable if to include failed tactic applications.

  Yields:
    A sequence of ((proof search node, tactic), target goals) used in RL
    training loop.
  """
  targets = None if not proof_tree.targets else proof_tree.targets
  for node in proof_tree.nodes:
    applications = node.successful_attempts
    if process_all:
      applications += node.failed_attempts
    else:
      applications = [tactic for tactic in applications if tactic.closed]

    for application in applications:
      yield (node, application), targets


def her_examples(proof_graph: ProofSearchTree,
                 sources_per_target: int = 3) -> GoalTacticExamples:
  """Generate hindsight experience replay examples in via random-DFS.

  Note: As written, this does not output the closed proof target. This
  is assumed to be taken care of by another example generator, e.g.,
  `edges_to_examples`.

  Args:
    proof_graph: `ProofSearchTree` object to generate hindsight examples from.
      Called `proof_graph` here since its typically not a tree which matters
      during DFS.
    sources_per_target: controls the number of sources generated for each target
      goal.

  Yields:
    A sequence of ((proof search node, tactic), target goals) used in RL
    training loop.
  """
  path = []  # Sequence of (prev_tactic, goal) along proof.
  root = proof_graph.nodes[0]
  stack = [(0, (None, root))]  # [(depth, (prev_tactic, goal))].
  visited = set()

  while stack:  # DFS loop
    depth, (prev_tactic, goal) = stack.pop()

    if depth > len(path):  # Backtrack?
      del path[depth:]
    path.append((prev_tactic, goal))

    if goal in visited:
      continue  # Avoid cycles + bias towards unseen subgraphs.
    visited.add(goal)
    stack.extend(_successors_hindsight(goal, depth=depth))

    yield from _hindsight_examples_from_path(path, sources_per_target)


def _hindsight_examples_from_path(path, srcs_per_tgt) -> GoalTacticExamples:
  """Sample training examples from partial path in search tree.

  Converts path into training examples by:
     1. Assuming the target was the final goal reached along path.
     2. Sampling (goal, tactic) pairs from prefix of path.

  Args:
    path: A list of (previous tactic, current node) pairs.
    srcs_per_tgt: The number of hindsight examples to generate.

  Yields:
    A sequence of ((proof search node, tactic), target goals) sampled from
    path.
  """
  if len(path) < 2:
    return  # Need result of tactic application.

  tgt = path[-1][1]

  # Sample sources by sampling consecutive path elements.
  size = len(path) - 1
  indices = random.sample(range(size), k=min(size, srcs_per_tgt))
  for idx in indices:
    _, goal = path[idx]
    tactic, _ = path[idx + 1]
    yield (goal, tactic), [tgt.goal]


def _successors_hindsight(goal, depth=0):
  kids = []
  for tactic in goal.successful_attempts:
    kids.extend([(tactic, subgoal) for subgoal in tactic.subgoals])
  random.shuffle(kids)

  # Annotate with increased depth.
  return [(depth + 1, kid) for kid in kids]
