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

from __future__ import absolute_import
from __future__ import division
# Import Type Annotations
from __future__ import print_function

import collections
import time

import tensorflow as tf
from typing import List, Optional, Tuple, Text
from deepmath.deephol.public import proof_assistant
from deepmath.deephol import deephol_pb2
from deepmath.deephol import theorem_fingerprint
from deepmath.proof_assistant import proof_assistant_pb2
from deepmath.public import error


def _extract_tactic_and_parameters(
    tactic_string: Text) -> Tuple[Text, List[deephol_pb2.TacticParameter]]:
  """Extract the tactic string and its parameter list from a string.

  Args:
    tactic_string: The tactic application string to be passed to ocaml.

  Returns:
    A pair of tactic name and tactic parameter list.
  """
  if '[' in tactic_string:
    s = tactic_string.replace(']', '').split('[')
    assert len(s) == 2, ('Expected single argument %s' % tactic_string)
    theorems = []
    for param_string in s[1].split(';'):
      ps = param_string.strip()
      if ps:
        t = ps.split()
        assert len(t) == 2, ('Invalid tactic parameter "%s"' % ps)
        assert t[0] == 'THM', ('Invalid tactic parameter "%s"' % ps)
        theorems.append(proof_assistant_pb2.Theorem(fingerprint=int(t[1])))
    return s[0].strip(), [
        deephol_pb2.TacticParameter(
            parameter_type=deephol_pb2.Tactic.THEOREM_LIST, theorems=theorems)
    ]
  else:
    s = tactic_string.split()
    if len(s) == 1:
      return s[0], []
    else:
      assert len(s) == 3
      assert s[1] == 'THM'
      return s[0], [
          deephol_pb2.TacticParameter(
              parameter_type=deephol_pb2.Tactic.THEOREM,
              theorems=[proof_assistant_pb2.Theorem(fingerprint=int(s[2]))])
      ]


def _theorem_to_string(thm: proof_assistant_pb2.Theorem) -> Text:
  """Turn the theorem into a string for map lookup.

  Args:
    thm: Theorem to turn to string format.

  Returns:
    String joining the hypotheses and the conclusion by '|:|'- separators.
  """
  return '|:|'.join([str(hyp) for hyp in thm.hypotheses] +
                    [str(thm.conclusion)])


# To reference a SubGoal of a tactic application, we need the following two
# pieces of information:
# - The TacticApplication that generated this subgoal.
# - The index of the subgoal in the list of subgoals.
# Note that we can't use NamedTuple, since TacticApplication can't be
# referenced due to recursive definitions.
SubGoalRef = collections.namedtuple('SubGoalRef',
                                    ['tactic_application', 'subgoal_index'])


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
               parent: Optional[SubGoalRef]):
    """Append a new node to the tree."""
    goal_as_string = _theorem_to_string(goal)
    if goal_as_string in self.nodes_map:
      node = self.nodes[self.nodes_map[goal_as_string]]
      # Make sure that we really match everything exactly
      assert len(node.goal.hypotheses) == len(goal.hypotheses)
      for i, hyp in enumerate(goal.hypotheses):
        assert hyp == node.goal.hypotheses[i]
      assert goal.conclusion == node.goal.conclusion
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

  def __init__(self, proof_assistant_obj: proof_assistant.ProofAssistant,
               goal: proof_assistant_pb2.Theorem):
    """Constructor for a proof search tree.

    Args:
      proof_assistant_obj: An interface to the proof assistant.
      goal: The root goal which is also used to limit the premise selection to
        preceding theorems. This is the first theorem in the theorem database
        that is not allowed to be used in the proof. For now, it is mandatory
        that the goal is in the theorem database. Later, we should relax this
        constraint.
    """
    self.proof_assistant = proof_assistant_obj
    self.nodes = []
    self.nodes_map = {}
    root = self.add_node(goal, None)
    assert root.index == 0
    self.cur_index = None

  def to_proto(self) -> deephol_pb2.ProofLog:
    """Serialize the proof search tree as a protobuf.

    Returns:
      A deephol_pb2.ProofLog protobuf representing the whole proof search tree.
    """
    proof_log = deephol_pb2.ProofLog()
    for node in self.nodes:
      status = deephol_pb2.ProofNode.UNKNOWN
      if node.closed:
        status = deephol_pb2.ProofNode.PROVED
      node_log = proof_log.nodes.add(
          goal=node.goal,
          status=status,
          action_generation_time_millisec=node.action_generation_time_millisec)
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
      successful_attempts: List[int],
      failed_attempts: List[int],
      tree: ProofSearchTree,
      request: proof_assistant_pb2.ApplyTacticRequest,
      score: float):
    """Constructor for the result of a tactic application.

    This function is a wrapper around a proof assistant's ApplyTactic.
    TacticApplication objects are always stored as elements in the
    tactic_applications field of SearchNode. These represents starts of
    proof attempts for particular goals or subgoals.

    Args:
      parent: ProofSearchNode to which the tactic was applied to.
      successful_attempts: List of successful tactic applications. If the tactic
        is applied successfully, then this application is added to this list and
        the index will refer to this list. The result field must contain
        deephol_pb2.TacticApplication.SUCCESS in this case.
      failed_attempts: List of failed tactic applications. If tactic could not
        be applied, timed out or did not change the goal, then the application
        is added to this list and the index will refer to this list. The result
        field must be any value different from
        deephol_pb2.TacticApplication.SUCCESS in this case.
      tree: ProofSearchTree to which this application belongs to.
      request: Tactic-application request to be run.
      score: Score produced by the action generator.
    """
    self.parent = parent
    # Index of the tactic application in either (successful or failed) list of
    # proof attempts in the ProofSearchNode. Will be filled once it is clear if
    # the application was successful or not.
    self.index = None
    self.result = None
    self.error_message = None
    self.time_spent = None
    # List of ProofSearchNodes corresponding to the subgoals of this tactic.
    self.subgoals = []
    self.tactic = request.tactic
    self.closed = False  # True if all subgoals are closed.
    self.failed = False  # True if any of the subgoals are failed to close.
    self.score = score
    self.rank = len(failed_attempts) + len(successful_attempts)
    start_time = time.time()
    try:
      response = tree.proof_assistant.ApplyTactic(request)
      elapsed_msecs = int((time.time() - start_time) * 1000.0 + 0.5)
      self.time_spent = elapsed_msecs
    except error.StatusNotOk as exception:
      elapsed_msecs = int((time.time() - start_time) * 1000.0 + 0.5)
      self.time_spent = elapsed_msecs
      tf.logging.info('Tactic application failed: %s with error %s',
                      str(self.tactic), exception.message)
      self.result = deephol_pb2.TacticApplication.ERROR
      self.failed = True
      self.error_message = exception.message
      self.index = len(failed_attempts)
      failed_attempts.append(self)
      # Sometimes, rarely, the prover gets into in which it stops
      # communicating and eventually requests hang. However we
      # can bail out before that happen and can prevent the whole
      # program to hang for a long time.
      if str(exception).startswith('Communication') and str(exception).endswith(
          'failed.'):
        raise exception
      return
    if response.HasField('error'):
      tf.logging.info('Tactic application failed: %s, %s', str(request.tactic),
                      response.error)
      self.result = deephol_pb2.TacticApplication.ERROR
      self.failed = True
      self.error_message = response.error
      self.index = len(failed_attempts)
      failed_attempts.append(self)
      return
    assert response.HasField('goals')
    new_subgoals = list(response.goals.goals)

    def is_same_expr(t1, t2):
      return t1.conclusion == t2.conclusion and t1.hypotheses == t2.hypotheses

    if len(new_subgoals) == 1 and is_same_expr(request.goal, new_subgoals[0]):
      tf.logging.info('Tactic %s applied, but did not change subgoals.',
                      request.tactic)
      self.result = deephol_pb2.TacticApplication.UNCHANGED
      self.failed = True
      self.index = len(failed_attempts)
      failed_attempts.append(self)
      return
    # We have a successful tactic application.
    assert not self.subgoals
    self.index = len(successful_attempts)
    for i, goal in enumerate(new_subgoals):
      thm = proof_assistant_pb2.Theorem(
          hypotheses=goal.hypotheses,
          conclusion=goal.conclusion,
          pretty_printed=goal.pretty_printed,
          tag=proof_assistant_pb2.Theorem.GOAL)
      subgoal_ref = SubGoalRef(tactic_application=self, subgoal_index=i)
      self.subgoals.append(tree.add_node(thm, subgoal_ref))
    self.result = deephol_pb2.TacticApplication.SUCCESS
    # We don't know if some of the subgoals will fail or not.
    self.failed = False
    tf.logging.info('Tactic %s successfully applied.', self.tactic)
    successful_attempts.append(self)
    if not new_subgoals:
      assert self.update_closed()

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
    for subgoal in self.subgoals:
      if not subgoal.closed:
        return False
    self.closed = True
    # We are marking the associated node closed. Note that this is a recursive
    # call and might update more associated TacticApplications upstream.
    self.parent.mark_closed(self)
    return True

  def mark_failed(self):
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

  def add_to_node_proto(self, node_proto: deephol_pb2.ProofNode):
    tactic, parameters = _extract_tactic_and_parameters(str(self.tactic))
    node_proto.proofs.add(
        tactic=tactic,
        parameters=parameters,
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
    # - False: the tree was expanded at this node, but it is not closed yet
    # - True: We have at least one proof attempt that was successful.
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
    # Action generation happens only once, when node is processed.
    self.action_generation_time_millisec = None

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
    for tac_app in self.successfull_attempts:
      if tac_app.closed:
        self.mark_closed(tac_app)
        break
      if not tac_app.failed:
        might_close = True
    if might_close and not self.closed:
      # Recursively remove the ignore flag in all descendants
      # that have a chance to close.
      for tac_app in self.successfull_attempts:
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
    for subgoal in tactic_application.subgoals:
      assert subgoal.closed
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

  def update_failed(self):
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
