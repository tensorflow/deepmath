"""This file contains utilities for analyze proofs exported by the prover.

The most important function here is a utility that creates an acyclic subgraph
of the proof graph that explains why a proof is correct.
"""
from __future__ import absolute_import
from __future__ import division
# Import Type Annotations
from __future__ import print_function
import tensorflow as tf
from typing import List, Optional, Tuple, Text
from deepmath.deephol import deephol_pb2
from deepmath.proof_assistant import proof_assistant_pb2


class GoalNotFoundError(Exception):
  """Raised when the goal is not found or not proven."""

  def __init__(self, goal: Text):
    super(GoalNotFoundError, self).__init__()
    self.goal = goal


def _thm_string(thm: proof_assistant_pb2.Theorem) -> Text:
  """Turn theorem into a string for unique representation.

  Args:
    thm: Theorem to be turned into a string.

  Returns:
    string: Joined hypotheses and conclusion.
  """
  return '|:|'.join([str(hyp) for hyp in thm.hypotheses] +
                    [str(thm.conclusion)])


class Node(object):
  """Temporary node object for proof analysis."""

  def __init__(self, node: deephol_pb2.ProofNode, index: int):
    # The corresponding proof node.
    self.node = node
    # The index of the node in the ProofLog.
    self.index = index
    # A map from index to TacticApplication of successful tactic applications.
    self.proofs = {}
    # None or the real explanation for the correctness of the node.
    # This proof should not be part of any circular reasoning.
    self.true_proof = -1
    # Closed: whether this was node was successfully closed in this analysis.
    self.closed = False
    # List of (ProofNode, int) pairs, corresponding to the tactic application
    # this node is supposed to be helpful to prove.
    self.parents = []
    # Processed: marks the proof to be processed for the final output.
    self.processed = False

  # Unfortunately, ptype does not allow to use "Node" in tactic applications
  # until the whole class is fully defined.
  def update_parents(self, goal_to_node, closed_nodes):
    """Create the parents arrays and mark leaf nodes as closed.

    This method must be called exactly once for each node marked as closed in
    the proof tree.
    Whenever a node is closed it closes additional parent nodes. This
    information is propagated to the parent nodes and the tactic application
    is marked as true proof of the parent if that closes in this turn.
    This method collects all the leaf nodes (those nodes with a tactic
    application without subgoal in the closed_nodes list).

    Args:
      goal_to_node: Callable[[proof_assistant_pb2.Theorem], Node] that finds a
        node based on its goal.
      closed_nodes: The list of node that got closed in the order of their
        closing, of type List[Node].
    """
    for i, proof in enumerate(self.node.proofs):
      if proof.result == deephol_pb2.TacticApplication.SUCCESS:
        if not proof.subgoals:
          assert proof.closed, str(proof)
          self.true_proof = i
          self.closed = True
          closed_nodes.append(self)
          self.proofs[i] = []
          return
    for i, proof in enumerate(self.node.proofs):
      if proof.closed:
        proof = [goal_to_node(subgoal) for subgoal in proof.subgoals]
        self.proofs[i] = proof
        for node in proof:
          node.parents.append((self, i))

  def update_closed(self, closed_nodes):
    """Propagate the true reason and closed flags to the parents of a proof.

    Args:
      closed_nodes: The list of nodes that has been closed so far of type
        List[Node].
    """
    assert self.true_proof >= 0
    assert self.closed
    for parent, proof_id in self.parents:
      if parent.closed:
        continue
      proof = parent.proofs[proof_id]
      parent.proofs[proof_id] = [n for n in proof if n.index != self.index]
      if not parent.proofs[proof_id]:
        parent.true_proof = proof_id
        parent.closed = True
        closed_nodes.append(parent)
        parent.update_closed(closed_nodes)


def find_reasons(
    proof_log: deephol_pb2.ProofLog
) -> Optional[Tuple[List[Tuple[int, int, List[int]]], List[int]]]:
  """Find the real reasons why the root node of a proof is is proved.

    This function assumes that the root node is closed, otherwise an
    error message is displayed and None is returned.

  Args:
    proof_log: Proof log to be analyzed.

  Returns:
    A pair of (reasons, sorting), where reasons is a list of (int, list of int),
    representing the acyclic hypergraph that explains why the proof node is
    closed. Each (i, js) in this list represents a TacticApplication for node
    with index i, and js is the list of node indices to which the
    TacticApplication refers to. All nodes are represented by their index in
    the proof_log.nodes list.
    The sorting represents a topological sorting of all the nodes that
    contribute to the above proof. This list starts with the theorem nodes and
    the subgoals always come after the nodes they prove.
  """
  # A map that maps the string representation of proof_assistant_pb2.Theorems to
  # their nodes. It stores only those nodes that are marked to be proved.
  thm_node = {}
  # Node objects corresponding to the root of the proof log. The roots should
  # be marked as THEOREM.
  to_process = []

  def goal_to_node(thm):
    thm_str = _thm_string(thm)
    node = thm_node.get(thm_str, None)
    if node is None:
      raise GoalNotFoundError(thm_str)
    return node

  # Create the mapping that maps theorem representations to Node.
  # Also updates the list of nodes for which the proofs should be reconstructed.
  for i, node in enumerate(proof_log.nodes):
    if node.status == deephol_pb2.ProofNode.PROVED:
      ths = _thm_string(node.goal)
      if ths in thm_node:
        other = thm_node[ths]
        other.node.proofs.extend(node.proofs)
        continue
      n = Node(node, i)
      thm_node[ths] = n
      if node.goal.tag == proof_assistant_pb2.Theorem.THEOREM:
        to_process.append(n)
  if not to_process:
    # We don't have anything to prove, so we just return an empty reasons and
    # an empty nodes list.
    return ([], [])
  closed = []
  # Initialize the parent node information and mark leaf nodes to be proved.
  try:
    for node in sorted(thm_node.values(), key=lambda n: n.index):
      node.update_parents(goal_to_node, closed)
    if not closed:
      tf.logging.error('There are no closed leafs (tactic applications without '
                       'subgoals .')

      return None
  except GoalNotFoundError as xcp:
    tf.logging.error(
        'Could not find subgoal "%s" of closed proof among closed '
        'nodes.', xcp.goal)
    return None
  i = 0
  # We mark the true reason for being closed backwards from the leaf nodes.
  while i < len(closed):
    closed[i].update_closed(closed)
    i += 1
  for n in to_process:
    if not n.closed:
      tf.logging.error('Root %d is marked closed, but it does not check out.',
                       n.index)
      return None
    n.processed = True
  # Collect the reasons for all the nodes in a BFS manner starting from the
  # theorem nodes.
  reasons = []
  i = 0
  while i < len(to_process):
    node = to_process[i]
    i += 1
    if node.true_proof < 0 or not node.closed:
      tf.logging.error('Node %d has no true proof, but it is marked proved.', i)
      return None
    proof = node.node.proofs[node.true_proof]
    try:
      subgoals = [goal_to_node(subgoal) for subgoal in proof.subgoals]
    except GoalNotFoundError as xcp:
      tf.logging.error('Could not find subgoal "%s" among proved nodes',
                       xcp.goal)
      return None
    reasons.append((node.index, node.true_proof,
                    [subgoal.index for subgoal in subgoals]))
    for subgoal in subgoals:
      if not subgoal.processed:
        subgoal.processed = True
        to_process.append(subgoal)
  return reasons, [node.index for node in to_process if node is not None]


def _keep_tac_app(node: deephol_pb2.ProofNode, i: int) -> deephol_pb2.ProofNode:
  """Keep only one tactic application with the given index in the node.

  Args:
    node: The node for which the tactic_applications list is to be reduced.
    i: Index of the tactic application.

  Returns:
    A new node with a single tactic_application.
  """
  tac_app = deephol_pb2.TacticApplication()
  tac_app.CopyFrom(node.proofs[i])
  del node.proofs[:]
  node.proofs.add().CopyFrom(tac_app)
  return node


def extract_proof(proof_log: deephol_pb2.ProofLog
                 ) -> Optional[deephol_pb2.ProofLog]:
  """Reduce the proof into a simply checkable format.

  The utility of this function is to prune the proof to an acyclic
  sub-hypergraph, so that the proof argument is trivial to check.
  All the nodes in this acyclic hypergraph are proved and correspond to
  the proof of the root node of the proof log. The nodes are ordered
  from the higherst level targets, that means each goals precede their
  subgoals in the list of nodes.

  Args:
    proof_log: The proof_log that needs to be reduced.

  Returns:
    A new ProofLog with a mimimal proof necessary to prove all closed
    Theorem nodes.
  """
  if not proof_log.nodes:
    return proof_log
  result = find_reasons(proof_log)
  if result is None:
    return None
  (reasons, _) = result
  new_log = deephol_pb2.ProofLog(
      error_message=proof_log.error_message,
      num_proofs=proof_log.num_proofs,
      prover_options=proof_log.prover_options,
      time_spent=proof_log.time_spent,
      theorem_in_database=proof_log.theorem_in_database)
  new_log.nodes.extend([
      _keep_tac_app(proof_log.nodes[node_index], tac_app_index)
      for node_index, tac_app_index, _ in reasons
  ])
  return new_log
