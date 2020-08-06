"""Proof pruning library.

The purpose of this library is to optimize proofs. Currently we
minimize the number of tactic application parameters in oder to generate
better training data (with minimum number of tactic parameters).
"""
import time
from typing import Dict, Iterable, List, Optional, Text
import tensorflow.compat.v1 as tf
from deepmath.deephol import deephol_pb2
from deepmath.deephol import tactic_utils
from deepmath.deephol import theorem_fingerprint
from deepmath.deephol import theorem_utils
from deepmath.public import proof_assistant
from deepmath.deephol.utilities import proof_analysis
from deepmath.proof_assistant import proof_assistant_pb2
from deepmath.public import error

MIN_HARD_NEGATIVES = 5
MAX_HARD_NEGATIVES = 10

MIN_TIMEOUT_MS = 50
MAX_TIMEOUT_MS = 5000
ALLOWED_TIME_VARIATION = 0.5


class InvalidProofError(Exception):
  pass


def _node_idx_map(nodes: Iterable[deephol_pb2.ProofNode]):
  return {
      theorem_fingerprint.Fingerprint(node.goal): idx
      for idx, node in enumerate(nodes)
  }


def _params_num(tapp: deephol_pb2.TacticApplication) -> int:
  return sum([len(p.theorems) for p in tapp.parameters])


def _extract_tactic_applications(
    node: deephol_pb2.ProofNode,
    proof_log: deephol_pb2.ProofLog,
    node_idx_map: Optional[Dict[int, int]] = None
) -> List[deephol_pb2.TacticApplication]:
  """Extract the proof for the given ProofNode as a list of tactic applications.

  Args:
    node: The proof node for which to extract a proof.
    proof_log: The proof log containing 'node' and a proof for it.
    node_idx_map: Optional argument containing a map from node fingerprints to
      node indices in the proof log. Can be given to avoid recomputation.

  Returns:
    List of deep copies of the tactic applications that prove the given node.

  Raises:
    InvalidProofError: InvalidProofError if no valid proof could be found.
  """
  if node.status != deephol_pb2.ProofNode.PROVED:
    raise InvalidProofError('Proof not closed')
  if node_idx_map is None:
    node_idx_map = _node_idx_map(proof_log.nodes)
  tactic_applications = []
  node_stack = [node]  # used as stack; mind the inverted order
  while node_stack:
    if not node_stack:
      raise InvalidProofError('Incomplete proof.')
    node = node_stack.pop()
    first_closed_tapp = None
    for tactic_application in node.proofs:
      if tactic_application.closed:
        first_closed_tapp = tactic_application
    if not first_closed_tapp:
      # Should not be possible since the given node is marked to be proven.
      raise InvalidProofError('No closed tactic application found for node.')
    tapp = deephol_pb2.TacticApplication()
    tapp.CopyFrom(first_closed_tapp)
    tactic_applications.append(tapp)
    subgoal_nodes = [
        proof_log.nodes[node_idx_map[theorem_fingerprint.Fingerprint(goal)]]
        for goal in tapp.subgoals
    ]
    node_stack.extend(subgoal_nodes[::-1])  # inverted order because stack
  return tactic_applications


def _proof_nodes_from_proof(
    root_goal: proof_assistant_pb2.Theorem,
    proof: List[deephol_pb2.TacticApplication]) -> List[deephol_pb2.ProofNode]:
  """Extract the list of ProofNodes that are part of the given proof.

  Args:
    root_goal: The goal to be proven.
    proof: The proof given as a list of tactic applications.

  Returns:
    New proof nodes for the root_goal and the subgoals of tactic applications in
    the proof.

  Raises:
    InvalidProofError: If the proof has leftover steps or is incomplete.
  """
  extracted_proof_nodes = []
  goal_stack = [root_goal]
  for tactic_application in proof:
    if not goal_stack:
      raise InvalidProofError('Leftover proof steps.')
    goal = goal_stack.pop()
    node = deephol_pb2.ProofNode()
    node.goal.CopyFrom(goal)
    node.goal.tag = proof_assistant_pb2.Theorem.GOAL
    node.status = deephol_pb2.ProofNode.PROVED
    node.proofs.add().CopyFrom(tactic_application)
    extracted_proof_nodes.extend([node])
    goal_stack.extend(tactic_application.subgoals[::-1])
  extracted_proof_nodes[0].root_goal = True
  if goal_stack:
    raise InvalidProofError('Incomplete proof. Goal stack not empty.')
  return extracted_proof_nodes


def _create_request(
    goal: proof_assistant_pb2.Theorem, tactic: Text,
    params: Iterable[deephol_pb2.TacticParameter], best_time_spent_ms: int,
    assumption_indices: Dict[int,
                             int]) -> proof_assistant_pb2.ApplyTacticRequest:
  """Creates ApplyTacticRequest for the given goal and parameters.

  Automatically sets the timeout to avoid unnecessary waiting for requests
  which take longer than the fastest previous successful request which had extra
  arguments. Allows 50% variation and at least 50 ms total timeout to account
  for random fluctuations.

  Args:
    goal: Goal of the ProofNode to which the tactic application belongs.
    tactic: The name of the tactic to apply.
    params: The arguments of the tactic.
    best_time_spent_ms: The duration in milliseconds of the fastest previous
      successful request which had extra arguments.
    assumption_indices: Map from fingerprints to assumption indices. Overrides
      theorem.assumption_index for theorem parameters.

  Returns:
    The created ApplyTacticRequest.
  """
  try:
    tactic_app_str = tactic_utils.tactic_string(
        tactic, params, asm_indices=assumption_indices)
  except ValueError as e:
    tf.logging.warning('Could not generate tactic string: %s', e)
    # This should make the tactic request always fail
    tactic_app_str = ''
  timeout_ms = int(best_time_spent_ms * (1 + ALLOWED_TIME_VARIATION) + 0.5)
  timeout_ms = max(MIN_TIMEOUT_MS, timeout_ms)
  timeout_ms = min(MAX_TIMEOUT_MS, timeout_ms)
  return proof_assistant_pb2.ApplyTacticRequest(
      goal=theorem_utils.theorem_to_goal(goal),
      tactic=tactic_app_str,
      timeout_ms=timeout_ms)


def _theorem_protos_equal(thm1: proof_assistant_pb2.Theorem,
                          thm2: proof_assistant_pb2.Theorem):
  """Tests if the theorem protos are semantically equivalent. Ignores tags."""
  result = thm1.conclusion == thm2.conclusion
  result = result and len(thm1.assumptions) == len(thm2.assumptions)
  result = result and thm1.hypotheses == thm2.hypotheses
  for a1, a2 in zip(thm1.assumptions, thm2.assumptions):
    # Recursion can only go 1 level deep as goals can have theorems as
    # assumptions, but theorems cannot have assumptions themselves.
    result = result and _theorem_protos_equal(a1, a2)
  return result


class ParameterPruning(object):
  """Class to do parameter pruning on proof nodes."""

  def __init__(self,
               theorem_db: proof_assistant_pb2.TheoremDatabase,
               hol_wrapper=None):
    if hol_wrapper and theorem_db:
      tf.logging.warning(
          'theorem_db provided will be ignored as hol_wrapper provided.')
    self.hol_wrapper = hol_wrapper
    if not self.hol_wrapper:
      self.hol_wrapper = proof_assistant.ProofAssistant()
      for theorem in theorem_db.theorems:
        self.hol_wrapper.RegisterTheorem(
            proof_assistant_pb2.RegisterTheoremRequest(theorem=theorem))
    self.communication_failed = False

  def _accept_response(
      self, tapp: deephol_pb2.TacticApplication,
      response: proof_assistant_pb2.ApplyTacticResponse,
      proof_tail: Optional[List[deephol_pb2.TacticApplication]]) -> bool:
    """Decides if the response matches expectations.

    Args:
      tapp: The tactic application containing the expected subgoals.
      response: The response from HOL Light after pruning the parameters.
      proof_tail: Rest of the proof (optional).

    Returns:
      Decides if the response yields the expected subgoals. If a proof tail is
      given, this method also returns true, if the subgoals do not match but the
      proof tail still satisfies the subgoals. In this case this method updates
      the subgoals.
    """
    if response.HasField('error'):
      return False
    assert response.HasField('goals'), 'response: %s' % response
    new_subgoals = list(response.goals.goals)
    for goal in new_subgoals:
      theorem_utils.convert_legacy_goal(goal)
    if len(new_subgoals) != len(tapp.subgoals):
      if len(new_subgoals) < len(tapp.subgoals):
        tf.logging.warning('Pruning: fewer subgoals than expected.')
      return False  # rerunning the proof_tail won't help here.
    subgoals_match = True
    for idx, subgoal in enumerate(new_subgoals):
      if not _theorem_protos_equal(subgoal, tapp.subgoals[idx]):
        subgoals_match = False
        break
    if subgoals_match:
      return True
    elif proof_tail:
      try:
        updated_proof_tail = self._rerun_proof(new_subgoals, proof_tail)
        del tapp.subgoals[:]
        tapp.subgoals.extend(new_subgoals)
        tapp.strong_pruning_successful_num += 1
        del proof_tail[:]
        proof_tail.extend(updated_proof_tail)
        return True
      except InvalidProofError:
        return False
    else:  # subgoals do not match and no proof tail is given
      return False

  def prune_tactic_application(
      self,
      goal: proof_assistant_pb2.Theorem,
      tapp: deephol_pb2.TacticApplication,
      proof_tail: Optional[List[deephol_pb2.TacticApplication]] = None):
    """Parameter pruning for a single tactic application.

    Args:
      goal: Goal of the ProofNode to which the tactic application belongs.
      tapp: The tactic application to be pruned.
      proof_tail: Optional list of tactic applications to check if proof still
        holds when subgoals do not match.
    """
    if self.communication_failed:
      tf.logging.error('Communication with prover failed. Not pruning...')
      return
    parameters = tapp.parameters
    if not parameters:
      return
    assert len(parameters) == 1
    param = parameters[0]
    if param.parameter_type != deephol_pb2.Tactic.THEOREM_LIST:
      return
    thms = list(param.theorems)
    if not thms:
      return
    pruning_start_time = time.time()
    index = len(thms) - 1
    tactic = tapp.tactic
    time_spent = tapp.time_spent
    false_positives = []
    other_negatives = []
    found_true_positive = False
    assumption_indices = tactic_utils.assumption_indices(goal)
    while index >= 0:
      thm = thms.pop(index)
      thm_list_to_test = deephol_pb2.TacticParameter(
          parameter_type=deephol_pb2.Tactic.THEOREM_LIST, theorems=thms)
      request = _create_request(
          goal,
          tactic, [thm_list_to_test],
          time_spent,
          assumption_indices=assumption_indices)
      start_time = time.time()
      response = proof_assistant_pb2.ApplyTacticResponse()
      try:
        response = self.hol_wrapper.ApplyTactic(request)
        elapsed_msecs = int((time.time() - start_time) * 1000.0 + 0.5)
      except error.StatusNotOk as exception:
        tf.logging.error(exception)
        elapsed_msecs = int((time.time() - start_time) * 1000.0 + 0.5)
        if exception.message.startswith(
            'Communication') and exception.message.endswith('failed.'):
          tf.logging.error('Communication with prover failed. Not pruning...')
          self.communication_failed = True
          return
      match = self._accept_response(tapp, response, proof_tail)
      if not match:
        thms.insert(index, thm)
        found_true_positive = True
      else:  # have found a premise that we can remove from the premise list
        if found_true_positive:
          false_positives.append(thm)
        else:
          other_negatives.append(thm)
        time_spent = elapsed_msecs
      index -= 1
    del tapp.parameters[0].theorems[:]
    tapp.parameters[0].theorems.extend(thms)
    tapp.parameters[0].hard_negative_theorems.extend(
        false_positives[:MAX_HARD_NEGATIVES])
    if len(false_positives) < MIN_HARD_NEGATIVES:
      other_negatives.reverse()
      tapp.parameters[0].hard_negative_theorems.extend(
          other_negatives[:(MIN_HARD_NEGATIVES - len(false_positives))])
    tapp.time_spent = time_spent
    tapp.pruning_time_spent_ms += int((time.time() - pruning_start_time) *
                                      1000.0 + 0.5)

  def prune_tactic_applications(self, proof_node: deephol_pb2.ProofNode):
    for proof in proof_node.proofs:
      if proof.result == deephol_pb2.TacticApplication.SUCCESS:
        self.prune_tactic_application(proof_node.goal, proof)

  def prune_closed_tactic_applications(self, proof_log: deephol_pb2.ProofLog):
    """Prune all closed tactic applications individually."""
    tf.logging.info('Pruning closed proof nodes...')
    start_time = time.time()
    for node in proof_log.nodes:
      if node.status == deephol_pb2.ProofNode.PROVED:
        for proof in node.proofs:
          if proof.closed:
            assert proof.result == deephol_pb2.TacticApplication.SUCCESS
            self.prune_tactic_application(node.goal, proof)
    total_time = time.time() - start_time
    proof_log.pruning_time_ms += int(total_time * 1000)

  def prune_proof_log(self, proof_log: deephol_pb2.ProofLog):
    """Prunes unnecessary premises and proof steps; O(n^2)."""
    # This function differs from proof_analysis.extract_proof in that it calls
    # the proof assistant and may change the proof.
    start_time = time.time()
    try:
      extracted_proof = proof_analysis.extract_proof(proof_log)
      if not extracted_proof:
        tf.logging.info('Pruning failed because no proof could be extracted.')
        return
      error_msg = proof_analysis.check_extracted_proof_assumptions(
          extracted_proof)
      if error_msg:
        raise ValueError('Assumptions on proof extraction violated: %s' %
                         error_msg)

      # Traverse the proof log in reverse order and check each step if it is
      # necessary.
      # ATTENTION: O(n^2)
      try:
        node_idx_map = _node_idx_map(extracted_proof.nodes)
        root = extracted_proof.nodes[0]
        proof = _extract_tactic_applications(root, extracted_proof,
                                             node_idx_map)
        # we must apply exactly one tactic application to each goal.
        if len(extracted_proof.nodes) != len(proof):
          tf.logging.error('Length of proof does not match number of proof '
                           'nodes. This can only happen if the same subgoal is '
                           'accidentally visited more than once.')
        updated_proof = []
        for node, tactic_application in list(zip(extracted_proof.nodes,
                                                 proof))[::-1]:
          proof_for_node = [tactic_application] + updated_proof
          updated_proof = self._prune_first_proof_step(node.goal,
                                                       proof_for_node)
        try:
          updated_proof = self._rerun_proof([root.goal], updated_proof)
        except InvalidProofError as e:
          tf.logging.error('Failed to rerun proof after pruning: %s.', e)
          return
        extracted_proof = _proof_nodes_from_proof(root.goal, updated_proof)
        proof_log.extracted_proof.extend(extracted_proof)
        proof_log.pruned_steps_num = len(proof) - len(updated_proof)
        assert proof_log.pruned_steps_num >= 0
      except InvalidProofError as e:
        tf.logging.error('Proof-level pruning unsuccessful: %s.', e)
        return
    finally:
      total_time = time.time() - start_time
      proof_log.pruning_time_ms += int(total_time * 1000)

  def _rerun_proof(
      self, goals: List[proof_assistant_pb2.Theorem],
      proof: List[deephol_pb2.TacticApplication]
  ) -> List[deephol_pb2.TacticApplication]:
    """Check if node can be proven in the context of the given proof log.

    Args:
      goals: The goals to be proven.
      proof: List of proof steps to apply to close the proof.

    Returns:
      Returns updated proof with new subgoals and potantially fewer tactic
      applications; or raises an exception if the proof does not check.

    Raises:
      InvalidProofError: if the proof does not check.
    """
    goal_stack = goals[::-1]  # is a stack, keep the inverted order in mind
    updated_proof = []
    for idx, tapp in enumerate(proof):
      if not goal_stack:
        if idx < len(proof) - 1:
          tf.logging.warning(
              'Proof finished with remaining tactic applications. Dropping.')
        break
      next_goal = goal_stack.pop()
      assumption_indices = tactic_utils.assumption_indices(next_goal)
      request = _create_request(
          next_goal,
          tapp.tactic,
          tapp.parameters,
          tapp.time_spent,
          assumption_indices=assumption_indices)
      response = proof_assistant_pb2.ApplyTacticResponse()
      start_time = time.time()
      try:
        response = self.hol_wrapper.ApplyTactic(request)
      except error.StatusNotOk as exception:
        tf.logging.error(exception)
        if exception.message.startswith(
            'Communication') and exception.message.endswith('failed.'):
          tf.logging.error('Communication with prover failed. Not pruning...')
          self.communication_failed = True
          raise InvalidProofError(
              'Communication error during tactic application')
        raise InvalidProofError('Exception occurred in tactic application')
      if response.HasField('error'):
        raise InvalidProofError('Proof step %d failed: %s' %
                                (idx, response.error))
      assert response.HasField('goals'), 'response: %s' % response
      new_subgoals = list(response.goals.goals)
      for subgoal in new_subgoals:
        theorem_utils.convert_legacy_goal(subgoal)
      goal_stack.extend(new_subgoals[::-1])  # stack has inverted order
      new_tapp = deephol_pb2.TacticApplication()
      new_tapp.CopyFrom(tapp)
      updated_proof.append(new_tapp)
      del new_tapp.subgoals[:]
      new_tapp.subgoals.extend(new_subgoals)
      new_tapp.time_spent = int((time.time() - start_time) * 1000.0)
    if goal_stack:
      raise InvalidProofError('Open goals.')
    return updated_proof

  def _prune_first_proof_step(
      self, goal: proof_assistant_pb2.Theorem,
      proof: List[deephol_pb2.TacticApplication]
  ) -> List[deephol_pb2.TacticApplication]:
    """Tries to omit the first proof step or defaults to parameter pruning.

    Args:
      goal: The goal to which this proof applies.
      proof: Proof of the goal.

    Returns:
      The updated proof. May be one step shorter than the old proof.
    """
    assert proof
    proof_tail = proof[1:]
    try:
      # Try skipping the first proof step
      updated_proof_tail = self._rerun_proof([goal], proof_tail)
      tf.logging.info('Pruning: successfully dropped tactic application.')
      return updated_proof_tail
    except InvalidProofError:
      tf.logging.info('Pruning: Could not prune entire tactic application.')
      # Try pruning parameters of type THEOREM_LIST
      self.prune_tactic_application(goal, proof[0], proof_tail)
      return proof
