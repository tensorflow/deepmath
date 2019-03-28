"""Action Geneator API.

From information about theorem prover's state, generate a set of possible
actions to take in the prover.
"""

from __future__ import absolute_import
from __future__ import division
# Import Type Annotations
from __future__ import print_function
import collections
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Optional, Text
from deepmath.deephol import deephol_pb2
from deepmath.deephol import embedding_store
from deepmath.deephol import predictions
from deepmath.deephol import process_sexp
from deepmath.deephol import proof_search_tree
from deepmath.deephol import theorem_fingerprint
from deepmath.proof_assistant import proof_assistant_pb2

Suggestion = collections.namedtuple('Suggestion', ['string', 'score'])


def _theorem_string_for_predictor(thm: proof_assistant_pb2.Theorem) -> Text:
  return process_sexp.process_sexp(str(thm.conclusion))


def _compute_parameter_string(types, no_params_score: float,
                              thm_scores: List[float], thm_strings: List[Text],
                              max_theorem_parameters: int) -> List[Text]:
  """Computes appropriate parameters from a ranked list based on tactic type.

  Args:
    types: Expected type of computed parameters (e.g. thm, list of thms, etc),
      of type List[deephol_pb2.Tactic.ParameterType].
    no_params_score: Score of passing no parameters to the tactic.
    thm_scores: Score assigned to each candidate parameter.
    thm_strings: Theorem name corresponding to the score.
    max_theorem_parameters: Maximum number of parameters in tactic parameters.

  Returns:
    A list of string-represented parameter candidates.
  Raises:
    ValueError: if appropriate parameter candidates cannot be generated.
  """
  if not types:
    return ['']

  thm_ranked = sorted(
      zip(thm_scores, thm_strings), reverse=True)[:max_theorem_parameters]
  if types == [deephol_pb2.Tactic.THEOREM]:
    if not thm_strings:
      # thm_strings is empty for the first theorem in the database.
      raise ValueError('Theorem parameter requested, but none supplied.')
    return [' %s' % thm_ranked[0][1]]

  if types == [deephol_pb2.Tactic.THEOREM_LIST]:
    ret = []
    if not thm_strings:
      ret.append(' [ ]')
      return ret

    # If predictor also suggests passing no arguments to the tactic, then
    # additionally return an empty list as a parameter string.
    if thm_ranked[-1][0] < no_params_score:
      ret.append(' [ ]')

    best_thms = [t for _, t in thm_ranked]
    ret.append(' [ %s ]' % ' ; '.join(best_thms))
    return ret
  raise ValueError('Unsupported tactic parameter types %s' % str(types))


class ActionGenerator(object):
  """Generates candidate actions given the theorem prover's current state."""

  def __init__(
      self,
      theorem_database: proof_assistant_pb2.TheoremDatabase,
      tactics: List[deephol_pb2.Tactic],
      predictor: predictions.Predictions,
      options: deephol_pb2.ActionGeneratorOptions,
      model_architecture: deephol_pb2.ProverOptions.ModelArchitecture,
      emb_store: Optional[embedding_store.TheoremEmbeddingStore] = None):
    self.theorem_database = theorem_database
    self.tactics = tactics
    self.predictor = predictor
    self.options = options
    self.model_architecture = model_architecture
    self.embedding_store = emb_store
    self.thm_names = [
        theorem_fingerprint.ToTacticArgument(thm)
        for thm in theorem_database.theorems
    ]
    self.thm_index_by_fingerprint = {
        theorem_fingerprint.Fingerprint(thm): i
        for (i, thm) in enumerate(theorem_database.theorems)
    }

  def _get_theorem_scores(self, proof_state_enc, thm_number: int,
                          tactic_id: int):
    """Get the scores of all the theorems before the given theorem index.

    This functions scores all preceding theorems in the list of theorems, by
    computing all pairwise scores with the given proof state encoding.

    Args:
       proof_state_enc: A numpy vector of the proof state encoding.
       thm_number: Index of the theorem in the theorem database.
       tactic_id: For tactic dependent prediction, provide tactic id.

    Returns:
       A numpy vector of theorem scores for all preceding theorems in the
       same order they are present in the theorem database.
    """
    if self.embedding_store:
      return self.embedding_store.get_thm_scores_for_preceding_thms(
          proof_state_enc, thm_number, tactic_id)

    relevant_thms = self.theorem_database.theorems[:thm_number]

    if relevant_thms:
      thms_emb = self.predictor.batch_thm_embedding(
          [_theorem_string_for_predictor(thm) for thm in relevant_thms])
    else:
      thms_emb = np.empty([0])
    tf.logging.debug(thms_emb)
    if len(thms_emb):  # pylint: disable=g-explicit-length-test
      thm_scores = self.predictor.batch_thm_scores(proof_state_enc, thms_emb,
                                                   tactic_id)
    else:
      thm_scores = []
    tf.logging.debug(thm_scores)
    return thm_scores

  def step(self, node: proof_search_tree.ProofSearchNode,
           premises: proof_assistant_pb2.PremiseSet) -> List[Suggestion]:
    """Generates a list of possible ApplyTactic argument strings from a goal.

    Args:
      node: state of the proof search, starting at current goal.
      premises: Specification of the selection of premises that can be used for
        tactic parameters. Currently we are supporting only a single
        DatabaseSection.

    Returns:
      List of string arugments for HolLight.ApplyTactic function, along with
      scores (Suggestion).
    """
    assert not premises.reference_sets, ('Premise reference sets are not '
                                         'supported.')
    assert len(premises.sections) == 1, ('Premise set must have exactly one '
                                         'section.')
    # TODO(szegedy): If the premise is not specified, we want the whole
    # database to be used. Not sure if -1 or len(database.theorems) would do
    # that or not. Assertion will certainly fail before that.
    # Also we don't have checks on this use case.
    assert premises.sections[0].HasField('before_premise'), ('Premise is '
                                                             'required.')
    fp = premises.sections[0].before_premise
    thm_number = self.thm_index_by_fingerprint.get(fp)
    assert thm_number is not None
    assert theorem_fingerprint.Fingerprint(
        self.theorem_database.theorems[thm_number]) == fp
    thm_names = self.thm_names[:thm_number]
    tf.logging.debug(thm_names)
    proof_state = predictions.ProofState(goal=str(node.goal.conclusion))
    proof_state_emb = self.predictor.proof_state_embedding(proof_state)
    proof_state_enc = self.predictor.proof_state_encoding(proof_state_emb)
    tf.logging.debug(proof_state_enc)
    tactic_scores = self.predictor.batch_tactic_scores([proof_state_enc])[0]

    empty_emb = self.predictor.thm_embedding('')
    empty_emb_batch = np.reshape(empty_emb, [1, empty_emb.shape[0]])

    enumerated_tactics = enumerate(self.tactics)
    if self.options.asm_meson_only:
      enumerated_tactics = [
          v for v in enumerated_tactics if str(v[1].name) == 'ASM_MESON_TAC'
      ]
      assert enumerated_tactics, (
          'action generator option asm_meson_only requires ASM_MESON_TAC.')

    ret = []
    thm_scores = None
    # TODO(smloos): This computes parameters for all tactics. It should cut off
    # based on the prover BFS options.
    for tactic_id, tactic in enumerated_tactics:
      if (thm_scores is None or self.model_architecture ==
          deephol_pb2.ProverOptions.PARAMETERS_CONDITIONED_ON_TAC):
        thm_scores = self._get_theorem_scores(proof_state_enc, thm_number,
                                              tactic_id)
        tf.logging.debug(thm_scores)
        no_params_score = self.predictor.batch_thm_scores(
            proof_state_enc, empty_emb_batch, tactic_id)[0]
        tf.logging.info('Theorem score for empty theorem: %f0.2',
                        no_params_score)

      tactic_str = str(tactic.name)
      try:
        tactic_params = _compute_parameter_string(
            list(tactic.parameter_types), no_params_score, thm_scores,
            thm_names, self.options.max_theorem_parameters)
        for params_str in tactic_params:
          ret.append(
              Suggestion(
                  string=tactic_str + params_str,
                  score=tactic_scores[tactic_id]))
      except ValueError as e:
        tf.logging.warning('Failed to compute parameters for tactic %s: %s',
                           tactic.name, str(e))
    return ret


class MesonActionGenerator(object):
  """Trivial action generator, which always returns MESON tactic."""

  def step(self, goal: proof_assistant_pb2.Theorem,
           thm: proof_assistant_pb2.Theorem) -> List[Tuple[Text, float]]:
    del goal  # unused
    del thm  # unused
    return [('ASM_MESON_TAC [ ]', 1.0)]
