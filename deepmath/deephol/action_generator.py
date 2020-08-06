"""Action Geneator API.

From information about theorem prover's state, generate a set of possible
actions to take in the prover.
"""
import collections
import time
from typing import List, Tuple, Optional, Text, Dict
import numpy as np
from scipy import spatial
import tensorflow.compat.v1 as tf

from deepmath.deephol import abstract_action_generator
from deepmath.deephol import deephol_pb2
from deepmath.deephol import embedding_store
from deepmath.deephol import predictions
from deepmath.deephol import process_sexp
from deepmath.deephol import proof_search_tree
from deepmath.deephol import theorem_fingerprint
from deepmath.deephol import to_sexpression
from deepmath.deephol.utilities import normalization_lib
from deepmath.proof_assistant import proof_assistant_pb2

Suggestion = abstract_action_generator.Suggestion
ScoredTheorems = List[Tuple[float, proof_assistant_pb2.Theorem]]

EPSILON = 1e-12
MAX_CLOSEST = 100
MIN_SCORED_PARAMETERS = 1

MAX_HARD_NEGATIVES = 5

ENCODER_CONFIG_NAME = 'encoder_config.json'
DECODER_CONFIG_NAME = 'decoder_config.json'


def logits_to_probabilities(scores, temperature: float = 1.0):
  scores = np.array(scores) / temperature
  weights = 1. / (1. + np.exp(-scores))
  return weights / sum(weights)


def probabilities_to_logits(probabilities):
  p = np.array(probabilities)
  for idx, value in enumerate(p):
    if value == 1.0:
      tf.logging.warning('Lowering probability 1.0 to .999 to avoid divition '
                         'by zero.')
      p[idx] = .999
  return np.log(p / (1 - p))


class SimilarityScorer(object):
  """SimilarityScorer."""

  def __init__(self, theorem_database: proof_assistant_pb2.TheoremDatabase,
               truncate_sexp: Optional[int],
               options: deephol_pb2.ActionGeneratorOptions):
    # We assume theorem database is normalized (so can use tokens directly).
    self.theorem_database = theorem_database
    self.num_words = 0
    self.word_to_index = {}
    self.vocab = []
    self.doc_freq = []
    self.truncate_sexp = truncate_sexp
    self.noise_scale = options.tfidf_noise_scale

    # build vocab, freq
    num_documents = 0
    for theorem in self.theorem_database.theorems:
      if theorem.training_split != proof_assistant_pb2.Theorem.TRAINING:
        continue
      num_documents += 1
      sexp = to_sexpression.convert_theorem(theorem, conclusion_only=False)
      words = set(self._process_sexp(sexp).split())
      for word in words:
        if word in self.word_to_index:
          index = self.word_to_index[word]
        else:
          index = self.num_words
          self.num_words += 1
          self.word_to_index[word] = index
          self.vocab.append(word)
          self.doc_freq.append(0)
        self.doc_freq[index] += 1
    self.doc_freq = np.array(self.doc_freq)
    self.idf = np.log(num_documents / self.doc_freq)
    tf.logging.info('Vocab size: %d', self.num_words)

    num_theorems = len(self.theorem_database.theorems)
    self.tf = np.zeros((num_theorems, self.num_words))
    for i, theorem in enumerate(self.theorem_database.theorems):
      words = collections.Counter(
          self._process_sexp(
              to_sexpression.convert_theorem(theorem,
                                             conclusion_only=False)).split())
      for word, freq in words.items():
        if word in self.word_to_index:
          j = self.word_to_index[word]
          if options.tfidf_variant == 0:
            self.tf[i, j] = freq
          elif options.tfidf_variant == 1:
            self.tf[i, j] = 1 + np.log(freq)
          elif options.tfidf_variant == 2:
            self.tf[i, j] = 1
          else:
            raise ValueError(f'Invalid tfidf_variant {options.tfidf_variant}')
    self.reset_word_weights()

  def _process_sexp(self, sexp: str) -> Text:
    return process_sexp.process_sexp(sexp, self.truncate_sexp)

  def reset_word_weights(self):
    """Reset word weights, and recompute premise_vectors."""
    tf.logging.info('Resetting word weights')
    # Dropout.
    noise = np.random.rand(self.num_words) >= self.noise_scale
    self.word_weights = np.multiply(self.idf, noise)
    self.premise_vectors = self.tf * self.word_weights

  def vectorize(self, sexp: str):
    v = np.zeros(self.num_words)
    words = collections.Counter(self._process_sexp(sexp).split())
    for word, freq in words.items():
      if word in self.word_to_index:
        index = self.word_to_index[word]
        v[index] = freq
    return v * self.word_weights


def _theorem_parameter(theorem, negatives=None) -> deephol_pb2.TacticParameter:
  return deephol_pb2.TacticParameter(
      parameter_type=deephol_pb2.Tactic.THEOREM,
      theorems=[theorem],
      hard_negative_theorems=negatives)


def _theorem_list_parameter(theorems,
                            negatives=None) -> deephol_pb2.TacticParameter:
  return deephol_pb2.TacticParameter(
      parameter_type=deephol_pb2.Tactic.THEOREM_LIST,
      theorems=theorems,
      hard_negative_theorems=negatives)


def _select_parameters(
    types,
    pass_no_arguments: bool,
    scored_premises: Optional[ScoredTheorems],
    num_samples: Optional[int] = None
) -> List[List[deephol_pb2.TacticParameter]]:
  """Computes appropriate parameters from a ranked list based on tactic type.

  Args:
    types: Expected type of computed parameters (e.g. thm, list of thms, etc),
      of type List[deephol_pb2.Tactic.ParameterType].
    pass_no_arguments: Pass no parameters to the tactic.
    scored_premises: ranked theorem parameters.
    num_samples: Number of parameter lists to sample.

  Returns:
    A list of parameter lists.
  Raises:
    ValueError: if appropriate parameter candidates cannot be generated.
  """
  if num_samples is not None and num_samples <= 0:
    raise ValueError('Illegal value for num_samples: %s' % str(num_samples))
  if not types:
    return [[]]
  for t in types:
    if t not in [deephol_pb2.Tactic.THEOREM, deephol_pb2.Tactic.THEOREM_LIST]:
      raise ValueError('Unsupported parameter type: %s' % str(t))
  if not scored_premises:
    raise ValueError('Theorem parameters are required.')

  best_thms = [thm for _, thm in scored_premises]

  if types == [deephol_pb2.Tactic.THEOREM]:
    if not best_thms:
      tf.logging.warning('Could not generate parameter of type THEOREM. '
                         'No premises available.')
      return []
    if num_samples is None:
      selection = [[_theorem_parameter(best_thms[0])]]
    else:
      size = min(len(best_thms), num_samples)
      selected_theorem_indices = np.random.choice(
          list(range(len(best_thms))), size=size, replace=False)
      selection = []
      for idx in selected_theorem_indices:
        thm = best_thms[idx]
        # We add the immediately preceeding theorems
        negatives = list(reversed(best_thms[:idx]))[:MAX_HARD_NEGATIVES]
        selection.append([_theorem_parameter(thm, negatives)])
    return selection

  elif types == [deephol_pb2.Tactic.THEOREM_LIST]:
    parameter_lists = []

    # If predictor also suggests passing no arguments to the tactic, then
    # additionally return an empty list as a parameter string.
    if pass_no_arguments:
      # add empty premise list
      parameter_lists.append([_theorem_list_parameter([])])

    if num_samples is None:
      parameter_lists.append([_theorem_list_parameter(best_thms)])
    else:
      num_samples = min(len(best_thms), num_samples)
      for _ in range(num_samples):
        size = (len(best_thms) // 2) + 1
        # We need the potential_negatives list, because unselected theorems only
        # become negatives if there is a positive with lower score.
        positives, negatives, potential_negatives = [], [], []
        for thm in best_thms:
          if np.random.choice([True, False]):  # uniformly drawn, for now
            positives.append(thm)
            negatives.extend(potential_negatives)
          else:
            potential_negatives.append(thm)
        # We only keep the negatives immediately preceeding the lowest scoring
        # positive.
        negatives = list(reversed(negatives))[:MAX_HARD_NEGATIVES]
        parameter_lists.append([_theorem_list_parameter(positives, negatives)])
    return parameter_lists
  else:
    raise ValueError('Unsupported tactic parameter types %s' % str(types))


class ActionGenerator(abstract_action_generator.AbstractActionGenerator):
  """Generates candidate actions given the theorem prover's current state."""

  def __init__(
      self,
      theorem_database: proof_assistant_pb2.TheoremDatabase,
      tactics: List[deephol_pb2.Tactic],
      predictor: predictions.Predictions,
      options: deephol_pb2.ActionGeneratorOptions,
      model_architecture: deephol_pb2.ProverOptions.ModelArchitecture,
      emb_store: Optional[embedding_store.TheoremEmbeddingStore] = None):
    super(ActionGenerator, self).__init__(theorem_database, tactics)
    self.predictor = predictor
    self.options = options
    self.model_architecture = model_architecture
    self.embedding_store = emb_store
    self.cached_assumption_embeddings = dict()
    if model_architecture == deephol_pb2.ProverOptions.GNN_GOAL:
      truncate_sexp = None
    else:
      truncate_sexp = 1000
    self.ranked_closest = None
    if self.options.bag_of_words_similar:
      self.similarity_scorer = SimilarityScorer(self.theorem_database,
                                                truncate_sexp, self.options)

  def _get_theorem_scores(self, proof_state_enc, thm_number: int,
                          tactic_id: int) -> ScoredTheorems:
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
      thm_scores = self.embedding_store.get_thm_scores_for_preceding_thms(
          proof_state_enc, thm_number, tactic_id)
      return list(zip(thm_scores, self.theorem_database.theorems))

    relevant_thms = self.theorem_database.theorems[:thm_number]

    if relevant_thms:
      thms_emb = self.predictor.batch_thm_proto_embedding(
          [normalization_lib.normalize(thm) for thm in relevant_thms])
    else:
      thms_emb = np.empty([0])
    tf.logging.debug(thms_emb)
    if len(thms_emb):  # pylint: disable=g-explicit-length-test
      thm_scores = self.predictor.batch_thm_scores(proof_state_enc, thms_emb,
                                                   tactic_id)
    else:
      thm_scores = []
    return list(zip(thm_scores, self.theorem_database.theorems))

  def _compute_assumption_embedding(self, assum: proof_assistant_pb2.Theorem):
    """Computes embeddings of assumptions; cached for performance."""
    if not assum.HasField('fingerprint'):
      assum.fingerprint = theorem_fingerprint.Fingerprint(assum)
    if assum.fingerprint in self.thm_index_by_fingerprint:
      index_in_thm_database = self.thm_index_by_fingerprint[assum.fingerprint]
      return self.embedding_store.thm_embeddings[index_in_thm_database]

    if len(self.cached_assumption_embeddings) > 10**5:
      tf.logging.warning('Assumptions embeddings grown cache too big (>100K).')
      self.cached_assumption_embeddings = dict()
    if assum.fingerprint not in self.cached_assumption_embeddings:
      self.cached_assumption_embeddings[
          assum.fingerprint] = self.predictor.thm_proto_embedding(
              normalization_lib.normalize(assum))
    return self.cached_assumption_embeddings[assum.fingerprint]

  def _compute_assumption_scores(self, node: proof_search_tree.ProofSearchNode,
                                 proof_state_enc,
                                 normalized_goal: proof_assistant_pb2.Theorem,
                                 tactic_id: int) -> ScoredTheorems:
    """Compute assumption scores for a (normalized) goal."""
    if normalized_goal.hypotheses:
      raise ValueError('Goals canot have hypotheses:\n %s' % normalized_goal)
    if not normalized_goal.assumptions:
      return []
    assumptions_embeddings = []
    for assum in normalized_goal.assumptions:
      assumptions_embeddings.append(self._compute_assumption_embedding(assum))
    batched_assumptions_emb = np.stack(assumptions_embeddings)
    assumption_scores = self.predictor.batch_thm_scores(
        proof_state_enc, batched_assumptions_emb, tactic_id)
    return list(zip(assumption_scores, node.goal.assumptions))

  def _compute_tactic_scores(self, proof_state_encoded):
    """Compute tactic scores, apply temperature, and add noise."""
    if (self.options.random_tactic_probability > 0.0 and
        self.options.random_tactic_probability > np.random.random()):
      return np.random.random([len(self.tactics)])

    tactic_scores = np.array(
        self.predictor.batch_tactic_scores([proof_state_encoded])[0])

    tf.logging.info('Action generator tactic scores: %s', tactic_scores)
    if self.options.HasField('tactic_temperature'):
      tactic_scores *= 1.0 / self.options.tactic_temperature

    if self.options.dirichlet_noise_for_tactics > 0.0:
      # auto-tuning dirichlet noise to parameters similar to AlphaGo
      dirichlet_alpha = 10.0 / len(tactic_scores)
      tf.logging.info('Adding %f dirichlet noise to tactics with alpha %f.',
                      self.options.dirichlet_noise_for_tactics, dirichlet_alpha)
      noise = np.random.dirichlet(np.ones(len(tactic_scores)) * dirichlet_alpha)
      eps = self.options.dirichlet_noise_for_tactics
      tactic_probabilities = logits_to_probabilities(tactic_scores)
      new_probs = (1 - eps) * tactic_probabilities + eps * noise
      tactic_scores = probabilities_to_logits(new_probs)
    return tactic_scores

  def _compute_closest(self, node: proof_search_tree.ProofSearchNode,
                       thm_number: int):
    """Computes similarity heuristic and sets the field rank_closest if None."""
    if not (self.options.HasField('num_similar_parameters') and
            self.options.num_similar_parameters.max_value > 0):
      return
    rank_closest_start_time = time.time()
    if self.options.bag_of_words_similar:
      closest = self._compute_bag_of_words_closest(node, thm_number)
    else:
      closest = self._compute_network_based_closest(node.goal, thm_number)
    self.ranked_closest = closest
    rank_closest_time = time.time() - rank_closest_start_time
    if node.heuristic_ranking_time_ms is None:
      node.heuristic_ranking_time_ms = 0
    node.heuristic_ranking_time_ms += int(round(rank_closest_time * 1000.0))
    tf.logging.info(
        'Cosine closest picked:\n%s',
        '\n'.join(['%s: %.6f' % (name, score) for score, name in closest]))

  def _compute_bag_of_words_closest(self,
                                    node: proof_search_tree.ProofSearchNode,
                                    thm_number: int) -> ScoredTheorems:
    """Finds nearest theorems using a randomized bag of words embedding.

    This is an extended version of the "BoW2" algorithm from section 2.2 of
    "Learning to Reason in Large Theories without Imitation"
    (https://arxiv.org/pdf/1905.10501.pdf).  It has been extended with an
    additional parameter to let it consider only a subset of the associated
    theorem database.

    Both the goal and considered premises are embedded into their own noisy
    bag-of-words vector wherein each element corresponds to a word in the
    theorem_database corpus.  The value of each element e is, based on its
    corresponding word w, is n_e*tf_w/df_w, where:
      * n_e is drawn from abs(normal(1, word_weights_noise_scale))
      * tf_w is the term frequency of w within the associated premise or goal
      * df_w is the document frequency of the word within the entire theorem
        database (not just the thm_number theorems considered).  In the linked
        paper's section 2.2, document frequency is represented as 'f_i'.


    Note that each n_e is drawn exactly once each time this method is called.

    Args:
      node: A node in the proof search tree to find similar premises to.
      thm_number: The number of premises to evaluate for nearness.  This method
        limits premises under consideration by using the ordering within the
        theorem database.

    Returns:
      The nearest (up to) self.options.max_theorem_parameters premises that come
      before the passed thm_number.  Each premise consist of a score and a
      theorem from the database.  The score is the computed distance
      from the passed goal.
    """
    if self.options.reset_similarity_word_weights_each_round:
      self.similarity_scorer.reset_word_weights()
    proof_state = proof_search_tree.proof_state_from_proof_search_node(node)
    goal_vector = self.similarity_scorer.vectorize(
        to_sexpression.convert_proof_state(
            proof_state, history_bound=0, conclusion_only=False))
    distances = spatial.distance.cdist(
        self.similarity_scorer.premise_vectors[:thm_number],
        goal_vector.reshape(1, -1), 'cosine').reshape(-1).tolist()
    ranked_closest = sorted(
        zip(distances, self.theorem_database.theorems), key=lambda x: x[0])
    return ranked_closest[:self.options.max_theorem_parameters]

  def _compute_network_based_closest(self, goal, thm_number) -> ScoredTheorems:
    """Compute closest based on premise embeddings."""
    # TODO(kbk): Add unit tests for this section (similar_parameters).
    goal_as_thm = proof_assistant_pb2.Theorem()
    goal_as_thm.CopyFrom(goal)
    goal_as_thm.tag = proof_assistant_pb2.Theorem.THEOREM
    goal_embedding_as_thm = self.predictor.thm_proto_embedding(
        normalization_lib.normalize(goal_as_thm))
    premise_embeddings = (
        self.embedding_store.get_embeddings_for_preceding_thms(thm_number))
    # distance_score each is in [0,2]
    distances = spatial.distance.cdist(premise_embeddings,
                                       goal_embedding_as_thm.reshape(1, -1),
                                       'cosine').reshape(-1).tolist()
    ranked_closest = sorted(
        zip(distances, self.theorem_database.theorems), key=lambda x: x[0])
    ranked_closest = ranked_closest[:MAX_CLOSEST]
    tf.logging.info(
        'Cosine closest in premise embedding space:\n%s', '\n'.join([
            '%d: %.6f' % (thm.fingerprint, score)
            for score, thm in ranked_closest
        ]))
    # add some noise to top few and rerank
    noise = np.random.normal(scale=0.2, size=MAX_CLOSEST)
    ranked_closest = [
        (score + noise[i], thm) for i, (score, thm) in enumerate(ranked_closest)
    ]
    ranked_closest = sorted(ranked_closest, key=lambda x: x[0])
    return ranked_closest[:self.options.max_theorem_parameters]

  def _similar_theorems(self, node: proof_search_tree.ProofSearchNode,
                        thm_number: int) -> ScoredTheorems:
    """Return a random number of similar theorems from ranked_closest."""
    self._compute_closest(node, thm_number)
    if self.ranked_closest is None:
      return []
    num_similar = np.random.randint(
        self.options.num_similar_parameters.min_value,
        self.options.num_similar_parameters.max_value + 1)
    num_similar = min(
        num_similar,
        self.options.max_theorem_parameters - MIN_SCORED_PARAMETERS)
    return self.ranked_closest[:num_similar]

  def _score_premises(self, node: proof_search_tree.ProofSearchNode,
                      tactic_id: int, tactic: deephol_pb2.Tactic, proof_state,
                      proof_state_enc, thm_number: int) -> ScoredTheorems:
    """Returns the scored premises."""
    theorem_scores_start_time = time.time()

    # Rank assumptions from the goal as potential tactic arguments.
    assumptions_ranking_start_time = time.time()
    if self.options.only_similar:
      scored_assumptions = []
    else:
      scored_assumptions = self._compute_assumption_scores(
          node, proof_state_enc, proof_state.goal, tactic_id)
    assumptions_ranking_time_s = time.time() - assumptions_ranking_start_time
    node.assumptions_ranking_time_ms += int(
        round(1000 * assumptions_ranking_time_s))

    scored_premises = scored_assumptions

    if not tactic.only_assumptions_as_arguments:
      if self.options.only_similar:
        scored_premises = []
      else:
        scored_theorem_db = self._get_theorem_scores(proof_state_enc,
                                                     thm_number, tactic_id)
        scored_premises.extend(scored_theorem_db)

      close_theorems = self._similar_theorems(node, thm_number)
      if close_theorems:
        scored_premises.sort(reverse=True, key=lambda x: x[0])
        scored_premises = scored_premises[:self.options.max_theorem_parameters]
        # TODO(mrabe): revise _similar_theorems to return proper scores instead
        # of distance. Then uncomment the following:
        # close_theorems.sort(reverse=True)
        # close_theorems = close_theorems[:self.options.max_theorem_parameters]

        # alternate between heuristic and neural ranks
        combined_ranks = []
        tf.logging.info('Action generator: mixing premises.')
        for idx in range(max(len(close_theorems), len(scored_premises))):
          if idx < len(close_theorems):
            combined_ranks.append(close_theorems[idx])
          if idx < len(scored_premises):
            combined_ranks.append(scored_premises[idx])
        scored_premises = combined_ranks

    scored_premises.sort(reverse=True, key=lambda x: x[0])
    scored_premises = scored_premises[:self.options.max_theorem_parameters]

    thm_score_s = time.time() - theorem_scores_start_time
    node.theorem_scores_time_ms += int(round(1000.0 * thm_score_s))
    tf.logging.info('Computed theorem scores in %f seconds.', thm_score_s)
    return scored_premises

  def _step(self, node: proof_search_tree.ProofSearchNode,
            thm_number: int) -> List[Suggestion]:
    """Generates a list of possible ApplyTactic argument strings from a goal.

    Args:
      node: state of the proof search, starting at current goal.
      thm_number: Index before which we can use any theorem in the theorem
        database.

    Returns:
      List of string arguments for HolLight.ApplyTactic function, along with
      scores (Suggestion).
    """
    proof_state = proof_search_tree.proof_state_from_proof_search_node(
        node=node, history_bound=None)
    proof_state_emb_start_time = time.time()
    proof_state_emb = self.predictor.proof_state_embedding(proof_state)
    node.proof_state_emb_time_ms = int(
        round(1000.0 * (time.time() - proof_state_emb_start_time)))
    proof_state_enc = self.predictor.proof_state_encoding(proof_state_emb)
    tf.logging.debug(proof_state_enc)
    tactic_scores = self._compute_tactic_scores(proof_state_enc)

    empty_emb = self.predictor.thm_embedding('')
    empty_emb_batch = np.reshape(empty_emb, [1, empty_emb.shape[0]])

    enumerated_tactics = enumerate(self.tactics)
    if self.options.asm_meson_only:
      enumerated_tactics = [
          v for v in enumerated_tactics if str(v[1].name) == 'ASM_MESON_TAC'
      ]
      assert enumerated_tactics, 'option asm_meson_only requires ASM_MESON_TAC.'

    suggestions = []
    scored_premises = None
    pass_no_arguments = None
    node.theorem_scores_time_ms = 0
    node.assumptions_ranking_time_ms = 0
    # TODO(smloos): This computes parameters for all tactics. It should cut
    # off based on the prover BFS options.
    for tactic_id, tactic in enumerated_tactics:
      # Check if we need to compute/recompute theorem scores
      parameter_types = list(tactic.parameter_types)
      parameter_type_has_theorem = (
          parameter_types == [deephol_pb2.Tactic.THEOREM] or
          parameter_types == [deephol_pb2.Tactic.THEOREM_LIST])
      tactic_dependent_scores = (
          self.model_architecture ==
          deephol_pb2.ProverOptions.PARAMETERS_CONDITIONED_ON_TAC or
          self.model_architecture ==
          deephol_pb2.ProverOptions.GRAPH_PARAMETERS_CONDITIONED_ON_TAC)

      if (parameter_type_has_theorem and
          (scored_premises is None or tactic_dependent_scores)):
        scored_premises = self._score_premises(node, tactic_id, tactic,
                                               proof_state, proof_state_enc,
                                               thm_number)
        no_params_score = self.predictor.batch_thm_scores(
            proof_state_enc, empty_emb_batch, tactic_id)[0]
        tf.logging.info('Theorem score for empty theorem: %0.2f',
                        no_params_score)
        pass_no_arguments = scored_premises and scored_premises[-1][
            0] < no_params_score

      try:
        num_samples = None
        if self.options.num_samples_per_tactic:
          num_samples = self.options.num_samples_per_tactic

        tf.logging.debug(
            'Action generator: generating premise(s) for tactic '
            '%s.', str(tactic))
        parameter_lists = _select_parameters(parameter_types, pass_no_arguments,
                                             scored_premises, num_samples)
        for params in parameter_lists:
          suggestions.append(
              Suggestion(
                  tactic=tactic.name,
                  params=params,
                  score=tactic_scores[tactic_id]))
      except ValueError as e:
        tf.logging.warning('Failed to compute parameters for tactic %s: %s',
                           tactic.name, str(e))
    return suggestions


def _sexp_to_tokens(goal: Text, input_vocab: Dict[Text, int],
                    input_sequence_length: int):
  tokens = _split_to_tokens(goal)
  ids = [_get_id(token, input_vocab) for token in tokens]
  return _pad_to_length(ids, input_sequence_length)


def _pad_to_length(line: List[int], max_len: int):
  line = line[:max_len]
  line = line + (max_len - len(line)) * [0]
  assert len(line) == max_len, len(line)
  return line


def _split_to_tokens(th: Text):
  return th.replace('(', ' ').replace(')', ' ').split()


def _get_id(token: Text, input_vocab: Dict[Text, int]):
  try:
    i = input_vocab[token]
  except KeyError:
    i = 0
  return i


class MesonActionGenerator(object):
  """Trivial action generator, which always returns MESON tactic."""

  def step(self, unused_goal, unused_premises) -> List[Suggestion]:
    empty_premise_list = _theorem_list_parameter([])
    return [
        Suggestion(
            tactic='ASM_MESON_TAC', params=[empty_premise_list], score=1.0)
    ]
