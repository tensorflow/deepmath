"""RL environment for theorem provers.

Currently this environment gets its actions from the action generator.
"""
import math
import time
from typing import List, Optional, Text, Tuple
import numpy as np
import tensorflow.compat.v1 as tf
from deepmath.deephol import abstract_action_generator
from deepmath.deephol import action_generator
from deepmath.deephol import deephol_pb2
from deepmath.deephol import mcts_environment
from deepmath.deephol import predictions
from deepmath.deephol import proof_search_tree
from deepmath.deephol.utilities import deephol_stat_pb2
from deepmath.deephol.utilities import stats
from deepmath.proof_assistant import proof_assistant_pb2

MAX_GOALS_PER_SEARCH_STATE = 20
MAX_ASSUMPTIONS_PER_GOAL = 30


class TacticAction(mcts_environment.Action):
  """An tactic or rule applied to a ProofSearchNode."""

  def __init__(self, node, tactic, params, score, state, probability, noise):
    super(TacticAction, self).__init__(state, probability, noise)
    self.node = node  # type: proof_search_tree.ProofSearchNode
    self.tactic = tactic  # type: Text
    self.params = params  # type: List[deephol_pb2.TacticParameter]
    self.score = score  # type: float
    self.tactic_application = None  # type: Optional[proof_search_tree.TacticApplication]

  def __repr__(self):
    return ('Action(P=%f, tactic=%s, succ=%r, goal=%s)') % (
        self.probability, self.tactic, self.successor is not None,
        self.node.goal.conclusion)

  def _expand(self) -> 'State':
    """Evaluate the action, and find or create a successor state."""
    # Evaluate tactic only if it hasn't been evaluated before.
    # Assuming small number of tactics per node; performing linear search.
    for app in self.node.successful_attempts + self.node.failed_attempts:
      if app.tactic == self.tactic and app.parameters == self.params:
        self.tactic_application = app
        break
    if not self.tactic_application:
      self.tactic_application = (
          self.node.apply_tactic(
              self.tactic, self.params,
              self.state.env.prover_options.tactic_timeout_ms, self.score))
    result_code = self.tactic_application.result
    if result_code == deephol_pb2.TacticApplication.SUCCESS:
      next_goals = [  # remaining goals from the previous state
          node for node in self.state.nodes if node is not self.node
      ]
      next_goals.extend(self.tactic_application.subgoals)
      successor = self.state.env.state_constructor(self.state.env, next_goals)
    else:
      successor = self.state.env.error_state
      if self.state.env.prover_options.mcts_options.avoid_expanding_failed_tactics:
        self.probability = 0.0
    return successor


class State(mcts_environment.State):
  """A state of the prover environment; represents a goal stack."""

  _action_constructor = TacticAction

  def __init__(self,
               env: 'ProverEnvironment',
               nodes: List[proof_search_tree.ProofSearchNode],
               failed: bool = False):
    """Initializer.

    Args:
      env: The environment this state belongs to.
      nodes: The goal stack; represented as a set. All of these goals have to be
        closed for this search state to be successful.
      failed: This state is a lost case - no actions and value 0.
    """
    super(State, self).__init__(env)
    self.nodes = nodes
    self.failed = failed
    self._actions = None  # type: Optional[Tuple[TacticAction]]
    self._value = None  # type: Optional[float]

    # Precomputing fingerprints and hash for efficient comparison and lookup
    self.goal_fingerprints = [node.goal.fingerprint for node in self.nodes]
    self.goal_fingerprints.sort()
    # For statistics
    env.states.append(self)
    self.value_prediction_time_sec = 0.0

  def __repr__(self) -> Text:
    return 'State: %s (failed: %s, visit count: %d, value: %s)' % (
        self.goal_fingerprints, self.failed, self.visit_count, str(
            self.value()))

  def is_terminal(self) -> bool:
    """A state is terminal when it does not have actions."""
    if self.failed or self.target_reached():
      return True
    for node in self.nodes:
      if len(node.goal.assumptions) > MAX_ASSUMPTIONS_PER_GOAL:
        return True
    return False

  def target_reached(self) -> bool:
    """Check if goal stack is a subset of the targets."""
    if self.failed:
      return False
    # TODO(marcellvc): Use fingerprints instead.
    return all(self.env.tree.within_targets(node.goal) for node in self.nodes)

  def actions(self) -> Tuple[TacticAction]:
    """Call the action generator, if needed."""
    if self.is_terminal():
      self._actions = tuple()
      return self._actions
    if self._actions is None:
      # The following code uses the scores from the action generator and applies
      # a softmax.
      tf.logging.info('Generating actions.')
      assert self.nodes  # because self is not terminal
      suggestions = []  # collect suggestions for all nodes
      first_node = self.nodes[0]
      if first_node not in self.env.suggestions_store:
        suggestions = self.env.action_gen.step(first_node,
                                               self.env.task.premise_set)
        if not suggestions:
          tf.logging.warning('Action generator failed to generate actions.')
        self.env.suggestions_store[first_node] = suggestions
      suggestions = self.env.suggestions_store[first_node]
      if not suggestions:
        self._actions = tuple()
        return self._actions

      if (self.env.prover_options.mcts_options.HasField('max_suggestions') and
          len(suggestions) >
          self.env.prover_options.mcts_options.max_suggestions):
        # descending scores
        suggestions.sort(reverse=True, key=lambda s: s.score)
        suggestions = suggestions[:self.env.prover_options.mcts_options
                                  .max_suggestions]

      probabilities = action_generator.logits_to_probabilities(
          [score for _, _, score in suggestions])

      if self.env.prover_options.mcts_options.dirichlet_noise_alpha:
        alpha = self.env.prover_options.mcts_options.dirichlet_noise_alpha
        noise = np.random.dirichlet(np.ones(len(suggestions)) * alpha)
      else:
        noise = np.zeros(len(suggestions))

      suggestions = zip(suggestions, probabilities, noise)
      actions = []
      for (tac, params, score), probability, noise in suggestions:
        action = State._action_constructor(first_node, tac, params, score, self,
                                           probability, noise)
        actions.append(action)
      probabilities = [action.probability for action in actions]
      tf.logging.info('Prover environment action probabilities: %s',
                      probabilities)
      if abs(1 - sum(probabilities)) > 0.00001:
        tf.logging.error('Not a probability distribution: %s',
                         str(probabilities))
      self._actions = tuple(actions)
    return self._actions

  def value(self) -> float:
    if self._value is None:
      if self.failed:
        self._value = 0.0
      elif self.target_reached():
        self._value = 1.0
      else:
        self._value = self._estimated_value()
    return self._value

  def target_value(self) -> float:
    if not self.on_best_path:
      raise ValueError('target_value should not have been called.')
    if self.failed:
      return 0.0
    if self.target_reached():
      return 1.0
    best = self.best_action()
    if best is None:
      return 0.0
    return best.average_reward()

  def _estimated_value(self) -> float:
    """Predictor gets search state value."""
    start_time = time.time()
    value = self.env.predictor.search_state_score(
        proof_state=self.to_proof_state())
    total_time = time.time() - start_time
    self.env.total_value_prediction_time_sec += total_time
    self.value_prediction_time_sec += total_time
    if math.isnan(value):
      raise ValueError('Observed NaN value from value network.')
    return value

  def to_proof_state(self) -> predictions.ProofState:
    if self.nodes:
      return predictions.ProofState(
          search_state=[node.goal for node in self.nodes])
    else:
      if self.failed:
        goal = proof_assistant_pb2.Theorem(
            conclusion='(c bool f)',
            tag=proof_assistant_pb2.Theorem.GOAL)  # hack
      else:
        goal = proof_assistant_pb2.Theorem(
            conclusion='(c bool t)',
            tag=proof_assistant_pb2.Theorem.GOAL)  # hack
      return predictions.ProofState(search_state=[goal])


class ProverEnvironment(mcts_environment.Environment):
  """Stateful prover environment simulating a goal stack."""

  def __init__(self, prover_options: deephol_pb2.ProverOptions, proof_assistant,
               action_gen: abstract_action_generator.AbstractActionGenerator,
               predictor: predictions.Predictions):
    self.prover_options = prover_options
    self.action_gen = action_gen
    self.predictor = predictor
    self.proof_assistant = proof_assistant
    self.states = []
    self.initial_state = None
    self.suggestions_store = None
    self.task = None
    self.error_state = self.state_constructor(self, [], failed=True)
    self.tree = None

    # Statistics
    self.total_value_prediction_time_sec = 0.0

  def reset(self, task: proof_assistant_pb2.ProverTask) -> State:
    self.task = task
    if len(task.goals) > 1:
      raise ValueError('Can currently only process tasks with single goals.')
    goal = task.goals[0]
    if goal.tag != proof_assistant_pb2.Theorem.GOAL:
      raise ValueError('ProverTask contained goal with tag: %s' % str(goal.tag))
    if len(task.targets) > 1:
      raise ValueError('Can currently only process task with 0 or 1 targets.')
    # TODO(marcellvc): Have ProofSearchTree get goal from task.
    self.tree = proof_search_tree.ProofSearchTree(self.proof_assistant, goal,
                                                  task)
    self.suggestions_store = dict()
    self.states = [self.error_state]
    self.initial_state = self.state_constructor(self, [self.tree.nodes[0]])
    return self.initial_state

  def extract_proof_log(self) -> deephol_pb2.ProofLog:
    if self.tree is None:
      raise ValueError('Cannot extract proof log before reset(task) is called.')
    proof_log = self.tree.to_proto()
    proof_log.search_statistics.total_prediction_time_sec = self.total_value_prediction_time_sec
    proof_log.search_statistics.num_search_states = len(self.states)
    values = [s.value() for s in self.states]
    proof_log.search_statistics.mcts_values_all_states.extend(values)
    proof_log.search_statistics.goals_per_search_state.CopyFrom(
        deephol_stat_pb2.Histogram(
            max_value=float(MAX_GOALS_PER_SEARCH_STATE),
            num_buckets=MAX_GOALS_PER_SEARCH_STATE))
    proof_log.search_statistics.assumptions_per_goal.CopyFrom(
        deephol_stat_pb2.Histogram(
            max_value=float(MAX_ASSUMPTIONS_PER_GOAL),
            num_buckets=MAX_ASSUMPTIONS_PER_GOAL))
    for state in self.states:
      stats.add_to_histogram(proof_log.search_statistics.goals_per_search_state,
                             len(state.nodes))
    for node in self.tree.nodes:
      stats.add_to_histogram(proof_log.search_statistics.assumptions_per_goal,
                             len(node.goal.assumptions))
    return proof_log

  def state_constructor(self, *args, **kwargs):
    return State(*args, **kwargs)
