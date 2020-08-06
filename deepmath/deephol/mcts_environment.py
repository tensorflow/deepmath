"""Abstract definition of a reinforcement learning environment for provers.

[1] "A general reinforcement learning algorithm that masters chess, shogi and Go
through self-play" by Silver et al.
(https://deepmind.com/documents/260/alphazero_preprint.pdf)

"""
import abc
import logging
import math
from typing import Optional, Text, Tuple
import numpy as np
import scipy.stats
import six
from deepmath.deephol import predictions
from deepmath.proof_assistant import proof_assistant_pb2


@six.add_metaclass(abc.ABCMeta)
class Action:
  """Abstract base class for actions in prover environments."""

  def __init__(self, state, probability, noise):
    self.state = state  # Is a State, but subclassing is easier without t hint.
    # Prior probabiltiy
    self.probability = probability  # type: float
    # Noise that can be added
    self.noise = noise  # type: float
    self.visit_count = 0  # N(s,a)
    self.accumulated_value = 0.0  # W(s,a)
    self.successor = None  # type: Optional[State]

  @abc.abstractmethod
  def __repr__(self):
    pass

  @abc.abstractmethod
  def _expand(self) -> 'State':
    """Evaluate the action if needed and return the successor state."""
    pass

  def expand(self):
    if self.is_expanded():
      raise ValueError('Action already expanded')
    self.successor = self._expand()
    return self.successor

  def is_expanded(self):
    return self.successor is not None

  def average_reward(self) -> float:
    """Corresponds to Q value in MCTS."""
    if self.visit_count == 0:
      return 0.
    return self.accumulated_value / self.visit_count

  def exploration_value(self) -> float:
    """Implements U as given in [1], page 17."""
    options = self.state.env.prover_options.mcts_options
    if options.HasField('c_puct'):
      c_puct = options.c_puct
    else:
      c_base = options.c_base
      c_init = options.c_init
      c_puct = np.log((1. + self.state.visit_count + c_base) / c_base) + c_init
    visit_term = np.sqrt(self.state.visit_count) / (1. + self.visit_count)
    probability = self.probability
    if (self.state.on_best_path and
        self.state.env.prover_options.mcts_options.noise_ratio_at_root):
      epsilon = self.state.env.prover_options.mcts_options.noise_ratio_at_root
      probability = (1 - epsilon) * probability + epsilon * self.noise
    return c_puct * probability * visit_term

  def mcts_score(self) -> float:
    return self.average_reward() + self.exploration_value()


@six.add_metaclass(abc.ABCMeta)
class State:
  """Abstract base class of the a state of the prover environment."""

  def __init__(self, env):
    # self.env is of type ProverEnvironment, but subclassing is easier without
    # type annotations.
    self.env = env
    self.visit_count = 0  # equal to sum of action visit counts
    self.on_best_path = False  # used to determine if to apply Dirichlet noise

  @abc.abstractproperty
  def _action_constructor(self, *args, **kwargs) -> Action:
    """Override this with the constructor of actions for each subclass."""
    pass

  @abc.abstractmethod
  def __repr__(self) -> Text:
    pass

  @abc.abstractmethod
  def actions(self) -> Tuple[Action]:
    """Return the actions of the prover - can be an empty list!"""
    pass

  @abc.abstractmethod
  def value(self) -> float:
    """Value prediction; called during the search."""
    pass

  @abc.abstractmethod
  def target_value(self) -> float:
    """Called after the search for states on the best path to generate example."""
    pass

  def select_action(self) -> Optional[Action]:
    """Select an action during building the search tree."""
    if not self.actions():
      return None
    return max(self.actions(), key=(lambda action: action.mcts_score()))

  def best_action(self) -> Optional[Action]:
    """Used by MCTS to select the best action after a search tree is built."""
    actions = self.actions()
    if not actions:
      return None
    logging.debug('MCTS env best action candidates: %s', actions)
    result = max(actions, key=(lambda a: a.visit_count))
    if not result.is_expanded():
      raise ValueError('Called best_action() on a state that has never been '
                       'expanded.')
    return result

  def policy_kl_divergence(self) -> float:
    """Difference between the prior policy and the average reward of actions."""
    prior = [a.probability for a in self.actions()]
    inferred = [a.visit_count for a in self.actions()]
    divergence = scipy.stats.entropy(prior, inferred)
    if math.isnan(divergence):
      logging.warning('KL divergence is nan for prior %s and inferred %s',
                      prior, inferred)
    return divergence

  @abc.abstractmethod
  def to_proof_state(self) -> predictions.ProofState:
    pass


@six.add_metaclass(abc.ABCMeta)
class Environment:
  """Absract base class for stateful prover environments.

  The prover environment is also a factory for states and actions.
  """

  @abc.abstractproperty
  def state_constructor(self, *args, **kwargs) -> State:
    """Override this with the constructor of states for each subclass."""
    pass

  @abc.abstractmethod
  def reset(self, task: proof_assistant_pb2.ProverTask) -> State:
    """Reset the prover environment and return the initial state."""
    pass

  @abc.abstractmethod
  def extract_proof_log(self):
    """Produce a proof log. Deprecated because this is prover specific."""
    pass
