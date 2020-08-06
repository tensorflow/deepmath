"""Simple implementation of MCTS.

This implementation is an adaption of
[1] "Bandit based Monte-Carlo Planning" by Levente Kocsis and Csaba Szepesvari
(http://ggp.stanford.edu/readings/uct.pdf),
and
[2] "Multi-armed Bandits with Episode Context" by Christopher D. Rosin.
(http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.172.9450&rep=rep1&type=pdf)
and
[3] "A general reinforcement learning algorithm that masters chess, shogi and Go
through self-play" by Silver et al.
(https://deepmind.com/documents/260/alphazero_preprint.pdf)

POTENTIALLY CONFUSING TERMINOLOGY: The MCTS search develops a tree of
SearchStates which are connected by Actions. But each SearchState consists of a
set of nodes of a proof_search_tree, which are connected by TacticApplications.
"""

import time
from typing import List, NewType, Optional, Text
import tensorflow.compat.v1 as tf
from deepmath.deephol import deephol_pb2  # for MCTSOptions
from deepmath.deephol import mcts_environment

Action = mcts_environment.Action
SearchState = mcts_environment.State

# A path from the root of the MCTS tree to an unevaluated action.
Path = NewType('Path', List[Action])


class MonteCarloTreeSearch:
  """A simple implementation of Alpha-Go style tree search."""

  def __init__(self, mcts_options: deephol_pb2.MCTSOptions):
    self.mcts_options = mcts_options

    self.total_expansions = 0  # type: int
    self.failed_expansions = 0  # type: int
    self.best_path = []  # type: List[SearchState]
    self.exit_message = ''
    self.timeout_seconds = None
    self.start_time = None

  def _select(self, search_state: SearchState) -> Optional[Path]:
    """Select a search state to expand.

    Starting from the root, select successors until we hit an unexpanded state
    action pair.

    The action generator is one of the two costly operations. To save some time,
    we generate actions only when we need to sample an action.

    Args:
      search_state: The SearchState that is the CURRENT root of the MCTS.

    Returns:
      A path from the root to an unexpanded state-action pair.
    """
    path = []
    while True:
      action = search_state.select_action()
      if action is None:
        # search_state has no actions
        if not path:
          return None
        return Path(path)  # hit a deadend
      path.append(action)
      if not action.is_expanded():
        return Path(path)  # found a path to expand
      search_state = action.successor

  def _backup(self, path: Path):
    """Evaluate the next_search_state and update the values along the path."""
    value = 0.0
    if path[-1].successor:
      value = path[-1].successor.value()
    for action in path[::-1]:
      action.visit_count += 1
      action.state.visit_count += 1
      action.accumulated_value += value
      value *= self.mcts_options.discount_factor

  def _timed_out(self) -> bool:
    """Returns whether the search has timed out."""
    return (self.start_time is not None and
            time.time() - self.start_time > self.timeout_seconds)

  def _out_of_resources(self) -> Optional[Text]:
    """Returns a message if the computational resources are exhausted."""
    if self._timed_out():
      return 'timed out'
    if (self.mcts_options.max_total_expansions and
        self.total_expansions >= self.mcts_options.max_total_expansions):
      return 'global expansion limit exhausted'
    if (self.mcts_options.max_search_depth and
        len(self.best_path) >= self.mcts_options.max_search_depth):
      return 'maximum MCTS search depth exceeded'
    return None

  def _build_search_tree(self, state: SearchState):
    """Build a search tree up to mcts_options.max_expansions states."""
    tf.logging.info(
        'Fresh MCTS search; %d expansions so far; number of MCTS '
        'searches so far: %d', self.total_expansions, len(self.best_path))
    while (self._out_of_resources() is None and
           state.visit_count < self.mcts_options.max_expansions):
      path = self._select(state)
      if not path:
        break
      last_action = path[-1]
      if not last_action.is_expanded():
        expanded_state = last_action.expand()
        tf.logging.info('Expanded search state: %s', expanded_state)
      else:
        self.failed_expansions += 1
      self.total_expansions += 1
      self._backup(path)

  def search(self, root_state: SearchState, timeout_seconds: Optional[float]):
    """Monte-Carlo tree search.

    Repeatedly build a fixed-size search tree before committing to the best
    next action.

    Args:
      root_state: SearchState from which to start the search
      timeout_seconds: timeout for the search in seconds or None
    """
    self.timeout_seconds = timeout_seconds
    self.start_time = time.time()
    next_state = root_state
    while True:
      next_state.on_best_path = True
      self._build_search_tree(next_state)
      if self._out_of_resources() is not None:
        self.exit_message = 'MCTS: %s.' % self._out_of_resources()
        break
      self.best_path.append(next_state)
      best_action = next_state.best_action()
      if best_action is None:
        self.exit_message = 'MCTS: No action to continue with.'
        break
      next_state = best_action.successor

    total_time = time.time() - self.start_time

    if self.exit_message:
      self.exit_message += ' (value of last state: %f; search depth: %d)' % (
          next_state.value(), len(self.best_path))
    tf.logging.info(self.exit_message)
    tf.logging.info('Time spent in ms: %d', total_time * 1000)
    tf.logging.info('Timeout: %s', self._timed_out())
    tf.logging.info('MCTS search depth: %d', len(self.best_path))
    tf.logging.info('Total number of expansions: %d', self.total_expansions)
