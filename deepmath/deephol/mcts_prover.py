"""Prover using MCTS to search for proofs with a proof assistant."""

import tensorflow.compat.v1 as tf
from deepmath.deephol import deephol_pb2
from deepmath.deephol import mcts
from deepmath.deephol import mcts_environment
from deepmath.deephol import prover
from deepmath.proof_assistant import proof_assistant_pb2


class MCTSProver(prover.Prover):
  """A simple implementation of MCTS using goal stacks as search states."""

  def __init__(self, prover_options: deephol_pb2.ProverOptions,
               proof_assistant_wrapper,
               theorem_db: proof_assistant_pb2.TheoremDatabase,
               env: mcts_environment.Environment):
    super(MCTSProver, self).__init__(
        prover_options, proof_assistant_wrapper, theorem_db, single_goal=True)
    self.prover_options = prover_options
    self.env = env
    # TODO(mrabe): remove the need to remember the last search in self.search.
    self.search = None

  def prove_one(self,
                task: proof_assistant_pb2.ProverTask) -> deephol_pb2.ProofLog:
    """Searches for a proof with MCTS.

    Args:
      task: ProverTask to be performed.

    Returns:
      A proof log.
    """
    self.search = mcts.MonteCarloTreeSearch(self.prover_options.mcts_options)
    root_state = self.env.reset(task)
    self.search.search(root_state, self.timeout_seconds)

    proof_log = self.env.extract_proof_log()
    tf.logging.info('Nodes in proof log: %d', len(proof_log.nodes))
    for node in proof_log.nodes:
      if node.root_goal:
        tf.logging.info('Proof search with MCTS successful: %s',
                        node.status == deephol_pb2.ProofNode.PROVED)
    proof_log.search_statistics.search_depth = len(self.search.best_path)
    proof_log.search_statistics.total_expansions = self.search.total_expansions
    proof_log.search_statistics.failed_expansions = self.search.failed_expansions
    predicted_values = [state.value() for state in self.search.best_path]
    proof_log.search_statistics.mcts_path_values.extend(predicted_values)
    target_values = [state.target_value() for state in self.search.best_path]
    proof_log.search_statistics.mcts_path_target_values.extend(target_values)
    value_diffs = [abs(x - y) for x, y in zip(predicted_values, target_values)]
    proof_log.search_statistics.mcts_path_values_difference.extend(value_diffs)
    proof_log.search_statistics.mcts_path_values_squared_difference.extend(
        [d**2 for d in value_diffs])
    kl = [state.policy_kl_divergence() for state in self.search.best_path]
    proof_log.search_statistics.policy_kl_divergences.extend(kl)
    return proof_log
