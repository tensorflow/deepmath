"""Turn goals, theorems, proof states into SExpression strings."""

from typing import List, Optional, Text

from deepmath.deephol import predictions
from deepmath.proof_assistant import proof_assistant_pb2


def convert_goal(goal: proof_assistant_pb2.Theorem,
                 conclusion_only: bool) -> Text:
  """Converts a GOAL Theorem proto object into an SExpression string.

  If conclusion_only = True, returns only the conclusion
  (used for backward-compatibility).

  If conclusion_only = False, the output looks as follows:
  1) Starts with a special token "<goal>".
  2) Next, lists all the assumptions space-separated. Note that the whole
     assumption-list is not itself enclosed in brackets.
  3) Ends with the conclusion string.

  Args:
    goal: Theorem proto object of type GOAL.
    conclusion_only: Whether only the conclusion is returned.

  Returns:
    Input Theorem proto converted into an SExpression string.
  """
  if not goal.HasField('tag') or goal.tag != proof_assistant_pb2.Theorem.GOAL:
    raise ValueError('Expected Goal tag in the input Theorem proto.')
  if goal.hypotheses:
    raise ValueError('Goals cannot have hypotheses.')
  if conclusion_only:
    # Assumptions are ignored, only the conclusion is returned.
    # Do not add the <goal> token to keep translation backward-compatible.
    return goal.conclusion
  if not goal.conclusion:
    raise ValueError('Goal with empty-string conclusion.')
  # Assumptions are included, special <goal> token at the beginning.
  if goal.assumptions:
    assumptions_string = ' '.join(
        [convert_theorem(a, conclusion_only) for a in goal.assumptions])
    result = '(<goal> %s %s)' % (assumptions_string, goal.conclusion)
  else:
    result = '(<goal> %s)' % goal.conclusion
  return result


def convert_theorem(theorem: proof_assistant_pb2.Theorem,
                    conclusion_only: bool) -> Text:
  """Converts a THEOREM/(TYPE_)DEFINITION proto into an SExpression string.

  If conclusion_only = True, returns only the conclusion
  (used for backward-compatibility).

  If conclusion_only = False, the output looks as follows:
  1) Starts with a special token "<theorem>".
  2) Ends with the conclusion string.

  We (for now) require hypotheses field to be empty in theorem protos.

  Args:
    theorem: Theorem proto object of type THEOREM/DEFINITION/TYPE_DEFINITION.
    conclusion_only: Whether only the conclusion is returned.

  Returns:
    Input Theorem proto converted into an SExpression string.
  """
  if theorem.tag not in [
      proof_assistant_pb2.Theorem.THEOREM,
      proof_assistant_pb2.Theorem.DEFINITION,
      proof_assistant_pb2.Theorem.TYPE_DEFINITION
  ]:
    raise ValueError('Expected one of {Theorem, Definition, Type_definition}'
                     ' tags in the input Theorem proto.')
  if theorem.tag in [
      proof_assistant_pb2.Theorem.DEFINITION,
      proof_assistant_pb2.Theorem.TYPE_DEFINITION
  ] and theorem.hypotheses:
    raise ValueError('Detected (type_)definition with hypotheses.')
  if conclusion_only:
    # Assumptions are ignored, only the conclusion is returned.
    # Do not add the <theorem> token to keep translation backward-compatible.
    return theorem.conclusion
  # Handle the special case of an empty theorem (no hypotheses, conclusion '')
  # We pass such theorem to represent an empty list of theorems
  # (e.g. an empty list of tactic parameters).
  if not theorem.conclusion:
    if theorem.hypotheses:
      raise ValueError('Theorem with hypotheses and empty-string conclusion.')
    return ''
  # Hypotheses are included, special <theorem> token at the beginning.
  if theorem.hypotheses:
    result = '(<theorem> %s %s)' % (' '.join(
        theorem.hypotheses), theorem.conclusion)
  else:
    result = '(<theorem> %s)' % theorem.conclusion
  return result


def _convert_search_state(proof_state: predictions.ProofState,
                          conclusion_only: bool) -> Text:
  """Private helper that converts search state into an SExpression string.

  Output format:
  (<search_state> sexp_1 sexp_2 ... sexp_n)
  Where:
  - <search_state> is a special token,
  - sexp_i is SExpression of the i-th goal in the search state.

  Args:
    proof_state: Proof State that contains the search state field.
    conclusion_only: Whether only conclusions of goals are represented.

  Returns:
    Search state of the input Proof State converted into an SExpression string.
  """
  assert proof_state.search_state
  goal_sexpressions = [
      convert_goal(goal, conclusion_only) for goal in proof_state.search_state
  ]
  return '(<search_state> %s)' % ' '.join(goal_sexpressions)


def _add_targets(
    sexpr: Text,
    targets: Optional[List[proof_assistant_pb2.Theorem]],
    conclusion_only: bool,
) -> Text:
  r"""Create a target conditioned goal sexpression for HER.

  Introduces two tokens <HER> and <target>. Resulting sexpression tree looks
  like:

             <HER>
             /  \
         `sexpr`  <target>
                    |
                   ...

  Flattened, the sexpression looks like: (<HER> (sexpr) (<target> ...)).

  If `targets` is `None` or `[]`, then simply returns `sexpr`.

  Note: `target` is put a last sexpression since the current GNNs
  encoding is able to distinguish the last element of a list from
  all others.

  Args:
    sexpr: SExpression to add `target` to.
    targets: Whether only conclusions of `targets` are represented.
    conclusion_only: Where

  Returns:
    Target annotated SExpression.
  """
  if not targets:
    return sexpr
  targets = ' '.join(convert_goal(goal, conclusion_only) for goal in targets)
  return f'(<HER> {sexpr}, (<target> {targets}))'


def convert_proof_state(proof_state: predictions.ProofState, history_bound: int,
                        conclusion_only: bool) -> Text:
  """Converts Proof State object into an SExpression string.

  If history_bound = 0, returns only SExpression of the current goal.

  Otherwise, the output looks as follows:
  (<proof_state_history> sexp_x sexp_x-1 ... sexp_1 sexp_current)
  Where:
  - <proof_state_history> is a special token,
  - sexp_y is SExpression of the goal y steps before the current goal
    (we consider only y <= history_bound in order to handle deep proof trees),
  - sexp_current is SExpression of the current goal.

  Args:
    proof_state: Proof State object to be converted.
    history_bound: How much (if any) history of the proof state is represented.
    conclusion_only: Whether only conclusions of goals are represented.

  Returns:
    Input Proof State converted into an SExpression string.
  """
  if history_bound == 0:
    if proof_state.search_state:
      output = _convert_search_state(proof_state, conclusion_only)
    else:
      output = convert_goal(proof_state.goal, conclusion_only)
  else:
    assert history_bound > 0

    sexpressions = []
    state_ptr = proof_state
    history_added = -1
    while state_ptr is not None and history_added < history_bound:
      if state_ptr.search_state:
        sexpressions.append(_convert_search_state(state_ptr, conclusion_only))
      else:
        sexpressions.append(convert_goal(state_ptr.goal, conclusion_only))
      state_ptr = state_ptr.previous_proof_state
      history_added += 1
    output = '(<proof_state_history> %s)' % ' '.join(reversed(sexpressions))

  return _add_targets(output, proof_state.targets, conclusion_only)
