# Lint as: python3
"""Common conversions for tactic applications in HOL Light."""

from typing import Dict, Iterable, List, Optional, Tuple, Text
import tensorflow.compat.v1 as tf
from deepmath.deephol import deephol_pb2
from deepmath.deephol import theorem_fingerprint
from deepmath.proof_assistant import proof_assistant_pb2


def extract_tactic_and_parameters(
    goal: proof_assistant_pb2.Theorem, tactic_application_string: Text
) -> Tuple[Text, List[deephol_pb2.TacticParameter]]:
  """Extract the tactic string and its parameter list from a string.

  Args:
    goal: Goal/subgoal the tactic is applied to (in case of ASSUM arguments).
    tactic_application_string: The tactic application string to be passed to
      ocaml.

  Returns:
    A pair of tactic name and tactic parameter list.
  """
  if '[' in tactic_application_string:
    s = tactic_application_string.replace(']', '').split('[')
    assert len(s) == 2, ('Expected single argument %s' %
                         tactic_application_string)
    theorems = []
    for param_string in s[1].split(';'):
      ps = param_string.strip()
      if ps:
        t = ps.split()
        assert len(t) == 2, ('Invalid tactic parameter "%s"' % ps)
        if t[0] == 'THM':
          theorems.append(proof_assistant_pb2.Theorem(fingerprint=int(t[1])))
        elif t[0] == 'ASSUM' and goal.assumptions:
          theorems.append(goal.assumptions[int(t[1])])
        else:
          raise ValueError('Invalid tactic parameter "%s"' % ps)

    return s[0].strip(), [
        deephol_pb2.TacticParameter(
            parameter_type=deephol_pb2.Tactic.THEOREM_LIST, theorems=theorems)
    ]
  elif '`' in tactic_application_string:
    tactic_split = tactic_application_string.replace('`', '').split()
    assert len(tactic_split
              ) >= 2, 'Invalid tactic string "%s"' % tactic_application_string
    return tactic_split[0], [
        deephol_pb2.TacticParameter(
            parameter_type=deephol_pb2.Tactic.TERM,
            term=' '.join(tactic_split[1:]))
    ]
  else:
    s = tactic_application_string.split()
    if len(s) == 1:
      return s[0], []
    elif len(s) == 3 and s[1] in ['THM', 'ASSUM']:
      try:
        if s[1] == 'THM':
          premise = proof_assistant_pb2.Theorem(fingerprint=int(s[2]))
        elif s[1] == 'ASSUM':
          assumption_idx = int(s[2])
          premise = goal.assumptions[assumption_idx]
        return s[0], [
            deephol_pb2.TacticParameter(
                parameter_type=deephol_pb2.Tactic.THEOREM, theorems=[premise])
        ]
      except Exception as e:  # pylint: disable=broad-except
        tf.logging.error('Error %s during parsing of tactic "%s".', e,
                         tactic_application_string)
    tf.logging.error('Could not extract TacticApplication from %s',
                     tactic_application_string)
    return s[0], [
        deephol_pb2.TacticParameter(
            parameter_type=deephol_pb2.Tactic.UNKNOWN, unknown=' '.join(s[1:]))
    ]


def _theorem_parameter_string(
    param: proof_assistant_pb2.Theorem,
    asm_indices: Optional[Dict[int, int]] = None) -> Text:
  """Compute string from theorem parameter.

  Args:
    param: The tactic parameter to turn to a string.
    asm_indices: Optional map from fingerprints to assumption index. If given,
      acts as an override to param.assumption_index.

  Returns:
    Either 'ASSUM idx' or 'THM fp'.
  """
  fp = theorem_fingerprint.Fingerprint(param)
  if param.HasField('assumption_index'):
    assumption_index = param.assumption_index
    if asm_indices is not None:  # override for assumption_index
      if fp not in asm_indices:
        # This error message is has a non-trivial effect: It asserts that this
        # tactic expected this premise to be in the assumption list. It makes
        # sure that if an upstream tactic imported a theorem from the theorem
        # database, we don't allow that import to be pruned away.
        raise ValueError('Theorem parameter is marked as assumption, but was '
                         'not found in the goal.')
      assumption_index = asm_indices[fp]
    return 'ASSUM %d' % assumption_index
  else:
    return 'THM %d' % fp


def tactic_application_to_string(t_app: deephol_pb2.TacticApplication) -> Text:
  """Generate a tactic string from a tactic application proto."""
  return tactic_string(t_app.tactic, t_app.parameters)


def tactic_string(tactic: Text,
                  params: Iterable[deephol_pb2.TacticParameter],
                  asm_indices: Optional[Dict[int, int]] = None) -> Text:
  """Computes the tactic string as compatible with the HOL Light tactic parser.

  Args:
    tactic: The tactic without the parameters, e.g. "MESON_TAC".
    params: Tactic parameters.
    asm_indices: Override for theorem.assumption_index of theorem parameters. Is
      used during pruning when assumption indices shift.

  Returns:
    A string that can be parsed by the tactic_parser of HOL Light.
    E.g. "MESON_TAC [ THM 12345 ; ASSUM 2 ]".
  """
  param_strings = []
  for param in params:
    if param.parameter_type == deephol_pb2.Tactic.THEOREM:
      param_strings.append(
          _theorem_parameter_string(param.theorems[0], asm_indices))
    elif param.parameter_type == deephol_pb2.Tactic.THEOREM_LIST:
      theorem_strings = [
          _theorem_parameter_string(t, asm_indices) for t in param.theorems
      ]
      if theorem_strings:
        param_strings.append('[ %s ]' % ' ; '.join(theorem_strings))
      else:
        param_strings.append('[ ]')  # avoiding surplus space
    elif param.parameter_type == deephol_pb2.Tactic.TERM:
      param_strings.append('`%s`' % param.term)
    elif param.parameter_type == deephol_pb2.Tactic.CONV:
      param_strings.append(param.conv)
    elif param.parameter_type == deephol_pb2.Tactic.UNKNOWN:
      tf.logging.error('Parameter with parameter_type UNKNOWN: %s', param)
      param_strings.append(param.unknown)
    else:
      raise ValueError('Unknown parameter type: %d; corresponding to '
                       'parameter: %r' % (param.parameter_type, param))
  if not param_strings:
    return str(tactic)
  return str('%s %s' % (tactic, ' '.join(param_strings)))


def assumption_indices(goal: proof_assistant_pb2.Theorem) -> Dict[int, int]:
  """Computes a map from fingerprints of assumptions to assumption_index."""
  if goal.tag != proof_assistant_pb2.Theorem.GOAL:
    raise ValueError('Expected goal')
  index_map = {}
  for index, assumption in list(enumerate(goal.assumptions))[::-1]:
    # inverted order to handle duplicates correctly
    if not assumption.HasField('assumption_index'):
      tf.logging.error('assumption_index not set.')
      assumption.assumption_index = index
    if assumption.assumption_index != index:
      raise ValueError('Inconsistent assumption_index.')
    index_map[theorem_fingerprint.Fingerprint(assumption)] = index
  return index_map
