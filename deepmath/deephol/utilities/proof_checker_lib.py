"""Exports proof logs to OCaml files to be loaded by HOL Light.

Processes multiple proof logs, but can generate at most one proof per theorem.
"""

from __future__ import absolute_import
from __future__ import division
# Import Type Annotations
from __future__ import print_function

import tensorflow as tf
from typing import Dict, Iterable, List, Text
from deepmath.deephol import deephol_pb2
from deepmath.deephol.utilities import proof_analysis
from deepmath.deephol import theorem_fingerprint
from deepmath.proof_assistant import proof_assistant_pb2


class ProofFailedError(Exception):
  pass


def put_in_quotes(s: Text):
  return '"%s"' % s


def _tactic_string_to_ocaml(tactic_string: Text) -> Text:
  return 'Parse_tactic.parse ' + put_in_quotes(tactic_string)


def tactic_application_to_string(t_app: deephol_pb2.TacticApplication) -> Text:
  """Generate tactic strings.

  Args:
    t_app: TacticApplication proto

  Returns:
    tactic string; to be parsed by third_party/hol_light/parse_tactic.ml

  Raises:
    ProofFailedError: When invariants of the tactic application are not met.
  """
  tactic_str = str(t_app.tactic)
  for i, param in enumerate(t_app.parameters):
    tactic_str += ' '
    if param.parameter_type == deephol_pb2.Tactic.UNKNOWN:
      if not param.unknown:
        raise ProofFailedError(
            'No (or empty) parameter UNKNOWN given for parameter '
            'index %d of tactic %s' % (i, t_app.tactic))
      tactic_str += str(param.unknown)
    elif param.parameter_type == deephol_pb2.Tactic.TERM:
      if not param.term:
        raise ProofFailedError('Tactic %s expected term at parameter index %d' %
                               (t_app.tactic, i))
      tactic_str += str(param.term)
    elif param.parameter_type == deephol_pb2.Tactic.THEOREM:
      if not param.theorems or len(param.theorems) != 1:
        raise ProofFailedError(
            'Tactic %s expected single theorem at parameter index %d' %
            (t_app.tactic, i))
      tactic_str += theorem_fingerprint.ToTacticArgument(param.theorems[0])
    elif param.parameter_type == deephol_pb2.Tactic.THEOREM_LIST:
      if not param.theorems:
        tactic_str += '[ ]'
      else:
        tactic_str += str('[ %s ]' % ' ; '.join([
            theorem_fingerprint.ToTacticArgument(thm) for thm in param.theorems
        ]))
    else:
      raise ProofFailedError('Unsupported param type: %s' %
                             str(param.parameter_type))
  return tactic_str


def proof_log_as_dict(log: deephol_pb2.ProofLog
                     ) -> Dict[int, deephol_pb2.ProofNode]:
  """Turns proof log into a dictionary."""
  d = {}
  for node in log.nodes:
    fingerprint = theorem_fingerprint.Fingerprint(node.goal)
    if fingerprint in d:
      raise ValueError('Duplicate subgoal in fingerprint. Ignoring')
    d[fingerprint] = node
  return d


def proof_linearization(proof_log: deephol_pb2.ProofLog
                       ) -> List[deephol_pb2.TacticApplication]:
  """Turns a proof into a list of tactic applications."""
  if not proof_log.HasField('theorem_in_database'):
    raise ValueError('Proof log requires field theorem_in_database')
  node_dict = proof_log_as_dict(proof_log)
  fingerprint = theorem_fingerprint.Fingerprint(proof_log.theorem_in_database)
  if fingerprint not in node_dict:
    raise ValueError(
        'Fingerprint of proof_log.theorem_in_database missing in the proof log.'
    )

  # Compute a linearization of the tactic applications in left-first order.
  tactics = []
  open_goals = [proof_log.theorem_in_database]
  visited = set()
  while open_goals:
    goal = open_goals.pop()
    fingerprint = theorem_fingerprint.Fingerprint(goal)
    if fingerprint in visited:
      raise ProofFailedError('Cycle detected!')
    visited.add(fingerprint)
    try:
      proofnode = node_dict[fingerprint]
    except KeyError:
      raise ProofFailedError('Subgoal not found in proof log: %s.' % str(goal))
    if not proofnode.proofs:
      raise ProofFailedError('No tactic app found for goal %s' % str(goal))
    if len(proofnode.proofs) > 1:
      tf.logging.warning('Multiple proofs detected for goal; ignoring all but '
                         'the first one.')
    tactic_application = proofnode.proofs[0]  # only checking the first one
    tactics.append(tactic_application)

    subgoals = list(tactic_application.subgoals)  # create a copy
    subgoals.reverse()  # to enable getting next goal with subgoals.pop()
    open_goals.extend(subgoals)
  return tactics


def ocaml_proof(proof_log: deephol_pb2.ProofLog) -> List[Text]:
  """Turns a proof log into OCaml code.

  Args:
    proof_log: Must contain exactly one proof of the given theorem.

  Returns:
    OCaml code for the proof.

  Raises:
    ProofFailedError: If an error in the proof is detected.
    ValueError: If an error in the checking logic is detected.
  """
  if not proof_log.HasField('theorem_in_database'):
    raise ValueError('Expected field proof_log.theorem_in_database to be set.')
  theorem = proof_log.theorem_in_database
  lines = ['']
  if theorem.pretty_printed:
    # Quotes around the expression are necessary to avoid
    # interpretation of '(*' and '*)' as nested comments.
    lines.append('(* "%s" *)' % theorem.pretty_printed)
    lines.append('')

  tactics = proof_linearization(proof_log)
  ocaml_parsed_tactics = [
      _tactic_string_to_ocaml(tactic_application_to_string(tactic))
      for tactic in tactics
  ]
  proof = ' THEN\n    '.join(ocaml_parsed_tactics)
  quoted_hypotheses = map(put_in_quotes, theorem.hypotheses)
  wrapped_proof = 'fun () ->\n    decode_goal [%s] "%s",\n    %s' % (
      '; '.join(quoted_hypotheses), theorem.conclusion, proof)
  in_core = 'true' if 'core' in theorem.library_tag else 'false'
  lines.append('register_proof %d (\n  %s) %s;;' %
               (theorem.goal_fingerprint, wrapped_proof, in_core))
  return lines


def ocaml_proof_header():
  """Creates the prelude to the OCaml file; enabling the proofs to be loaded."""
  return [
      'set_jrh_lexer;;', 'open Lib;;', 'open Printer;;',
      'open Theorem_fingerprint;;', 'open Import_proofs;;', 'open Tactics;;',
      '', 'Printer.current_encoding := Printer.Sexp;;', ''
  ]


def verify(proof_logs: Iterable[deephol_pb2.ProofLog],
           theorem_database: proof_assistant_pb2.TheoremDatabase) -> Text:
  """Generates an OCaml file of proofs for HOL Light to replay.

  Args:
    proof_logs: Proofs to be checked; assumes the top theorem is the first node
      of each proof log, and that there is at most one proof log for each
      theorem.
    theorem_database: list of theorems and definitions

  Returns:
    An OCaml file as string.

  Raises:
    ValueError: If the proof logs could not be converted to OCaml.
  """
  proof_logs_processed = 0
  proof_logs_with_closed_proofs = 0
  proof_logs_without_proof = 0

  theorems_with_closed_proofs = 0
  successful_proofs = 0
  failed_proofs = 0
  missing_proofs = 0
  missing_in_database = 0
  duplicate_proofs = 0

  # Prepare theorem databse for efficient lookup
  theorem_database_fingerprints = {
      theorem_fingerprint.Fingerprint(t) for t in theorem_database.theorems
  }

  # Count closed proofs in proof log and index by fingerprint of theorems
  proof_logs_dict = {}
  for log in proof_logs:
    proof_logs_processed += 1
    if not log.nodes or log.nodes[0].status != deephol_pb2.ProofNode.PROVED:
      proof_logs_without_proof += 1
      continue
    proof_logs_with_closed_proofs += 1

    # Ensure consistency of log.nodes[0] and log.theorem_in_database
    node0_is_thm = log.nodes[0].goal.tag == proof_assistant_pb2.Theorem.THEOREM
    if not node0_is_thm and not log.HasField('theorem_in_database'):
      raise ValueError('Not sure which theorem this log proves.')
    if not log.HasField('theorem_in_database'):
      log.theorem_in_database.CopyFrom(log.nodes[0].goal)

    # Start the actual loop logic
    fingerprint = theorem_fingerprint.Fingerprint(log.theorem_in_database)
    if fingerprint in proof_logs_dict:
      tf.logging.warning(
          'Can generate at most one OCaml proof per theorem. '
          'Dectected an additional proof for fingerprint %d.\n\n%s',
          fingerprint, str(log.nodes[0].goal))
      duplicate_proofs += 1
      continue
    proof_logs_dict[fingerprint] = log
    theorems_with_closed_proofs += 1
    if fingerprint not in theorem_database_fingerprints:
      missing_in_database += 1

  # MAIN LOOP
  lines = ocaml_proof_header()
  for theorem in theorem_database.theorems:
    # Find theorem and its proof in the proof logs
    fingerprint = theorem_fingerprint.Fingerprint(theorem)
    try:
      proof_log = proof_logs_dict[fingerprint]
    except KeyError:
      continue

    try:
      # Extract a single proof from the proof log
      extracted = proof_analysis.extract_proof(proof_log)
      if not extracted:
        raise ValueError('Proof log claims a closed proof for '
                         'fingerprint %d, but no proof could be '
                         'extracted' % fingerprint)
      lines.extend(ocaml_proof(extracted))
      successful_proofs += 1
    except ProofFailedError as e:
      tf.logging.error('Proof of %s failed: %s',
                       theorem_fingerprint.ToTacticArgument(theorem), str(e))
      failed_proofs += 1

  # Detailed stats
  tf.logging.info('PROOF LOG STATS')
  tf.logging.info('Proof logs processed: %d', proof_logs_processed)
  tf.logging.info('Proof logs without proofs: %d', proof_logs_without_proof)
  tf.logging.info('Proof logs with closed proofs: %d',
                  proof_logs_with_closed_proofs)

  tf.logging.info('PROOF STATS')
  tf.logging.info('Successful proofs: %d', successful_proofs)
  tf.logging.info('Missing proofs: %d', missing_proofs)
  tf.logging.info('Failed proofs: %d', failed_proofs)
  tf.logging.info('Theorems with proofs in proof logs: %d',
                  theorems_with_closed_proofs)
  if duplicate_proofs:
    tf.logging.warning('Proofs in proof logs that were ignored: %d',
                       duplicate_proofs)
  if missing_in_database:
    tf.logging.warning(
        'Found a proof for a theorem that is not in the theorem database',
        missing_in_database)
  if successful_proofs + failed_proofs != theorems_with_closed_proofs:
    raise ValueError('Internal error in the proof checker. Number of theorems '
                     'checked did not match the proof log.')
  if successful_proofs < theorems_with_closed_proofs or failed_proofs > 0:
    tf.logging.warning('Proof log could NOT be verified.')

  return '\n'.join(lines)
