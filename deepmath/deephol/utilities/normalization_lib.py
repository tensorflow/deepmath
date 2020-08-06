"""Simple functions for checking and modifying theorem databases."""
import sys
from typing import Callable, List, Text
import tensorflow.compat.v1 as tf
from deepmath.deephol import theorem_fingerprint
from deepmath.deephol.utilities import sexpression_graphs as sexpr
from deepmath.proof_assistant import proof_assistant_pb2


class InvalidTheoremDatabaseError(Exception):
  pass


def is_genpvar(s: Text) -> bool:
  return s.startswith('GEN%PVAR%') and s[9:].isdigit()


def is_gentype(s: Text) -> bool:
  return s.startswith('?') and s[1:].isdigit()


def extend_context(context: List[Callable[[Text], Text]], old_name: Text,
                   new_name: Text):
  """Extends the stack of variable renamings."""
  new_context = list(context)
  new_context.append(lambda s: new_name if s == old_name else s)
  return new_context


def normalize_genpvars(expr: Text):
  """Recursive function for replacement of variable names; slow but simple."""
  sys.setrecursionlimit(2000)

  def recursive_helper(expr_dag: sexpr.SExpressionGraph, node: sexpr.NodeID,
                       context):
    """Traverses DAG as if it was a tree.

    This function exploits that GEN%PVARs cannot have the same names to begin
    with unless they are normalized already.

    Args:
      expr_dag: dag of the S-expression
      node: current node
      context: A stack of variable renamings; i.e. functions from Text to Text
    """
    if expr_dag.is_leaf_node(node):
      label = expr_dag.labels[node]
      if len(label) <= 9 or not label.startswith('GEN'):
        # Cannot be GEN%PVAR%<num>
        return
      # Apply renaming context; this is a hack as it changes the internal state
      # of the DAG.
      for renaming in context:
        label = renaming(label)
      expr_dag.labels[node] = label
    else:
      if expr_dag.labels[expr_dag.children[node][0]] in ['fun', 'c']:
        # cannot contain genpvar, can be skipped; efficiency hack
        return
      var_name = expr_dag.get_bound_variable(node)
      if var_name and is_genpvar(var_name):
        # GENPVAR indices by nesting depth
        new_name = 'GEN%%PVAR%%%d' % len(context)
        context = extend_context(context, var_name, new_name)
      for child in expr_dag.children[node][1:]:  # can ignore first child
        recursive_helper(expr_dag, child, context)
    return

  # Traverse expression and rename bound GEN%PVARs
  expr_dag = sexpr.SExpressionGraph(expr)
  roots = expr_dag.roots()
  if len(roots) != 1:
    raise ValueError('Attempting to normalize S-expression without root: %s' %
                     expr)
  recursive_helper(expr_dag, roots[0], [])
  return expr_dag.to_text(roots[0])


def _words_no_parens(s: Text) -> List[Text]:
  """Returns the list of words without parentheses."""
  return s.replace('(', ' ').replace(')', ' ').split()


def _normalization_function(theorem: proof_assistant_pb2.Theorem):
  """Returns a function that normalizes terms within the given theorem."""
  terms = [theorem.conclusion]
  terms.extend(theorem.hypotheses)
  for a in theorem.assumptions:
    terms.append(a.conclusion)
    terms.extend(a.hypotheses)
  words = []
  for t in terms:
    words.extend(_words_no_parens(t))

  # Extract type numbers
  type_nums = []
  for w in words:
    if is_gentype(w):
      type_number = int(w[1:])
      if type_number not in type_nums:
        type_nums.append(type_number)

  def renaming(s: Text) -> Text:
    """The transformation on the theorem text."""
    # Renaming generic types in two steps to avoid interactions
    for idx, type_num in enumerate(type_nums):
      s = s.replace('?%d' % type_num, 'TYPE%d' % idx)
    for idx, _ in enumerate(type_nums):
      s = s.replace('TYPE%d' % idx, '?%d' % idx)
    if 'GEN%PVAR%' in s:  # only apply costly normalization if needed
      s = normalize_genpvars(s)
    return s

  return renaming


def normalize(
    theorem: proof_assistant_pb2.Theorem) -> proof_assistant_pb2.Theorem:
  """Renames types and certain variables to more unique names.

  Args:
    theorem: Theorem proto that we want to normalize.

  Returns:
    A normalized copy of the input Theorem proto.
  """
  renaming = _normalization_function(theorem)

  def rename_conclusion_and_hypotheses(theorem):
    theorem.conclusion = renaming(theorem.conclusion)
    for idx, h in enumerate(theorem.hypotheses):
      theorem.hypotheses[idx] = renaming(h)

  unique_rep = proof_assistant_pb2.Theorem()
  unique_rep.CopyFrom(theorem)  # deep-copy hypotheses, tag, split info etc.
  rename_conclusion_and_hypotheses(unique_rep)
  for a in unique_rep.assumptions:
    rename_conclusion_and_hypotheses(a)

  # we need to recompute the fingerprint since the conclusion and
  # hypotheses may have been changed during normalization
  for assum in unique_rep.assumptions:
    assum.ClearField('fingerprint')
    assum.fingerprint = theorem_fingerprint.Fingerprint(assum)
  unique_rep.ClearField('fingerprint')
  unique_rep.fingerprint = theorem_fingerprint.Fingerprint(unique_rep)
  return unique_rep


def normalize_inplace(
    theorem: proof_assistant_pb2.Theorem) -> proof_assistant_pb2.Theorem:
  """Normalizes an input Theorem proto in place.

  Args:
    theorem: Theorem proto that we want to normalize.

  Returns:
    The input Theorem proto modified to be normalized.
  """
  theorem.CopyFrom(normalize(theorem))
  return theorem


def normalized_fingerprint(theorem: proof_assistant_pb2.Theorem) -> int:
  """Turn theorems into a more unique representation and compute the fingeprint.

  Map types of the form ?XXXX to something more unique: Use ordering of the type
  numbers to assign to "?X", with number starting from 0 (per expression). Same
  variables with the name GEN%PVAR%XXXX.

  Args:
    theorem: A theorem

  Returns:
    The fingerprint of the normalized theorem
  """
  return theorem_fingerprint.Fingerprint(normalize(theorem))


def theorem_database_contains_duplicates(
    database: proof_assistant_pb2.TheoremDatabase):
  """Returns whether the database contains a duplicate w.r.t normalized fp."""
  fingerprints = set()
  for theorem in database.theorems:
    if theorem.tag != proof_assistant_pb2.Theorem.THEOREM:
      continue
    f = normalized_fingerprint(theorem)
    if f in fingerprints:
      return True
    else:
      fingerprints.add(f)
  return False


def theorem_database_contains_escaped_single_quotes(
    database: proof_assistant_pb2.TheoremDatabase):
  for theorem in database.theorems:
    if '\\\'' in theorem.conclusion or any(
        [('\\\'' in h) for h in theorem.hypotheses]):
      return True
  return False


def theorem_databases_intersect(db1: proof_assistant_pb2.TheoremDatabase,
                                db2: proof_assistant_pb2.TheoremDatabase):
  fingerprints1 = {normalized_fingerprint(t) for t in db1.theorems}
  fingerprints2 = {normalized_fingerprint(t) for t in db2.theorems}
  return fingerprints1 & fingerprints2


def validate_theorem_database(database: proof_assistant_pb2.TheoremDatabase):
  r"""Performs checks on a _single_ theorem database.

  Args:
    database: Theorem database to validate

  Raises:
    InvalidTheoremDatabaseError: In case of duplicates, or containing "\'"
  """
  if theorem_database_contains_duplicates(database):
    raise InvalidTheoremDatabaseError('Duplicate in theorem database')
  if theorem_database_contains_escaped_single_quotes(database):
    raise InvalidTheoremDatabaseError('Theorem database contains "\\\'"')


def deduplicate_modulo_normalization(db: proof_assistant_pb2.TheoremDatabase
                                    ) -> proof_assistant_pb2.TheoremDatabase:
  """Creates a new thm database with the first occurrences of each theorem."""
  seen = set()
  collisions = set()
  new_thm_db = proof_assistant_pb2.TheoremDatabase()
  num_duplicates = 0
  for t in db.theorems:
    if t.tag != proof_assistant_pb2.Theorem.THEOREM:
      new_thm_db.theorems.extend([t])
      continue
    nf = normalized_fingerprint(t)
    if nf in seen:
      num_duplicates += 1
      collisions.add(nf)
    else:
      seen.add(nf)
      new_thm_db.theorems.extend([t])
      assert normalized_fingerprint(new_thm_db.theorems[-1]) == nf
  tf.logging.info(
      'Removed %d duplicates of %d theorems; keeping earliest occurrence only.',
      num_duplicates, len(collisions))
  return new_thm_db
