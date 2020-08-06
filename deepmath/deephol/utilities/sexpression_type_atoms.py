# Lint as: python3
"""Inserts <TYPE> atoms into an S-expression.

Inserts an additional <TYPE> atom as the first atom in each subexpression
representing a type constructor application with zero or more arguments.

The implementation of this library is HOL Light S-expression syntax specific.

S-expressions with bound variable names replaced with their de Bruijn indexes
are also supported.
"""
from typing import List, Optional, Text

from deepmath.deephol.utilities import hol_light_sexpression_syntax
from deepmath.deephol.utilities import hol_light_sexpression_trees
from deepmath.deephol.utilities import sexpression_parser


def insert_type_atoms(
    sexp: Text, kind: hol_light_sexpression_syntax.SyntaxElementKind) -> Text:
  """Inserts <TYPE> atoms into an S-expression.

  Inserts an additional <TYPE> atom as the first atom in each subexpression
  representing a type constructor application with zero or more arguments.
  The <TYPE> atoms represent the kind of the S-expression in the same way as
  'v' atoms do for variables, 'c' atoms do for constants, etc.

  The implementation of this function is HOL Light S-expression syntax specific.

  Args:
    sexp: An S-expression string.
    kind: Syntax element kind represented by the S-expression.

  Returns:
    An S-expression string with <TYPE> atoms inserted.
  """
  if not sexp:
    return ''
  result_parts = []  # type: List[Text]

  def _enter_fn(node: sexpression_parser.SExpressionTreeNode,
                node_kind: hol_light_sexpression_syntax.SyntaxElementKind,
                index_in_parent: Optional[int]) -> None:
    """Adds the start of string representation of a node to result_parts.

    Adds the part of the string representation which precedes children. Also
    adds the <TYPE> atom where necessary. If there are no children, adds the
    whole string representation.

    Must be invoked for each node pre-order in a depth first traversal of an
    S-expression tree.

    Args:
      node: The current S-expression node.
      node_kind: The syntax element kind represented by the node.
      index_in_parent: The index of the current node in its parent node. None
        for the root node.
    """
    if index_in_parent is not None and index_in_parent > 0:
      result_parts.append(' ')
    if not node.children:
      result_parts.append(repr(node))
      return
    result_parts.append('(')
    if node_kind == hol_light_sexpression_syntax.SyntaxElementKind.TYPE:
      result_parts.append(hol_light_sexpression_syntax.TYPE_ATOM)
      result_parts.append(' ')

  def _exit_fn(node: sexpression_parser.SExpressionTreeNode,
               node_kind: hol_light_sexpression_syntax.SyntaxElementKind,
               index_in_parent: Optional[int]) -> None:
    """Adds the end of string representation of a node to result_parts.

    Adds the part of the string representation which succeeds children. If there
    are no children, adds nothing.

    Must be invoked for each node post-order in a depth first traversal of an
    S-expression tree.

    Args:
      node: The current S-expression node.
      node_kind: The syntax element kind represented by the node.
      index_in_parent: The index of the current node in its parent node. None
        for the root node.
    """
    del node_kind, index_in_parent  # Unused.
    if not node.children:
      return
    result_parts.append(')')

  tree = sexpression_parser.to_tree(sexp)
  hol_light_tree = hol_light_sexpression_trees.HolLightSExpressionTree(
      tree, kind, has_type_atoms=False)
  hol_light_tree.traverse_depth_first(_enter_fn, _exit_fn)
  return ''.join(result_parts)
