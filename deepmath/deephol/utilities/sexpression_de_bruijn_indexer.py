# Lint as: python3
"""Renames bound variables in S-expressions using their de Bruijn indexes.

The implementation of this library is HOL Light S-expression syntax specific.
"""
from typing import Dict, List, Optional, Text, Tuple

from deepmath.deephol.utilities import hol_light_sexpression_syntax
from deepmath.deephol.utilities import hol_light_sexpression_trees
from deepmath.deephol.utilities import sexpression_parser


def rename_bound_variables(sexp: Text) -> Text:
  """Renames bound variables in an S-expression using their de Bruijn indexes.

  The implementation of this function is HOL Light S-expression syntax specific.

  Args:
    sexp: An S-expression string.

  Returns:
    An S-expression string with all the bound variable names replaced with their
    de Bruijn indexes.
  """
  if not sexp:
    return ''
  tree = sexpression_parser.to_tree(sexp)
  hol_light_tree = hol_light_sexpression_trees.HolLightSExpressionTree(
      tree,
      hol_light_sexpression_syntax.SyntaxElementKind.UNKNOWN,
      has_type_atoms=False)

  def _compute_max_indexes(
  ) -> Dict[sexpression_parser.SExpressionTreeNode, int]:
    """Computes max de Bruijn indexes among variables bound inside each node.

    Returns:
      A dictionary which stores each node of the tree as a key and max de Bruijn
      index among variables bound inside the node as a value. Zero value is used
      for nodes which have no variables bound inside them.
    """
    max_indexes = {}  # type: Dict[sexpression_parser.SExpressionTreeNode, int]

    def _exit_fn(node: sexpression_parser.SExpressionTreeNode,
                 node_kind: hol_light_sexpression_syntax.SyntaxElementKind,
                 index_in_parent: Optional[int]) -> None:
      """Computes max de Bruijn index for the given node.

      Must be invoked for each node post-order in a depth first traversal of an
      S-expression tree.

      The computed indexes are stored in max_indexes dictionary.

      Args:
        node: The current S-expression node.
        node_kind: The syntax element kind represented by the node.
        index_in_parent: The index of the current node in its parent node. None
          for the root node.
      """
      del index_in_parent  # Unused.
      max_index = (
          max([max_indexes[child_node] for child_node in node.children])
          if node.children else 0)
      if hol_light_tree.is_abstraction_tree_node(node, node_kind):
        max_index = max_index + 1
      max_indexes[node] = max_index

    hol_light_tree.traverse_depth_first(enter_fn=None, exit_fn=_exit_fn)
    return max_indexes

  max_indexes = _compute_max_indexes()

  # Contains de Bruijn indexes for bound variables identified by tuples
  # of their names and types. The indexes are stored as a list to support bound
  # variables with the same name and type. The current index is the last index
  # in the list.
  bound_variable_indexes = {}  # type: Dict[Tuple[Text, Text], List[int]]
  result_parts = []  # type: List[Text]

  def _get_variable_id(variable_node: sexpression_parser.SExpressionTreeNode):
    variable_name = repr(
        hol_light_tree.get_variable_name_tree_node(variable_node))
    variable_type = repr(
        hol_light_tree.get_variable_type_tree_node(variable_node))
    return (variable_name, variable_type)

  def _get_abstraction_variable_id(
      abstraction_node: sexpression_parser.SExpressionTreeNode):
    return _get_variable_id(
        hol_light_tree.get_abstraction_variable_tree_node(abstraction_node))

  def _enter_fn(node: sexpression_parser.SExpressionTreeNode,
                node_kind: hol_light_sexpression_syntax.SyntaxElementKind,
                index_in_parent: Optional[int]) -> None:
    """Adds the start of string representation of a node to result_parts.

    Adds the part of the string representation which precedes children. If there
    are no children, adds the whole string representation.

    Also adds bound variables with their de Bruijn indexes to
    bound_variable_indexes dictionary.

    Must be invoked for each node pre-order in a depth first traversal of an
    S-expression tree.

    Args:
      node: The current S-expression node.
      node_kind: The syntax element kind represented by the node.
      index_in_parent: The index of the current node in its parent node. None
        for the root node.
    """
    if hol_light_tree.is_abstraction_tree_node(node, node_kind):
      variable_id = _get_abstraction_variable_id(node)
      if variable_id not in bound_variable_indexes:
        bound_variable_indexes[variable_id] = []
      bound_variable_indexes[variable_id].append(max_indexes[node])
    if index_in_parent is not None and index_in_parent > 0:
      result_parts.append(' ')
    if not node.children:
      if (node_kind ==
          hol_light_sexpression_syntax.SyntaxElementKind.VARIABLE_NAME):
        variable_id = _get_variable_id(node.parent)
        if (variable_id in bound_variable_indexes and
            bound_variable_indexes[variable_id]):
          result_parts.append(str(bound_variable_indexes[variable_id][-1]))
          return
      result_parts.append(repr(node))
      return
    result_parts.append('(')

  def _exit_fn(node: sexpression_parser.SExpressionTreeNode,
               node_kind: hol_light_sexpression_syntax.SyntaxElementKind,
               index_in_parent: Optional[int]) -> None:
    """Adds the end of string representation of a node to result_parts.

    Adds the part of the string representation which succeeds children. If there
    are no children, adds nothing.

    Also removes bound variables with their de Bruijn indexes from
    bound_variable_indexes dictionary.

    Must be invoked for each node post-order in a depth first traversal of an
    S-expression tree.

    Args:
      node: The current S-expression node.
      node_kind: The syntax element kind represented by the node.
      index_in_parent: The index of the current node in its parent node. None
        for the root node.
    """
    del index_in_parent  # Unused.
    if hol_light_tree.is_abstraction_tree_node(node, node_kind):
      variable_id = _get_abstraction_variable_id(node)
      bound_variable_indexes[variable_id].pop()
    if not node.children:
      return
    result_parts.append(')')

  hol_light_tree.traverse_depth_first(_enter_fn, _exit_fn)
  return ''.join(result_parts)
