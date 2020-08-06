# Lint as: python3
"""Provides syntax-dependent functions for an S-expression tree.

S-expression trees are produced by the sexpression_parser module.

The implementation of this library is HOL Light S-expression syntax specific.

S-expressions with <TYPE> atoms and with bound variable names replaced with
their de Bruijn indexes are also supported.
"""
from typing import Callable, Optional, Text
from deepmath.deephol.utilities import hol_light_sexpression_syntax
from deepmath.deephol.utilities import sexpression_parser


class _TraversalState(object):
  """A state of a single tree node in a scope of a depth-first tree traversal.

  Attributes:
    node_kind: The kind of syntax element represented by the tree node.
    index_in_parent: The index of the tree node in its parent. None for the root
      node.
    child_index: The index of the current child node being traversed. Stores -1
      before any child is traversed. Stores the count of children after all the
      children has been traversed.
  """

  def __init__(self, node_kind: hol_light_sexpression_syntax.SyntaxElementKind,
               index_in_parent: Optional[int]):
    self.node_kind = node_kind
    self.index_in_parent = index_in_parent
    self.child_index = -1


def _get_child_sexp_fn(
    node: sexpression_parser.SExpressionTreeNode
) -> hol_light_sexpression_syntax.ChildSexpFn:
  """Returns ChildSexpFn for the given tree node.

  Args:
    node: The parent tree node.

  Returns:
    A function that returns the text of a child S-expression of the parent tree
    node by the index of the child. Follows the signature:
      * Args:
        * `child_index`: The index of the child S-expression to return.
      * Returns: The text of the child S-expression.
  """

  def _child_sexp_fn(child_index: int) -> Text:
    return repr(node.children[child_index])

  return _child_sexp_fn


# A function that is called for each tree node being traversed.
# Follows the signature:
#   * Args:
#     * `node`: The current tree node.
#     * `node_kind`: The kind of the syntax element represented by the node.
#     * `index_in_parent`: The index of the current tree node in its parent.
#       None for the top node being traversed.
TraversalFn = Callable[[
    sexpression_parser.SExpressionTreeNode, hol_light_sexpression_syntax
    .SyntaxElementKind, Optional[int]
], None]


class HolLightSExpressionTree(object):
  """Provides syntax analysis functions for the given S-expression tree.

  S-expression trees are produced by the sexpression_parser module.

  The implementation of this class is HOL Light S-expression syntax specific.

  S-expressions with <TYPE> atoms and with bound variable names replaced with
  their de Bruijn indexes are also supported.
  """

  def __init__(self, tree: sexpression_parser.SExpressionTreeNode,
               kind: hol_light_sexpression_syntax.SyntaxElementKind,
               has_type_atoms: bool):
    """Init function.

    Args:
      tree: The root node of an S-expression tree.
      kind: The syntax element kind represented by the S-expression.
      has_type_atoms: Whether <TYPE> atoms have been inserted in the
        S-expression (see sexpression_type_atoms.py).
    """
    assert tree.parent is None
    self._tree = tree
    self._kind = kind
    self._syntax = hol_light_sexpression_syntax.HolLightSExpressionSyntax(
        has_type_atoms)

  def is_variable_tree_node(
      self, node: sexpression_parser.SExpressionTreeNode,
      node_kind: hol_light_sexpression_syntax.SyntaxElementKind) -> bool:
    """Check whether the given S-expression represents a variable term.

    Args:
      node: The S-expression tree node.
      node_kind: The kind of the syntax element represented by the S-expression
        (may also be UNKNOWN).

    Returns:
      Whether the S-expression represents a variable term.
    """
    return self._syntax.is_variable(node_kind, len(node.children),
                                    _get_child_sexp_fn(node))

  def is_abstraction_tree_node(
      self, node: sexpression_parser.SExpressionTreeNode,
      node_kind: hol_light_sexpression_syntax.SyntaxElementKind) -> bool:
    """Check whether the given S-expression represents an abstraction term.

    Args:
      node: The S-expression tree node.
      node_kind: The kind of the syntax element represented by the S-expression
        (may also be UNKNOWN).

    Returns:
      Whether the S-expression represents an abstraction term.
    """
    return self._syntax.is_abstraction(node_kind, len(node.children),
                                       _get_child_sexp_fn(node))

  def is_combination_tree_node(
      self, node: sexpression_parser.SExpressionTreeNode,
      node_kind: hol_light_sexpression_syntax.SyntaxElementKind) -> bool:
    """Check whether the given S-expression represents a combination term.

    Args:
      node: The S-expression tree node.
      node_kind: The kind of the syntax element represented by the S-expression
        (may also be UNKNOWN).

    Returns:
      Whether the S-expression represents a combination term.
    """
    return self._syntax.is_combination(node_kind, len(node.children),
                                       _get_child_sexp_fn(node))

  def is_constant_tree_node(
      self, node: sexpression_parser.SExpressionTreeNode,
      node_kind: hol_light_sexpression_syntax.SyntaxElementKind) -> bool:
    """Check whether the given S-expression represents a constant term.

    Args:
      node: The S-expression tree node.
      node_kind: The kind of the syntax element represented by the S-expression
        (may also be UNKNOWN).

    Returns:
      Whether the S-expression represents a constant term.
    """
    return self._syntax.is_constant(node_kind, len(node.children),
                                    _get_child_sexp_fn(node))

  def get_abstraction_variable_tree_node(
      self, abstraction_node: sexpression_parser.SExpressionTreeNode
  ) -> sexpression_parser.SExpressionTreeNode:
    """Finds the variable bound in the given abstraction.

    Args:
      abstraction_node: The S-expression tree node representing the abstraction.

    Returns:
      The child S-expression tree node representing the variable.
    """
    assert self.is_abstraction_tree_node(
        abstraction_node, hol_light_sexpression_syntax.SyntaxElementKind.TERM)
    return abstraction_node.children[
        self._syntax.abstraction_variable_child_index]

  def get_abstraction_body_tree_node(
      self, abstraction_node: sexpression_parser.SExpressionTreeNode
  ) -> sexpression_parser.SExpressionTreeNode:
    """Finds the body of the given abstraction.

    Args:
      abstraction_node: The S-expression tree node representing the abstraction.

    Returns:
      The child S-expression tree node representing the body of the abstraction.
    """
    assert self.is_abstraction_tree_node(
        abstraction_node, hol_light_sexpression_syntax.SyntaxElementKind.TERM)
    return abstraction_node.children[self._syntax.abstraction_body_child_index]

  def get_combination_function_tree_node(
      self, combination_node: sexpression_parser.SExpressionTreeNode
  ) -> sexpression_parser.SExpressionTreeNode:
    """Finds the function applied in the given combination.

    Args:
      combination_node: The S-expression tree node representing the combination.

    Returns:
      The child S-expression tree node representing the function.
    """
    assert self.is_combination_tree_node(
        combination_node, hol_light_sexpression_syntax.SyntaxElementKind.TERM)
    return combination_node.children[
        self._syntax.combination_function_child_index]

  def get_combination_argument_tree_node(
      self, combination_node: sexpression_parser.SExpressionTreeNode
  ) -> sexpression_parser.SExpressionTreeNode:
    """Finds the argument of the given combination.

    Args:
      combination_node: The S-expression tree node representing the combination.

    Returns:
      The child S-expression tree node representing the argument.
    """
    assert self.is_combination_tree_node(
        combination_node, hol_light_sexpression_syntax.SyntaxElementKind.TERM)
    return combination_node.children[
        self._syntax.combination_argument_child_index]

  def get_variable_type_tree_node(
      self, variable_node: sexpression_parser.SExpressionTreeNode
  ) -> sexpression_parser.SExpressionTreeNode:
    """Finds the type of the given variable.

    Args:
      variable_node: The S-expression tree node representing the variable.

    Returns:
      The child S-expression tree node representing the type.
    """
    assert self.is_variable_tree_node(
        variable_node, hol_light_sexpression_syntax.SyntaxElementKind.TERM)
    return variable_node.children[self._syntax.variable_type_child_index]

  def get_variable_name_tree_node(
      self, variable_node: sexpression_parser.SExpressionTreeNode
  ) -> sexpression_parser.SExpressionTreeNode:
    """Finds the name of the given variable.

    Args:
      variable_node: The S-expression tree node representing the variable.

    Returns:
      The child S-expression tree node representing the name.
    """
    assert self.is_variable_tree_node(
        variable_node, hol_light_sexpression_syntax.SyntaxElementKind.TERM)
    return variable_node.children[self._syntax.variable_name_child_index]

  def get_constant_type_tree_node(
      self, constant_node: sexpression_parser.SExpressionTreeNode
  ) -> sexpression_parser.SExpressionTreeNode:
    """Finds the type of the given constant term.

    Args:
      constant_node: The S-expression tree node representing the constant term.

    Returns:
      The child S-expression tree node representing the type.
    """
    assert self.is_constant_tree_node(
        constant_node, hol_light_sexpression_syntax.SyntaxElementKind.TERM)
    return constant_node.children[self._syntax.constant_type_child_index]

  def get_constant_name_tree_node(
      self, constant_node: sexpression_parser.SExpressionTreeNode
  ) -> sexpression_parser.SExpressionTreeNode:
    """Finds the name of the given constant term.

    Args:
      constant_node: The S-expression tree node representing the constant term.

    Returns:
      The child S-expression tree node representing the name.
    """
    assert self.is_constant_tree_node(
        constant_node, hol_light_sexpression_syntax.SyntaxElementKind.TERM)
    return constant_node.children[self._syntax.constant_name_child_index]

  def traverse_subtree_depth_first(
      self, top_node: sexpression_parser.SExpressionTreeNode,
      top_node_kind: hol_light_sexpression_syntax.SyntaxElementKind,
      enter_fn: Optional[TraversalFn], exit_fn: Optional[TraversalFn]) -> None:
    """Traverses the given S-expression subtree depth-first.

    At least one of enter_fn and exit_fn must be specified.

    Args:
      top_node: The top node of the subtree to traverse.
      top_node_kind: The kind of the syntax element represented by the top node.
      enter_fn: An optional function invoked pre-order for each tree node being
        traversed. Follows the signature:
          * Args:
            * `node`: The current tree node.
            * `node_kind`: The kind of the syntax element represented by the
              node.
            * `index_in_parent`: The index of the current tree node in its
              parent. None for the top node.
      exit_fn: An optional function invoked post-order for each tree node being
        traversed. Follows the signature:
          * Args:
            * `node`: The current tree node.
            * `node_kind`: The kind of the syntax element represented by the
              node.
            * `index_in_parent`: The index of the current tree node in its
              parent. None for the top node.
    """
    assert enter_fn is not None or exit_fn is not None
    node = top_node  # type: Optional[sexpression_parser.SExpressionTreeNode]
    state_stack = [_TraversalState(top_node_kind, index_in_parent=None)]
    while state_stack:
      assert node is not None
      state = state_stack[-1]
      state.child_index = state.child_index + 1
      if state.child_index == 0 and enter_fn is not None:
        enter_fn(node, state.node_kind, state.index_in_parent)
      if state.child_index >= len(node.children):
        if exit_fn is not None:
          exit_fn(node, state.node_kind, state.index_in_parent)
        node = node.parent
        state_stack.pop()
      else:
        state_stack.append(
            _TraversalState(
                self._syntax.get_child_kind(state.node_kind, len(node.children),
                                            _get_child_sexp_fn(node),
                                            state.child_index),
                index_in_parent=state.child_index))
        node = node.children[state.child_index]

  def traverse_depth_first(self, enter_fn: Optional[TraversalFn],
                           exit_fn: Optional[TraversalFn]) -> None:
    """Traverses the S-expression tree depth-first.

    At least one of enter_fn and exit_fn must be specified.

    Args:
      enter_fn: An optional function invoked pre-order for each tree node being
        traversed. Follows the signature:
          * Args:
            * `node`: The current tree node.
            * `node_kind`: The kind of the syntax element represented by the
              node.
            * `index_in_parent`: The index of the current tree node in its
              parent. None for the root node.
      exit_fn: An optional function invoked post-order for each tree node being
        traversed. Follows the signature:
          * Args:
            * `node`: The current tree node.
            * `node_kind`: The kind of the syntax element represented by the
              node.
            * `index_in_parent`: The index of the current tree node in its
              parent. None for the root node.
    """
    self.traverse_subtree_depth_first(self._tree, self._kind, enter_fn, exit_fn)
