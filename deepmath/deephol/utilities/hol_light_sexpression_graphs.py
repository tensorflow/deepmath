# Lint as: python3
"""Provides syntax analysis functions for an S-expression graph.

The implementation of this library is HOL Light S-expression syntax specific.

S-expressions with <TYPE> atoms and with bound variable names replaced with
their de Bruijn indexes are also supported.
"""
from typing import Callable, Dict, FrozenSet, List, Optional, Text, Tuple

from deepmath.deephol.utilities import hol_light_sexpression_syntax
from deepmath.deephol.utilities import sexpression_graphs


class _TraversalState(object):

  def __init__(self, node_kind: hol_light_sexpression_syntax.SyntaxElementKind,
               parent: Optional[sexpression_graphs.NodeID]):
    self.node_kind = node_kind
    self.parent = parent
    self.child_index = -1


class HolLightSExpressionGraph(object):
  """Provides syntax analysis functions for the given S-expression graph.

  Caches the values which are expensive to compute.
  """

  def __init__(self, graph: sexpression_graphs.SExpressionGraph,
               has_type_atoms: bool):
    """Init function.

    Args:
      graph: The S-expression graph.
      has_type_atoms: Whether <TYPE> atoms have been inserted in the graph (see
        sexpression_type_atoms.py).
    """

    self._graph = graph
    self._syntax = hol_light_sexpression_syntax.HolLightSExpressionSyntax(
        has_type_atoms)
    self._has_type_variables_cache = {
    }  # type: Dict[Tuple[sexpression_graphs.NodeID, hol_light_sexpression_syntax.SyntaxElementKind], bool]
    self._free_variables_cache = {
    }  # type: Dict[Tuple[sexpression_graphs.NodeID, hol_light_sexpression_syntax.SyntaxElementKind], FrozenSet[sexpression_graphs.NodeID]]

  def _get_child_sexp_fn(
      self, node: sexpression_graphs.NodeID) -> Callable[[int], Text]:

    def _child_sexp_fn(child_index: int) -> Text:
      return self._graph.to_text(self._graph.children[node][child_index])

    return _child_sexp_fn

  def _get_child_count(self, node: sexpression_graphs.NodeID) -> int:
    return len(self._graph.children[node])

  def is_variable(
      self, node: sexpression_graphs.NodeID,
      node_kind: hol_light_sexpression_syntax.SyntaxElementKind) -> bool:
    """Check whether the given node represents a variable term.

    Args:
      node: The node to examine.
      node_kind: The kind of the node to examine (may also be UNKNOWN).

    Returns:
      Whether the node represents a variable term.
    """
    return self._syntax.is_variable(node_kind, self._get_child_count(node),
                                    self._get_child_sexp_fn(node))

  def is_abstraction(
      self, node: sexpression_graphs.NodeID,
      node_kind: hol_light_sexpression_syntax.SyntaxElementKind) -> bool:
    """Check whether the given node represents an abstraction term.

    Args:
      node: The node to examine.
      node_kind: The kind of the node to examine (may also be UNKNOWN).

    Returns:
      Whether the node represents an abstraction term.
    """
    return self._syntax.is_abstraction(node_kind, self._get_child_count(node),
                                       self._get_child_sexp_fn(node))

  def is_constant(
      self, node: sexpression_graphs.NodeID,
      node_kind: hol_light_sexpression_syntax.SyntaxElementKind) -> bool:
    """Check whether the given node represents a constant term.

    Args:
      node: The node to examine.
      node_kind: The kind of the node to examine (may also be UNKNOWN).

    Returns:
      Whether the node represents a constant term.
    """
    return self._syntax.is_constant(node_kind, self._get_child_count(node),
                                    self._get_child_sexp_fn(node))

  def is_combination(
      self, node: sexpression_graphs.NodeID,
      node_kind: hol_light_sexpression_syntax.SyntaxElementKind) -> bool:
    """Check whether the given node represents a combination term.

    Args:
      node: The node to examine.
      node_kind: The kind of the node to examine (may also be UNKNOWN).

    Returns:
      Whether the node represents a combination term.
    """
    return self._syntax.is_combination(node_kind, self._get_child_count(node),
                                       self._get_child_sexp_fn(node))

  def get_abstraction_variable(
      self,
      abstraction_node: sexpression_graphs.NodeID) -> sexpression_graphs.NodeID:
    """Returns the NodeID of the variable bound in the given abstraction.

    Args:
      abstraction_node: The abstraction node to examine. The node must represent
        an abstraction term.

    Returns:
      The NodeID of the variable bound in the given abstraction.
    """
    assert self.is_abstraction(
        abstraction_node, hol_light_sexpression_syntax.SyntaxElementKind.TERM)
    return self._graph.children[abstraction_node][
        self._syntax.abstraction_variable_child_index]

  def get_child(
      self, node: sexpression_graphs.NodeID,
      node_kind: hol_light_sexpression_syntax.SyntaxElementKind,
      child_index: int
  ) -> Tuple[sexpression_graphs.NodeID,
             hol_light_sexpression_syntax.SyntaxElementKind]:
    """Returns a child of the given node together with its kind.

    Args:
      node: The NodeID of the parent node.
      node_kind: The SyntaxElementKind of the parent node.
      child_index: The index of the child node.

    Returns:
      A tuple. The first element of the tuple is the NodeID of the child node.
      The second element of the tuple is the SyntaxElementKind of the child
      node.
    """
    child = self._graph.children[node][child_index]
    return (child,
            self._syntax.get_child_kind(node_kind, self._get_child_count(node),
                                        self._get_child_sexp_fn(node),
                                        child_index))

  def get_children(
      self, node: sexpression_graphs.NodeID,
      node_kind: hol_light_sexpression_syntax.SyntaxElementKind
  ) -> List[Tuple[sexpression_graphs.NodeID,
                  hol_light_sexpression_syntax.SyntaxElementKind]]:
    """Returns the children of the given node together with their kinds.

    Args:
      node: The NodeID of the parent node.
      node_kind: The SyntaxElementKind of the parent node.

    Returns:
      A list that contains a tuple for each child in order. The first element of
      the tuple is the NodeID of the child node. The second element of the tuple
      is the SyntaxElementKind of the child node.
    """
    children = self._graph.children[node]
    children_with_kinds = [
    ]  # type: List[Tuple[sexpression_graphs.NodeID, hol_light_sexpression_syntax.SyntaxElementKind]]
    for child_index, child in enumerate(children):
      children_with_kinds.append(
          (child,
           self._syntax.get_child_kind(node_kind, self._get_child_count(node),
                                       self._get_child_sexp_fn(node),
                                       child_index)))
    return children_with_kinds

  def has_type_variables(
      self, node: sexpression_graphs.NodeID,
      node_kind: hol_light_sexpression_syntax.SyntaxElementKind) -> bool:
    """Determines whether an S-expression has type variables.

    Uses a cache for the result. Does not use recursion.

    Args:
      node: The NodeID of the starting node of the subgraph representing the
        S-expression.
      node_kind: The SyntaxElementKind of the node.

    Returns:
      Whether the S-expression has type variables.
    """
    current_node = node  # type: Optional[sexpression_graphs.NodeID]
    state_stack = [_TraversalState(node_kind, parent=None)]
    result = False
    while state_stack:
      assert current_node is not None
      state = state_stack[-1]
      state.child_index = state.child_index + 1
      cache_key = (current_node, state.node_kind)

      # Already cached.
      if state.child_index == 0 and cache_key in self._has_type_variables_cache:
        if self._has_type_variables_cache[cache_key]:
          result = True
        current_node = state.parent
        state_stack.pop()
        continue

      children = self._graph.children[current_node]
      if (self._graph.is_leaf_node(current_node) and
          state.node_kind == hol_light_sexpression_syntax.SyntaxElementKind.TYPE
         ):
        result = True

      # Skip unnecessary traversal when result is already known to be True.
      if result:
        state.child_index = len(children)

      if state.child_index >= len(children):
        self._has_type_variables_cache[cache_key] = result
        current_node = state.parent
        state_stack.pop()
      else:
        state_stack.append(
            _TraversalState(
                self._syntax.get_child_kind(
                    state.node_kind, len(children),
                    self._get_child_sexp_fn(current_node), state.child_index),
                parent=current_node))
        current_node = children[state.child_index]
    return result

  def get_free_variables(
      self, node: sexpression_graphs.NodeID,
      node_kind: hol_light_sexpression_syntax.SyntaxElementKind
  ) -> FrozenSet[sexpression_graphs.NodeID]:
    """Finds all the free variables in an S-expression.

    Uses a cache for the result. Does not use recursion.

    Args:
      node: The NodeID of the starting node of the subgraph representing the
        S-expression.
      node_kind: The SyntaxElementKind of the node.

    Returns:
      A frozen set of IDs of all the nodes representing variables which are used
      in the S-expression and are not bound in it.
    """
    current_node = node  # type: Optional[sexpression_graphs.NodeID]
    state_stack = [_TraversalState(node_kind, parent=None)]
    while state_stack:
      assert current_node is not None
      state = state_stack[-1]
      state.child_index = state.child_index + 1
      cache_key = (current_node, state.node_kind)

      # Already cached.
      if state.child_index == 0 and cache_key in self._free_variables_cache:
        current_node = state.parent
        state_stack.pop()
        continue

      children = self._graph.children[current_node]
      if state.child_index >= len(children):
        # Compute for the current node using cached results for child nodes.
        free_variables = set()
        if self.is_variable(current_node, state.node_kind):
          free_variables.add(current_node)
        else:
          for child_node, child_node_kind in self.get_children(
              current_node, state.node_kind):
            free_variables.update(self._free_variables_cache[(child_node,
                                                              child_node_kind)])
          if self.is_abstraction(current_node, state.node_kind):
            free_variables.discard(self.get_abstraction_variable(current_node))

        self._free_variables_cache[cache_key] = frozenset(free_variables)
        current_node = state.parent
        state_stack.pop()
      else:
        state_stack.append(
            _TraversalState(
                self._syntax.get_child_kind(
                    state.node_kind, len(children),
                    self._get_child_sexp_fn(current_node), state.child_index),
                parent=current_node))
        current_node = children[state.child_index]
    return self._free_variables_cache[(node, node_kind)]
