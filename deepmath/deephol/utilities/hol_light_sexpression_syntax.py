# Lint as: python3
"""Provides syntax analysis functions for S-expressions in any format.

The implementation of this library is HOL Light S-expression syntax specific.

S-expressions with <TYPE> atoms and with bound variable names replaced with
their de Bruijn indexes are also supported.
"""
import enum
from typing import Callable, Text


class SyntaxElementKind(enum.Enum):
  """The kind of syntax element represented by an S-expression or subexpression.

  Values:
    UNKNOWN: The exact kind of syntax element is not known or not covered with
      other enum values. In case of an ambiguity between terms and types
      it is assumed that the element is a term.
    TERM: The element represents a HOL Light term.
    TYPE: The element represents a HOL Light type.
    VARIABLE_NAME: The element represents the name of a variable.
    CONSTANT_NAME: The element represents the name of a constant.
    TYPE_CONSTRUCTOR_NAME: The element represents the name of a type
      constructor.
    SEXPRESSION_KIND: The element represents the kind of the parent S-expression
      node (whether it is an abstraction, combination, variable or constant).
    TYPE_ATOM: The element is the <TYPE> atom.
  """
  UNKNOWN = 0
  TERM = 1
  TYPE = 2
  VARIABLE_NAME = 3
  CONSTANT_NAME = 4
  TYPE_CONSTRUCTOR_NAME = 5
  SEXPRESSION_KIND = 6
  TYPE_ATOM = 7


_TYPE_ATOM_CHILD_INDEX = 0
_TYPE_CONSTRUCTOR_CHILD_INDEX_WITHOUT_TYPE_ATOM = 0
_TYPE_CONSTRUCTOR_CHILD_INDEX_WITH_TYPE_ATOM = 1
_TERM_CHILD_COUNT = 3
_TERM_SEXPRESSION_KIND_CHILD_INDEX = 0
_CONSTANT_SEXPRESSION_KIND_LABEL = 'c'
_CONSTANT_TYPE_CHILD_INDEX = 1
_CONSTANT_NAME_CHILD_INDEX = 2
_VARIABLE_SEXPRESSION_KIND_LABEL = 'v'
_VARIABLE_TYPE_CHILD_INDEX = 1
_VARIABLE_NAME_CHILD_INDEX = 2
_ABSTRACTION_SEXPRESSION_KIND_LABEL = 'l'
_ABSTRACTION_VARIABLE_CHILD_INDEX = 1
_ABSTRACTION_BODY_CHILD_INDEX = 2
_COMBINATION_SEXPRESSION_KIND_LABEL = 'a'
_COMBINATION_FUNCTION_CHILD_INDEX = 1
_COMBINATION_ARGUMENT_CHILD_INDEX = 2

# The text used to represent type atoms. Type atoms may be inserted as the first
# child of each type application so that every composite S-expression has its
# kind specified in the first child (see sexpression_type_atoms.py).
TYPE_ATOM = '<TYPE>'

# A function that returns the text of a child S-expression by its index.
# Follows the signature:
#   * Args:
#     * `child_index`: The index of the child S-expression to return.
#   * Returns:
#     * The text of the child S-expression.
ChildSexpFn = Callable[[int], Text]


class HolLightSExpressionSyntax(object):
  """Does syntax analysis for S-expressions in any format.

  The implementation of this class is HOL Light S-expression syntax specific.

  S-expressions with <TYPE> atoms and with bound variable names replaced with
  their de Bruijn indexes are also supported.
  """

  def __init__(self, has_type_atoms: bool):
    """Init function.

    Args:
      has_type_atoms: Whether <TYPE> atoms have been inserted in the
        S-expressions (see sexpression_type_atoms.py).
    """

    self._has_type_atoms = has_type_atoms

  def _is_term_with_sexpression_kind_label(self, kind: SyntaxElementKind,
                                           child_count: int,
                                           get_child_sexp_fn: ChildSexpFn,
                                           label: Text) -> bool:
    """Check whether the given S-expression represents a term of the given kind.

    Args:
      kind: The kind of the syntax element represented by the S-expression (may
        also be UNKNOWN).
      child_count: The count of child S-expressions.
      get_child_sexp_fn: A function that returns the text of a child
        S-expression by its index. Follows the signature:
        * Args:
          * `child_index`: The index of the child S-expression to return.
        * Returns: The text of the child S-expression.
      label: The label used in HOL Light S-expressions for the terms of that
        kind.

    Returns:
      Whether the S-expression represents a term of the kind which is
      represented by the given label.
    """
    if (kind != SyntaxElementKind.TERM and kind != SyntaxElementKind.UNKNOWN):
      return False
    if child_count != _TERM_CHILD_COUNT:
      return False
    sexpression_kind_label = get_child_sexp_fn(
        _TERM_SEXPRESSION_KIND_CHILD_INDEX)
    return sexpression_kind_label == label

  def is_variable(self, kind: SyntaxElementKind, child_count: int,
                  get_child_sexp_fn: ChildSexpFn) -> bool:
    """Check whether the given S-expression represents a variable term.

    Args:
      kind: The kind of the syntax element represented by the S-expression (may
        also be UNKNOWN).
      child_count: The count of child S-expressions.
      get_child_sexp_fn: A function that returns the text of a child
        S-expression by its index. Follows the signature:
        * Args:
          * `child_index`: The index of the child S-expression to return.
        * Returns: The text of the child S-expression.

    Returns:
      Whether the S-expression represents a variable term.
    """
    return self._is_term_with_sexpression_kind_label(
        kind, child_count, get_child_sexp_fn, _VARIABLE_SEXPRESSION_KIND_LABEL)

  def is_abstraction(self, kind: SyntaxElementKind, child_count: int,
                     get_child_sexp_fn: ChildSexpFn) -> bool:
    """Check whether the given S-expression represents an abstraction term.

    Args:
      kind: The kind of the syntax element represented by the S-expression (may
        also be UNKNOWN).
      child_count: The count of child S-expressions.
      get_child_sexp_fn: A function that returns the text of a child
        S-expression by its index. Follows the signature:
        * Args:
          * `child_index`: The index of the child S-expression to return.
        * Returns: The text of the child S-expression.

    Returns:
      Whether the S-expression represents an abstraction term.
    """
    return self._is_term_with_sexpression_kind_label(
        kind, child_count, get_child_sexp_fn,
        _ABSTRACTION_SEXPRESSION_KIND_LABEL)

  def is_constant(self, kind: SyntaxElementKind, child_count: int,
                  get_child_sexp_fn: ChildSexpFn) -> bool:
    """Check whether the given S-expression represents a constant term.

    Args:
      kind: The kind of the syntax element represented by the S-expression (may
        also be UNKNOWN).
      child_count: The count of child S-expressions.
      get_child_sexp_fn: A function that returns the text of a child
        S-expression by its index. Follows the signature:
        * Args:
          * `child_index`: The index of the child S-expression to return.
        * Returns: The text of the child S-expression.

    Returns:
      Whether the S-expression represents a constant term.
    """
    return self._is_term_with_sexpression_kind_label(
        kind, child_count, get_child_sexp_fn, _CONSTANT_SEXPRESSION_KIND_LABEL)

  def is_combination(self, kind: SyntaxElementKind, child_count: int,
                     get_child_sexp_fn: ChildSexpFn) -> bool:
    """Check whether the given S-expression represents a combination term.

    Args:
      kind: The kind of the syntax element represented by the S-expression (may
        also be UNKNOWN).
      child_count: The count of child S-expressions.
      get_child_sexp_fn: A function that returns the text of a child
        S-expression by its index. Follows the signature:
        * Args:
          * `child_index`: The index of the child S-expression to return.
        * Returns: The text of the child S-expression.

    Returns:
      Whether the S-expression represents a combination term.
    """
    return self._is_term_with_sexpression_kind_label(
        kind, child_count, get_child_sexp_fn,
        _COMBINATION_SEXPRESSION_KIND_LABEL)

  def _get_type_child_kind(self, child_index: int) -> SyntaxElementKind:
    """Determines the SyntaxElementKind of a child S-expression of a type.

    Args:
      child_index: The index of the child S-expression (0-based).

    Returns:
      The SyntaxElementKind of the child S-expression.
    """
    if self._has_type_atoms:
      if child_index == _TYPE_ATOM_CHILD_INDEX:
        return SyntaxElementKind.TYPE_ATOM
      if child_index == _TYPE_CONSTRUCTOR_CHILD_INDEX_WITH_TYPE_ATOM:
        return SyntaxElementKind.TYPE_CONSTRUCTOR_NAME
      return SyntaxElementKind.TYPE
    if child_index == _TYPE_CONSTRUCTOR_CHILD_INDEX_WITHOUT_TYPE_ATOM:
      return SyntaxElementKind.TYPE_CONSTRUCTOR_NAME
    return SyntaxElementKind.TYPE

  def _get_constant_child_kind(self, child_index: int) -> SyntaxElementKind:
    """Determines the SyntaxElementKind of a child S-expression of a constant.

    Args:
      child_index: The index of the child S-expression (0-based).

    Returns:
      The SyntaxElementKind of the child S-expression.
    """
    if child_index == _TERM_SEXPRESSION_KIND_CHILD_INDEX:
      return SyntaxElementKind.SEXPRESSION_KIND
    if child_index == _CONSTANT_TYPE_CHILD_INDEX:
      return SyntaxElementKind.TYPE
    if child_index == _CONSTANT_NAME_CHILD_INDEX:
      return SyntaxElementKind.CONSTANT_NAME
    raise ValueError('invalid child index')

  def _get_variable_child_kind(self, child_index: int) -> SyntaxElementKind:
    """Determines the SyntaxElementKind of a child S-expression of a varible.

    Args:
      child_index: The index of the child S-expression (0-based).

    Returns:
      The SyntaxElementKind of the child S-expression.
    """
    if child_index == _TERM_SEXPRESSION_KIND_CHILD_INDEX:
      return SyntaxElementKind.SEXPRESSION_KIND
    if child_index == _VARIABLE_TYPE_CHILD_INDEX:
      return SyntaxElementKind.TYPE
    if child_index == _VARIABLE_NAME_CHILD_INDEX:
      return SyntaxElementKind.VARIABLE_NAME
    raise ValueError('invalid child node index')

  def _get_abstraction_child_kind(self, child_index: int) -> SyntaxElementKind:
    """Determines the SyntaxElementKind of a child of an abstraction.

    Args:
      child_index: The index of the child S-expression (0-based).

    Returns:
      The SyntaxElementKind of the child S-expression.
    """
    if child_index == _TERM_SEXPRESSION_KIND_CHILD_INDEX:
      return SyntaxElementKind.SEXPRESSION_KIND
    if child_index == _ABSTRACTION_VARIABLE_CHILD_INDEX:
      return SyntaxElementKind.TERM
    if child_index == _ABSTRACTION_BODY_CHILD_INDEX:
      return SyntaxElementKind.TERM
    raise ValueError('invalid child node index')

  def _get_combination_child_kind(self, child_index: int) -> SyntaxElementKind:
    """Determines the SyntaxElementKind of a child of a combination.

    Args:
      child_index: The index of the child S-expression (0-based).

    Returns:
      The SyntaxElementKind of the child S-expression.
    """
    if child_index == _TERM_SEXPRESSION_KIND_CHILD_INDEX:
      return SyntaxElementKind.SEXPRESSION_KIND
    if child_index == _COMBINATION_FUNCTION_CHILD_INDEX:
      return SyntaxElementKind.TERM
    if child_index == _COMBINATION_ARGUMENT_CHILD_INDEX:
      return SyntaxElementKind.TERM
    raise ValueError('invalid child node index')

  def get_child_kind(self, kind: SyntaxElementKind, child_count: int,
                     get_child_sexp_fn: ChildSexpFn,
                     child_index: int) -> SyntaxElementKind:
    """Determines the SyntaxElementKind of a child S-expression.

    Args:
      kind: The kind of the syntax element represented by the parent
        S-expression (may also be UNKNOWN).
      child_count: The count of child S-expressions in the parent S-expression.
      get_child_sexp_fn: A function that returns the text of a child
        S-expression by its index. Follows the signature:
        * Args:
          * `child_index`: The index of the child S-expression to return.
        * Returns: The text of the child S-expression.
      child_index: The index of the child S-expression (0-based).

    Returns:
      The SyntaxElementKind of the child node.
    """
    if child_index >= child_count:
      raise ValueError('invalid child node index')
    if kind == SyntaxElementKind.TYPE:
      return self._get_type_child_kind(child_index)
    if kind != SyntaxElementKind.TERM and kind != SyntaxElementKind.UNKNOWN:
      return SyntaxElementKind.UNKNOWN
    if child_count == _TERM_CHILD_COUNT:
      sexpression_kind_label = get_child_sexp_fn(
          _TERM_SEXPRESSION_KIND_CHILD_INDEX)
      if sexpression_kind_label == _CONSTANT_SEXPRESSION_KIND_LABEL:
        return self._get_constant_child_kind(child_index)
      if sexpression_kind_label == _VARIABLE_SEXPRESSION_KIND_LABEL:
        return self._get_variable_child_kind(child_index)
      if sexpression_kind_label == _ABSTRACTION_SEXPRESSION_KIND_LABEL:
        return self._get_abstraction_child_kind(child_index)
      if sexpression_kind_label == _COMBINATION_SEXPRESSION_KIND_LABEL:
        return self._get_combination_child_kind(child_index)
    if (self._has_type_atoms and child_count > _TYPE_ATOM_CHILD_INDEX and
        get_child_sexp_fn(_TYPE_ATOM_CHILD_INDEX) == TYPE_ATOM):
      return self._get_type_child_kind(child_index)
    return SyntaxElementKind.UNKNOWN

  @property
  def abstraction_variable_child_index(self) -> int:
    """The index of the child representing the variable of an abstraction."""
    return _ABSTRACTION_VARIABLE_CHILD_INDEX

  @property
  def abstraction_body_child_index(self) -> int:
    """The index of the child representing the body of an abstraction."""
    return _ABSTRACTION_BODY_CHILD_INDEX

  @property
  def combination_function_child_index(self) -> int:
    """The index of the child representing the function of a combination."""
    return _COMBINATION_FUNCTION_CHILD_INDEX

  @property
  def combination_argument_child_index(self) -> int:
    """The index of the child representing the argument of a combination."""
    return _COMBINATION_ARGUMENT_CHILD_INDEX

  @property
  def variable_type_child_index(self) -> int:
    """The index of the child representing the type of a variable."""
    return _VARIABLE_TYPE_CHILD_INDEX

  @property
  def variable_name_child_index(self) -> int:
    """The index of the child representing the name of a variable."""
    return _VARIABLE_NAME_CHILD_INDEX

  @property
  def constant_type_child_index(self) -> int:
    """The index of the child representing the type of a constant."""
    return _CONSTANT_TYPE_CHILD_INDEX

  @property
  def constant_name_child_index(self) -> int:
    """The index of the child representing the name of a constant."""
    return _CONSTANT_NAME_CHILD_INDEX
