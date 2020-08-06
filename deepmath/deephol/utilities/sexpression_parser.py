"""A minimal SExpression parser for terms, goals, and theorems from HOL Light.

Assumes SExpressions of the form '(word1 word1 (word1) () (() ()))'.
That is, spaces and parentheses are treated as separators, bare words are
accepted as SExpressions, and nodes can have 0 children. The expression above
has 5 children: 'word1', 'word1', '(word1)', '()', and '(() ())'. The order of
children is respected.
"""
from typing import List, Optional, Text


class SExpParseError(Exception):
  pass


def validate_parens(sexp: Text):
  """Counts the opening and closing parentheses."""
  if sexp[0] != '(' or sexp[-1] != ')':
    raise SExpParseError(
        'SExpressions must start and end with parentheses: %s' % sexp)
  parenthesis_counter = 0
  for idx, c in enumerate(sexp):
    if c == '(':
      parenthesis_counter += 1
    elif c == ')':
      parenthesis_counter -= 1
    if parenthesis_counter <= 0 and idx != len(sexp) - 1:
      raise SExpParseError(
          'Closing parenthesis before end of expression at pos %d' % idx)
  if parenthesis_counter > 0:
    raise SExpParseError(
        'Expression of length %d not closed; not enough closing parentheses.' %
        len(sexp))


def is_bare_word(sexp: Text):
  """Base case of SExpressions."""
  for c in sexp:
    if c in [' ', '(', ')']:
      return False
  return True


class SExpressionTreeNode(object):
  """A node in a representation of an S-expression as a tree.

  Attributes:
    sexp: The whole S-expression string.
    start: The start index of the current node in the whole S-expression string.
    end: The end index (not inclusive) of the current node in the whole
      S-expression string.
    parent: The parent node. None for the root node.
    children: The list of child nodes.
  """

  def __init__(self, sexp: Text, start: int, end: int,
               parent: Optional['SExpressionTreeNode'],
               children: List['SExpressionTreeNode']):
    self.sexp = sexp
    self.start = start
    self.end = end
    self.parent = parent
    self.children = children

  def __repr__(self):
    return self.sexp[self.start:self.end]


def to_tree(sexp: Text) -> SExpressionTreeNode:
  """Transforms an S-expression into its tree representation.

  Creates no new multi-character strings in the process. Does not use recursion.
  Has linear complexity.

  Args:
    sexp: An S-expression string.

  Returns:
    The root node of the S-expression tree.

  Raises:
    SExpParseError: Parsing error.
  """

  def _skip_spaces(pos: int) -> int:
    while pos < len(sexp) and sexp[pos] == ' ':
      pos = pos + 1
    return pos

  def _skip_bare_word(pos: int) -> int:
    while pos < len(sexp) and sexp[pos] not in [' ', '(', ')']:
      pos = pos + 1
    return pos

  def _validate_pos(pos: int, is_inside_parenthesis: bool) -> None:
    if pos >= len(sexp):
      if is_inside_parenthesis:
        raise SExpParseError(
            'Expression of length %d not closed; not enough closing parentheses.'
            % len(sexp))
      raise SExpParseError('Empty S-expression.')

  def _new_node(start: int,
                parent: Optional[SExpressionTreeNode]) -> SExpressionTreeNode:
    # The end and children will be corrected later.
    node = SExpressionTreeNode(
        sexp, start=start, end=len(sexp), parent=parent, children=[])
    if parent is not None:
      parent.children.append(node)
    return node

  def _parse_node_continuation(node: SExpressionTreeNode,
                               pos: int) -> Optional[SExpressionTreeNode]:
    """Continues parsing of a newly created node at the given position.

    Args:
      node: The newly created node after some children may have been added.
      pos: The next position in the S-expression to be parsed.

    Returns:
      The next node to be parsed, otherwise None.

    Raises:
      SExpParseError: Parsing error.
    """
    pos = _skip_spaces(pos)
    _validate_pos(pos, is_inside_parenthesis=True)
    if sexp[pos] == ')':
      node.end = pos + 1
      return node.parent
    return _new_node(start=pos, parent=node)

  def _parse_bare_word_node(
      node: SExpressionTreeNode) -> Optional[SExpressionTreeNode]:
    """Parses a newly created node containing just a bare word.

    Args:
      node: The newly created node.

    Returns:
      The next node to be parsed, otherwise None.

    Raises:
      SExpParseError: Parsing error.
    """
    node.end = _skip_bare_word(node.start)
    return node.parent

  def _parse_node_start(
      node: SExpressionTreeNode) -> Optional[SExpressionTreeNode]:
    """Parses the start of a newly created node.

    Args:
      node: The newly created node.

    Returns:
      The next node to be parsed, otherwise None.

    Raises:
      SExpParseError: Parsing error.
    """
    pos = node.start
    _validate_pos(pos, is_inside_parenthesis=node.parent is not None)
    assert sexp[pos] != ' '  # Leading spaces must have been skipped.
    if sexp[pos] == ')':
      raise SExpParseError(
          'Closing parenthesis before end of expression at pos %d.' % pos)
    if sexp[pos] == '(':
      return _parse_node_continuation(node, pos + 1)
    return _parse_bare_word_node(node)

  def _parse_node(node: SExpressionTreeNode) -> Optional[SExpressionTreeNode]:
    """Parses a newly created node.

    Args:
      node: The newly created node.

    Returns:
      The next node to be parsed, otherwise None.

    Raises:
      SExpParseError: Parsing error.
    """
    if node.children:
      return _parse_node_continuation(node, node.children[-1].end)
    return _parse_node_start(node)

  root = _new_node(start=_skip_spaces(0), parent=None)
  node = root  # type: Optional[SExpressionTreeNode]

  while node is not None:
    node = _parse_node(node)

  if _skip_spaces(root.end) != len(sexp):
    raise SExpParseError(
        'Extra characters after the end of expression at pos %d' % root.end)
  return root
