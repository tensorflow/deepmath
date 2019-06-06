"""A minimal SExpression parser for terms, goals, and theorems from HOL Light.

Assumes SExpressions of the form '(word1 word1 (word1) () (() ()))'.
That is, spaces and parantheses are treated as separators, bare words are
accepted as SExpressions, and nodes can have 0 children. The expression above
has 5 children: 'word1', 'word1', '(word1)', '()', and '(() ())'. The order of
children is respected.
"""

from __future__ import absolute_import
from __future__ import division
# Import Type Annotations
from __future__ import print_function

from typing import List, Optional, Text


class SExpParseError(Exception):
  pass


def is_start_of_word(sexp: Text, pos: int) -> bool:
  if pos < 0 or pos > len(sexp):
    raise SExpParseError('Position %d out of bounds of string %s.' %
                         (pos, sexp))
  return pos == 0 or sexp[pos - 1] in [' ', '('] and sexp[pos] not in [' ', ')']


def is_end_of_word(sexp: Text, pos: int) -> int:
  if pos < 0 or pos > len(sexp):
    raise SExpParseError('Position %d out of bounds of string %s.' %
                         (pos, sexp))
  return pos == len(sexp) or sexp[pos] in [' ', ')']


def end_of_word(sexp: Text, start: int) -> Optional[int]:
  """Returns the end of the bare word starting at start."""
  if not is_start_of_word(sexp, start):
    raise SExpParseError('end_of_word called in the middle of a word pos %d.' %
                         start)
  if is_end_of_word(sexp, start):
    raise SExpParseError('Beginning and end of word coincide at pos %d.' %
                         start)
  for pos in range(start, len(sexp) + 1):
    if is_end_of_word(sexp, pos):
      return pos
  return None


def end_of_child(sexp: Text, start: int) -> int:
  """Returns the index of the end of the word + 1."""
  if not is_start_of_word(sexp, start):
    raise SExpParseError(
        'end_of_child must be called at begginning of a word (pos %d)' % start)
  if sexp[start] == '(':
    parenthesis_counter = 0
    for idx, c in enumerate(sexp[start:]):
      if c == '(':
        parenthesis_counter += 1
      elif c == ')':
        parenthesis_counter -= 1
      if parenthesis_counter == 0:
        return start + idx + 1
  else:
    return end_of_word(sexp, start)  # pytype: disable=bad-return-type


def validate_parens(sexp: Text):
  """Counts the opening and closing parantheses."""
  if sexp[0] != '(' or sexp[-1] != ')':
    raise SExpParseError(
        'SExpressions must start and end with parantheses: %s' % sexp)
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
        'Expression not closed; not enough closing parantheses: %s' % sexp)


def is_bare_word(sexp: Text):
  """Base case of SExpressions."""
  for c in sexp:
    if c in [' ', '(', ')']:
      return False
  return True


def children(sexp: Text) -> List[Text]:
  """Returns the children of an SExpression."""
  if is_bare_word(sexp):
    return []
  validate_parens(sexp)
  pos = 1
  result = []
  while pos < len(sexp) - 1:
    while not is_start_of_word(sexp, pos):
      pos += 1
    end = end_of_child(sexp, pos)
    result.append(sexp[pos:end])
    pos = end
  return result
