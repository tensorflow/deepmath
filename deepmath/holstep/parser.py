# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Parse HOLStep terms into S-expressions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import sys


# Our tokens are either parentheses, commas, or anything else sans whitespace.
_TOKEN_RE = re.compile(r'[(),]|[^\s(),]+')
_BINDER_RE = re.compile(r"^([!?\\@]|\?!|lambda)([a-zA-Z0-9_%'<>]+)\.$")
_BINDERS = {'\\', '?!', '?', '!', '@'}
_OPERATORS = {'IN', 'SUBSET', '>', '<', '=', '..', '-', '+', '*', '==',
              '<=', '==>', '>=', 'MOD', '$', '|-', 'UNION', '\\/', ',',
              'INSERT', 'INTER', 'DIFF', 'PCROSS', 'DELETE', 'HAS_SIZE',
              'EXP', 'PSUBSET', 'DIV', 'CROSS', 'o', '=_c', '<=_c', 'treal_eq',
              '<_c', 'treal_le', 'treal_mul', 'treal_add', '/\\', '>=_c',
             }
_BINDERS_AND_OPERATORS = _BINDERS | _OPERATORS


def _finalize(t):
  """Finalize the top level of an S-expression."""
  if len(t) == 3:
    # Fix operators to go in the front.
    assert isinstance(t[1], str)  # Operators are never calls
    assert t[1] in _OPERATORS
    t = t[1], t[0], t[2]
  elif len(t) == 2:
    # Flatten currying
    if isinstance(t[0], tuple):
      # Don't uncurry binders or operators
      if t[0][0] not in _BINDERS_AND_OPERATORS:
        t = t[0] + t[1:]
    # Split out the variables for the quantifiers.
    else:
      match = _BINDER_RE.match(t[0])
      if match:
        binder = match.group(1)
        if binder == 'lambda':
          binder = '\\'
        variable = match.group(2)
        t = (binder, variable, t[1])
  elif len(t) > 3 and t[-2] == '|-':
    # Handle turnstiles with more than one comma separated assumption
    assert t[1:-2:2] == (',',) * ((len(t) - 3) // 2)
    return (t[-2],) + t[::2]
  else:
    raise ValueError('Invalid intermediate parse %r' % (t,))
  return t


def parse_term(term):
  """Parse a bracketed expression from HOLStep.

  HOLStep's bracketed expressions appear to be a subset of valid HOL Light
  terms.  They are fully parenthesized so precedence doesn't matter, and there
  are always spaces between operators and identifiers.

  Args:
    term: HOLStep term or thm to parse, as a str.

  Returns:
    The parsed term or thm as an S-expression.

  Raises:
    ValueError: If the parse fails.
  """
  # Split into tokens
  words = _TOKEN_RE.findall(term)

  # Convert to S-expressions
  stack = [[]]
  for w in words:
    if w == '(':
      stack.append([])
    else:
      if w == ')':
        w = _finalize(tuple(stack.pop()))
      stack[-1].append(w)
  term, = stack

  # All done!
  return _finalize(tuple(term))


def show_sexp(e):
  """Convert an S-expr to a string."""
  if isinstance(e, tuple):
    return '(%s)' % ' '.join(show_sexp(s) for s in e)
  elif isinstance(e, str):
    return e
  else:
    raise TypeError('Expected tuple or str, got %r' % (e,))


def _main():
  """Parse everything in one or more HOLStep files.

  `parser <files...>` prints each of the HOLStep files back, adding
  lines starting with 'S' with the S-expression versions of each thm.

  For debugging purposes only.
  """
  for path in sys.argv[1:]:
    for line in open(path):
      # Always print the line back unchanged
      print(line, end='')
      # Parse tokenized lines and print S-expr versions
      if line and line[0] in '+-':
        s = parse_term(line[2:])
        print('S', show_sexp(s))


if __name__ == '__main__':
  _main()
