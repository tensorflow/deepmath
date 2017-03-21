# Copyright 2017 Google Inc. All Rights Reserved.
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
"""Utility functions for arithmetic expressions used in arith_*.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def stringify_expr(expr):
  if expr[0] == 'number':
    return expr[1][0]
  elif expr[0] == 'plus':
    return '(%s + %s)' % (stringify_expr(expr[1]), stringify_expr(expr[2]))
  elif expr[0] == 'minus':
    return '(%s - %s)' % (stringify_expr(expr[1]), stringify_expr(expr[2]))


def none_as_zero(n):
  if n is None:
    return 0
  else:
    return n


def eval_expr(expr):
  if expr is None or expr[0] is None:
    return float('nan')
  elif expr[0] == 'number':
    return none_as_zero(expr[1][0])
  elif expr[0] == 'plus':
    return eval_expr(expr[1]) + eval_expr(expr[2])
  elif expr[0] == 'minus':
    return eval_expr(expr[1]) - eval_expr(expr[2])
  else:
    raise ValueError('Invalid expression: %s' % repr(expr))
