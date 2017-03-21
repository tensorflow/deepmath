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
"""Generating random binary trees of a desired shape."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import random
import numpy as np
import scipy.misc
from six.moves import xrange  # pylint: disable=redefined-builtin


class BinaryTreeNode(object):

  def __init__(self, value=None, left=None, right=None):
    self.value = value
    self.left = left
    self.right = right


def memoize(func):
  cache = {}
  def returned_func(*args, **kwargs):
    key = (frozenset(enumerate(args)), frozenset(kwargs.items()))
    result = cache.get(key)
    if result is None:
      result = func(*args, **kwargs)
      cache[key] = result
    return result
  return returned_func


#
# Generating random binary trees with a fixed number of internal (non-leaf)
# nodes, uniformly at random.
#
def find_irreducible_prefix(brackets):
  """Find minimal prefix of string which is balanced (equal Ls and Rs).

  Args:
    brackets: A string containing an equal number of (and only) 'L's and 'R's.

  Returns:
    A two-element tuple.

    The first element is the minimal "irreducible prefix" of
    `brackets`, which is the shortest non-empty prefix of `brackets` which
    contains an equal number of 'L's and 'R's.

    The second element is the rest of the string.

  Raises:
    ValueError: No irreducible prefix could be found, or `brackets` was empty.
  """

  depth = 0
  for i, bracket in enumerate(brackets):
    depth += 1 if bracket == 'L' else -1
    if i > 0 and depth == 0:
      return brackets[:i + 1], brackets[i + 1:]

  raise ValueError('unbalanced or empty: %s' % ''.join(brackets))


def make_well_formed(brackets):
  """Transform bracket string into a well-formed one.

  This uses the algorithm from "Generating binary trees at random" by M.D.
  Atkinson and J.-R. Sack.

  Args:
    brackets: A string containing an equal number of 'L's and 'R's.

  Returns:
    A well-formed version of `brackets`, which begins with L, ends with R, and
    where the cumulative count of Rs (going left to right) never exceeds that of
    Ls.
  """

  if not brackets:
    return []

  irreducible, rest = find_irreducible_prefix(brackets)
  if irreducible[0] == 'L':
    return irreducible + make_well_formed(rest)
  else:
    return (['L'] + make_well_formed(rest) + ['R'] +
            ['R' if c == 'L' else 'L' for c in irreducible[1:-1]])


def brackets_to_tree(brackets):
  """Transform a well-formed bracket string into a tree.

  This mapping is bijective. See the Art of Computer Programming, sections 2.3.2
  and 7.2.1.6, for more details.

  Args:
    brackets: A string containing Ls and Rs.

  Returns:
    A `BinaryTreeNode` corresponding to `brackets`.
  """
  if not brackets:
    return BinaryTreeNode()

  node = BinaryTreeNode()
  irreducible, rest = find_irreducible_prefix(brackets)
  node.left = brackets_to_tree(irreducible[1:-1])
  node.right = brackets_to_tree(rest)

  return node


@memoize
def count_trees_with_num_nodes(n):
  top, bottom = 1, 1
  for k in xrange(2, n+1):
    top *= n + k
    bottom *= k
  return top // bottom


def random_tree_with_num_nodes(n):
  """Sample a binary tree containing N internal nodes, uniformly at random.

  This binary tree of N internal nodes has 0 or 2 children at each node, and
  therefore has N + 1 leaf nodes for a total of 2N + 1 nodes.

  Args:
    n: Number of internal nodes in tree.

  Returns:
    A `BinaryTreeNode`, which is the root of a tree with N internal nodes.
  """
  if n == 0:
    return BinaryTreeNode()

  samples = set(random.sample(xrange(2 * n), n))
  random_brackets = ['L' if i in samples else 'R' for i in xrange(2 * n)]
  well_formed_brackets = make_well_formed(random_brackets)

  return brackets_to_tree(well_formed_brackets)


#
# Generating random binary trees of a fixed depth uniformly at random.
#
@memoize
def count_tree_configs(num_roots, remaining_depth, summed=True):
  """Count number of binary tree configurations with given parameters.

  count_tree_configs(1, d - 1) gives the number of all binary tree
  structures with depth `d`.

  Args:
    num_roots: Maximal number of roots (i.e. number of trees in the forest).
    remaining_depth: Remaining depth of tree.
    summed: Whether to sum results over `num_roots` or return a list for each of
      [1, ..., num_roots].

  Returns:
    Number of possible forests with up to `num_roots` roots and
    `remaining_depth` maximal depth. If `summed` is True, this is an integer;
    if `summed` is False, this is a list (with length `num_roots`).
  """

  if remaining_depth == 0:
    return 1

  counts_by_num_children_expanded = (scipy.misc.comb(num_roots, i) *
                                     count_tree_configs(2 * i,
                                                        remaining_depth - 1)
                                     for i in xrange(1, num_roots + 1))
  if summed:
    return sum(counts_by_num_children_expanded)
  else:
    return list(counts_by_num_children_expanded)


@memoize
def num_subtree_probs(num_roots, remaining_depth):
  counts = count_tree_configs(num_roots, remaining_depth, summed=False)
  return counts / sum(counts)


def random_tree_of_depth(depth):
  """Randomly generate a binary tree with a specified depth.

  This function samples uniformly among all possible well-formed binary trees
  of a certain depth (where depth is the maximal distance from the root to any
  other node).

  Args:
    depth: Depth of desired tree.

  Returns:
    A `BinaryTreeNode` at the root of a randomly-generated tree.
  """

  if depth < 1:
    return None
  elif depth == 1:
    return BinaryTreeNode()

  root = BinaryTreeNode(left=BinaryTreeNode(), right=BinaryTreeNode())
  bottom = np.array([root.left, root.right])
  for i in xrange(2, depth):
    # Decide how many bottom nodes we should add children to.
    # The probability of choosing each count is proportional to the number of
    # different trees which have that substructure.
    extension_probs = num_subtree_probs(len(bottom), depth - i)
    num_nodes_to_extend = np.random.choice(
        len(bottom), 1, p=extension_probs)[0] + 1
    nodes_to_extend = random.sample(list(bottom), num_nodes_to_extend)

    new_bottom = []
    for node in nodes_to_extend:
      node.left = BinaryTreeNode()
      node.right = BinaryTreeNode()
      new_bottom += [node.left, node.right]
    bottom = np.array(new_bottom)
  return root


def all_trees_of_depth(depth):
  """Yields all trees of the given depth."""
  if depth < 1:
    return
  elif depth == 1:
    yield BinaryTreeNode()
    return
  elif depth == 2:
    yield BinaryTreeNode(left=BinaryTreeNode(), right=BinaryTreeNode())
    return

  for node in all_trees_of_depth(depth - 1):
    yield BinaryTreeNode(left=node, right=BinaryTreeNode())
    yield BinaryTreeNode(left=BinaryTreeNode(), right=node)

  for left_node in all_trees_of_depth(depth - 1):
    for right_node in itertools.chain.from_iterable(all_trees_of_depth(d)
                                                    for d in xrange(2, depth)):
      yield BinaryTreeNode(left=left_node, right=right_node)

  for left_node in itertools.chain.from_iterable(all_trees_of_depth(d)
                                                 for d in xrange(2, depth - 1)):
    for right_node in all_trees_of_depth(depth - 1):
      yield BinaryTreeNode(left=left_node, right=right_node)
