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
"""Jagged arrays."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from deepmath.guidance import gen_jagged_ops

repeats = gen_jagged_ops.repeats


class Jagged(object):
  """An array of tensors with varying first dimensions."""

  def __init__(self, sizes, flat):
    """Constructs a jagged array.

    Args:
      sizes: 1-D integer tf.Tensor with the size of each first dimension.  Must
          sum to the first dimension of flat.
      flat: tf.Tensor of subarrays concatenated along the first dimension.
    """
    self._sizes = sizes
    self._flat = flat
    self._size = None
    self._offsets = None

  @property
  def size(self):
    """The number of sequences (a 0-D int Tensor)."""
    if self._size is None:
      self._size = tf.size(self.sizes)
    return self._size

  @property
  def sizes(self):
    """The size of each sequence (a 1-D int Tensor)."""
    return self._sizes

  def get_shape(self):
    """What we know about the shape, as a TensorShape."""
    return (self.sizes.get_shape().concatenate([None]).
            concatenate(self.flat.shape[1:]))

  @property
  def offsets(self):
    """The offsets of each sequence, starting with zero (a 1-D int Tensor)."""
    if self._offsets is None:
      self._offsets = tf.cumsum(tf.concat([[0], self.sizes], 0))
    return self._offsets

  @property
  def flat(self):
    """All sequences concatenated along the first dimension."""
    return self._flat

  def __getitem__(self, i):
    """Returns the ith sequence."""
    offsets = self.offsets
    return self._flat[offsets[i]:offsets[i + 1]]


def pack(values, name=None):
  """Pack a Python list of tensors into a single Jagged tensor.

  This is similar to tf.pack, but allows the first dimensions to vary.

  Args:
    values: List of tensors to pack.
    name: Optional name for this operation.

  Returns:
    A `Jagged` equivalent to values.
  """
  with tf.name_scope(name, 'jagged_pack'):
    values = [tf.convert_to_tensor(v) for v in values]
    sizes = tf.stack([tf.shape(v)[0] for v in values])
    flat = tf.concat(values, 0)
    return Jagged(sizes, flat)


def flatten(value):
  """Flatten value if it is a Jagged, return unchanged if it's a Tensor."""
  return value.flat if isinstance(value, Jagged) else value


def jagged(sizes, flat):
  """Constructs a possibly jagged array.

  This routine is for generic code that might be running on either jagged
  or rectangular input.  For example,

      sizes, flat = unjagged(data)
      flat = tf.nn.relu(flat)
      data = jagged(sizes, falt)

  Args:
    sizes: Either None for rectangular input, or a possibly jagged tensor
        of sizes.
    flat: tf.Tensor of subarrays concatenated along the first dimension.

  Returns:
    flat if sizes is None, or Jagged(sizes, flat)
  """
  return flat if sizes is None else Jagged(sizes, flat)


def unjagged(value):
  """Split a Tensor or Jagged into (sizes, flat).

  This routine is for generic code that might be running on either jagged
  or rectangular input.  For example,

      sizes, flat = unjagged(data)
      flat = tf.nn.relu(flat)
      data = jagged(sizes, falt)

  Args:
    value: Possibly Jagged array to split.

  Returns:
    sizes: value.sizes for Jagged input, otherwise None.
    flat: value.flat for Jagged input, otherwise value.
  """
  if isinstance(value, Jagged):
    return value.sizes, value.flat
  return None, value


def conv1d_stack(sequences, filters, activations, name=None):
  """Convolve a jagged batch of sequences with a stack of filters.

  This is equivalent to running several `conv1d`s on each `sequences[i]` and
  reassembling the results as a `Jagged`.  The padding is always 'SAME'.

  Args:
    sequences: 4-D `Jagged` tensor.
    filters: List of 3-D filters (one filter per layer).  Must have odd width.
    activations: List of activation functions to apply after each layer, or
        None to indicate no activation.
    name: Optional name for this operation.

  Returns:
    `Jagged` convolution results.

  Raises:
    TypeError: If sequences is not Jagged.
    ValueError: If the filters or activations are invalid.
  """
  if not isinstance(sequences, Jagged):
    raise TypeError('Expected Jagged sequences, got %s' % type(Jagged))
  if len(filters) != len(activations):
    raise ValueError('Got %d filters != %d activations' %
                     (len(filters), len(activations)))
  if not filters:
    return sequences

  with tf.name_scope(name, 'jagged_conv1d_stack') as name:
    # Compute maximum filter width
    filters = [tf.convert_to_tensor(f, name='filter') for f in filters]
    width = 0
    for filt in filters:
      shape = filt.get_shape()
      if shape.ndims != 3 or shape[0] is None or shape[0].value % 2 == 0:
        raise ValueError('Expected known odd filter width, got shape %s' %
                         shape)
      width = max(width, shape[0].value)
    between = width // 2  # Rounds down since width is odd

    # Add 'between' zeros between each sequence
    flat = sequences.flat
    sizes = flatten(sequences.sizes)
    size = tf.size(sizes)
    flat_shape = tf.shape(flat)
    flat_len = flat_shape[0]
    indices = (tf.range(flat_len) + repeats(between * tf.range(size), sizes))
    padded_len = between * tf.nn.relu(size - 1) + flat_len
    flat = tf.unsorted_segment_sum(flat, indices, padded_len)[None]

    # Make a mask to reset between portions to zero
    if len(filters) > 1:
      mask = tf.unsorted_segment_sum(
          tf.ones(flat_shape[:1], dtype=flat.dtype), indices, padded_len)
      mask = mask[:, None]

    # Do each convolution
    for i, (filt, activation) in enumerate(zip(filters, activations)):
      if i:
        flat *= mask
      flat = tf.nn.conv1d(flat, filt, stride=1, padding='SAME')
      if activation is not None:
        flat = activation(flat)

    # Extract results and repackage as a Jagged
    flat = tf.squeeze(flat, [0])
    flat = tf.gather(flat, indices, name=name)
    return Jagged(sequences.sizes, flat)


def reduce_max(sequences, name=None):
  """Computes componentwise maxes along each sequence in a `Jagged` tensor.

  Equivalent to `pack([reduce_max(s, axis=0) for s in sequences])`.

  Args:
    sequences: `Jagged` tensor of sequences.
    name: Optional name for this operation.

  Returns:
    Tensor of componentwise maxima.
  """
  if not isinstance(sequences, Jagged):
    raise TypeError('Expected Jagged sequences, got %s' % type(Jagged))
  with tf.name_scope(name, 'jagged_reduce_max') as name:
    sizes, flat = unjagged(sequences)
    sizes_sizes, sizes_flat = unjagged(sizes)
    maxes, _ = gen_jagged_ops.jagged_max(sizes_flat, flat, name=name)
    return jagged(sizes_sizes, maxes)


@tf.RegisterGradient('JaggedMax')
def _jagged_max_grad(op, grad, _):
  """Gradient of JaggedMax."""
  flat = op.inputs[1]
  argmax = op.outputs[1]
  out_shape = tf.shape(argmax)

  # Reduce to 2-D case
  inner = tf.reduce_prod(out_shape[1:])
  matrix_shape = tf.stack([out_shape[0], inner])
  grad = tf.reshape(grad, matrix_shape)
  argmax = tf.reshape(argmax, matrix_shape) * inner + tf.range(inner)

  # Compute gradients
  grad = tf.unsorted_segment_sum(grad, argmax, tf.size(flat))
  return None, tf.reshape(grad, tf.shape(flat))
