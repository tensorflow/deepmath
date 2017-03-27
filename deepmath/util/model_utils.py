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
"""Model utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow_fold.public import loom

layers = tf.contrib.layers


def pad_up_to(value, size, axis, name=None):
  """Pad a tensor with zeros on the right along axis to a least the given size.

  Args:
    value: Tensor to pad.
    size: Minimum size along axis.
    axis: A nonnegative integer.
    name: Optional name for this operation.

  Returns:
    Padded value.
  """
  with tf.name_scope(name, 'pad_up_to') as name:
    value = tf.convert_to_tensor(value, name='value')
    axis = tf.convert_to_tensor(axis, name='axis')
    need = tf.nn.relu(size - tf.shape(value)[axis])
    ids = tf.stack([tf.stack([axis, 1])])
    paddings = tf.sparse_to_dense(ids, tf.stack([tf.rank(value), 2]), need)
    padded = tf.pad(value, paddings, name=name)
    # Fix shape inference
    axis = tf.contrib.util.constant_value(axis)
    shape = value.get_shape()
    if axis is not None and shape.ndims is not None:
      shape = shape.as_list()
      shape[axis] = None
      padded.set_shape(shape)
    return padded


def pad_to_multiple(value, size, axis, name=None):
  """Pad a tensor with zeros on the right to a multiple of the given size.

  Args:
    value: Tensor to pad.
    size: The result will be a multiple of `size` along `axis`.
    axis: A nonnegative integer.
    name: Optional name for this operation.

  Returns:
    Padded value.
  """
  with tf.name_scope(name, 'pad_to_multiple') as name:
    length = tf.shape(value)[axis]
    new_length = length // -size * -size  # Round up to multiple of size
    return pad_up_to(value, size=new_length, axis=axis, name=name)


def shared_embedding_lookup(params, indices, name=None):
  """Lookup embeddings, deduping across multiple index tensors.

  Each value in any of the ids will be looked up once.

  Args:
    params: Embedding matrix or list or matrices.
    indices: List of integer id tensors, each of any shape.
    name: Optional name for this operation.

  Returns:
    List of embedding results.  The ith result has shape
      `indices[i].shape + (params.shape[1],)`.
  """
  if not isinstance(params, (list, tuple)):
    params = [params]
  with tf.name_scope(name, 'shared_embedding', list(params) + list(indices)):
    params = [tf.convert_to_tensor(p, name='params') for p in params]
    indices = [tf.convert_to_tensor(ids, name='ids') for ids in indices]
    all_ids = tf.concat([tf.reshape(ids, [-1]) for ids in indices], 0)
    all_results = layers.embedding_lookup_unique(params, all_ids)
    dim = params[0].get_shape()[1].value
    dim_shape = tf.constant([dim])
    results = []
    parts = tf.split(all_results, [tf.size(ids) for ids in indices])
    for ids, part in zip(indices, parts):
      result = tf.reshape(part, tf.concat([tf.shape(ids), dim_shape], 0))
      result.set_shape(ids.get_shape().concatenate([dim]))
      results.append(result)
    return results


def embedding_weights(dim, size, shards=1, name='embedding'):
  """Makes embedding weights for use in {shared_,}embedding_layer.

  Args:
    dim: Embedding dimension.
    size: Vocabulary size.
    shards: Number of embedding matrix shards.
    name: Name for the embedding variables.

  Returns:
    A list of embedding weight variables.
  """
  scale = 1 / np.sqrt(dim)
  initializer = tf.random_uniform_initializer(minval=-scale, maxval=scale)
  partitioner = tf.fixed_size_partitioner(num_shards=shards)
  with tf.variable_scope(name, initializer=initializer,
                         partitioner=partitioner):
    return tf.get_variable(name, shape=[size, dim], dtype=tf.float32)


def shared_embedding_layer(indices, dim, size, shards=1, name='embedding'):
  """Make an embedding layer that dedupes across multiple index tensors.

  Each value in any of the ids will be looked up once.

  Args:
    indices: List of integer id tensors, each of any shape.
    dim: Embedding dimension.
    size: Vocabulary size.
    shards: Number of embedding matrix shards.
    name: Name for the embedding variables.

  Returns:
    List of embedding results.  The ith result has shape
      `indices[i].shape + (dim,)`.
  """
  params = embedding_weights(dim=dim, size=size, shards=shards, name=name)
  return shared_embedding_lookup(params, indices)


def embedding_layer(indices, dim, size, shards=1, name='embedding'):
  """Make a simple embedding layer.

  Each value in any of the ids will be looked up once.

  Args:
    indices: An integer id tensor.
    dim: Embedding dimension.
    size: Vocabulary size.
    shards: Number of embedding matrix shards.
    name: Name for the embedding variables.

  Returns:
    Embedding results of shape `indices.shape + (dim,)`.
  """
  params = embedding_weights(dim=dim, size=size, shards=shards, name=name)
  return layers.embedding_lookup_unique(params, indices)


def merge_hparams(*params):
  """Merge several HParams objects into one, complaining about duplicates.

  Args:
    *params: Zero or more HParams.

  Returns:
    A single HParams with all hyperparameters together.

  Raises:
    ValueError: If any hyperparameters are duplicated
  """
  all_params = {}
  for ps in params:
    for k, v in ps.values().items():
      if k in all_params:
        raise ValueError('Hyperparameter %r occurs twice with values %r and %r'
                         % (k, all_params[k], v))
      all_params[k] = v
  return tf.contrib.training.HParams(**all_params)


def as_loom_op(inputs, outputs, name=None):
  """Decorator to turn a function into a loom op.

  Example usage:

      @as_loom_op((in0_ts, in1_ts), out_ts)
      def foo(in0, in1):
        return in0 + in1

      @as_loom_op((in0_ts, in1_ts), (out0_ts, out1_ts))
      def bar(in0, in1):
        return in0 + in1, in1 - in0

  A single use of @as_loom_op reuses one variable_scope if used multiple times
  by TensorLoom, so any variables created by layers will be shared.  This
  happens if max_depth != None in TensorLoom.  If you want multiple independent
  sets of variables, make a function that decorates a nested function with
  @as_loom_op:

      def merge(arity):
        @as_loom_op([in_ts] * arity, out_ts)
        def merge(*args):
          return layers.relu(tf.concat(1, args), size)
        return merge
      op1 = merge(1)
      op2 = merge(2)

  Args:
    inputs: Input type shapes (either a list or a single type shape).
    outputs: Output type shapes (either a list or a single type shape).  If a
        single type shape is given, the function should return a single Tensor.
    name: Name scope (defaults to `f.__name__`).

  Returns:
    A decorator to apply to the instantiate batch function (see examples).
  """
  if isinstance(inputs, loom.TypeShape):
    inputs = inputs,
  single_output = isinstance(outputs, loom.TypeShape)
  if single_output:
    outputs = outputs,

  def decorator(f):
    """Decorator to turn a function into a loom op."""
    # Make a fresh scope.  We do this here so that it happens at decorator
    # time so that scope names do not depend on the order in which TensorLoom
    # calls instantiate_batch.
    scope_name = f.__name__ if name is None else name
    with tf.variable_scope(None, scope_name) as scope:

      class AsLoomOp(loom.LoomOp):

        def __init__(self):
          super(AsLoomOp, self).__init__(inputs, outputs)
          self._reuse = False

        def instantiate_batch(self, tensors):
          with tf.variable_scope(scope, reuse=self._reuse):
            self._reuse = True
            tensors = f(*tensors)
            return (tensors,) if single_output else tensors

      return AsLoomOp()

  return decorator


def paired_joint_logits_and_labels(logits, labels, name=None):
  """Compute positive - negative logits and labels assuming a paired ordering.

  The labels are assumed to be true, false, true, false, etc., where each pair
  of two instances comes from the same context.

  Args:
    logits: 1-D logits.
    labels: 1-D binary labels.
    name: Optional name for this operation

  Returns:
    joint_logits: logits[::2] - logits[1::2]
    joint_labels: All true.

  Raises:
    TypeError: If labels is not boolean.
  """
  with tf.name_scope(name, 'paired_joint_logits_and_labels'):
    logits = tf.convert_to_tensor(logits, name='logits')
    labels = tf.convert_to_tensor(labels, name='labels')
    if labels.dtype != tf.bool:
      raise TypeError('labels.dtype = %r != tf.bool' % labels.dtype)
    expected = tf.tile([True, False], tf.shape(labels) // 2)
    check = tf.Assert(tf.reduce_all(tf.equal(labels, expected)),
                      [labels, expected])
    with tf.control_dependencies([check]):
      joint_logits = logits[::2] - logits[1::2]
      joint_labels = tf.ones_like(joint_logits, dtype=tf.bool)
    return joint_logits, joint_labels
