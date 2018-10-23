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
"""TensorLoom ops used in this directory."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_fold.public import loom

framework = tf.contrib.framework
distributions = tfp.distributions


def logsumexp(tensor, name=None):
  with tf.op_scope([tensor], name, 'LogSumExp') as scope:
    max_value = tf.reduce_max(tensor)
    return tf.add(max_value,
                  tf.log(tf.reduce_sum(tf.exp(tensor - max_value))),
                  name=scope)


def logsumexp_loom(inputs, max_op, add_op, sub_op, exp_op, log_op):
  # TODO(ricshin): Add variadic arguments to TensorLoom to avoid these binary
  # reductions that add many iterations to the Loom while loop.
  max_value = reduce_loom(max_op, inputs)
  return add_op(max_value, log_op(
      reduce_loom(add_op, [exp_op(sub_op(inp, max_value)) for inp in inputs])))


def kl_div_gaussians(mean_1, stdev_1, mean_2, stdev_2):
  if mean_1 == 0 and stdev_1 == 1:
    return 0.5 * tf.reduce_sum(
        tf.square(1 / stdev_2) + tf.square(mean_2) - 1 + 2 * tf.log(stdev_2), 1)
  elif mean_2 == 0 and stdev_2 == 1:
    return 0.5 * tf.reduce_sum(
        tf.square(stdev_1) + tf.square(mean_1) - 1 - 2 * tf.log(stdev_1), 1)

  return 0.5 * tf.reduce_sum(
      tf.square(stdev_1 / stdev_2) + tf.square(mean_2 - mean_1) - 1 + 2 *
      (tf.log(stdev_2) - tf.log(stdev_1)), 1)


# TODO(ricshin): Switch to first-party or tf.contrib layer_norm when it exists.
def layer_normalization(x, w, alpha):
  """Matmul `x` with `w` and apply layer norm, scaling with `alpha`."""
  # Notation is from https://arxiv.org/pdf/1607.06450v1.pdf appendix
  z = tf.matmul(x, w)
  mu = tf.reduce_mean(z, [1], keep_dims=True)
  sigma = tf.sqrt(tf.reduce_mean((z - mu)**2, [1], keep_dims=True) + 1e-6)
  return alpha / sigma * (z - mu)


class TagLoomOp(loom.LoomOp):

  def __init__(self, dtype, shape, input_tag='', output_tag=''):
    super(TagLoomOp, self).__init__([loom.TypeShape(dtype, shape, input_tag)],
                                    [loom.TypeShape(dtype, shape, output_tag)])

  def instantiate_batch(self, inputs):
    return inputs


class LinearLoomOp(loom.LoomOp):
  """Concatenate inputs, apply matmul, and then split."""

  def __init__(self,
               name,
               num_inputs,
               input_size,
               num_outputs,
               output_size,
               hidden_sizes=(),
               activation=tf.nn.relu,
               hidden_activation=None,
               layer_norm=False):
    """Constructor.

    Args:
      name: Name of the op, used to create variables.
      num_inputs: Number of input vectors accepted by the LoomOp.
      input_size: The length of each input vector.
      num_outputs: Number of output vectors produced by the LoomOp.
      output_size: The length of each output vector.
      hidden_sizes: Sizes of hidden layers.
      activation: Function to apply on last layer before
        splitting into `num_outputs`.
      hidden_activation: Function to apply to all but the last layer. Same as
        `activation` if not specified.
      layer_norm: If True, apply layer normalization before activation.
    """

    self.num_inputs = num_inputs
    self.num_outputs = num_outputs
    self.layer_norm = layer_norm
    self.num_layers = len(hidden_sizes) + 1
    if hidden_activation is None:
      hidden_activation = activation
    self.activations = (
        (hidden_activation,) * len(hidden_sizes) + (activation,))
    hidden_sizes = tuple(hidden_sizes)

    input_shape = loom.TypeShape(tf.float32, (input_size,))
    output_shape = loom.TypeShape(tf.float32, (output_size,))

    with tf.variable_scope(name):
      if self.layer_norm:
        dims = (input_size,) + hidden_sizes + (num_outputs * output_size,)
        # For the first layer, we apply layer norm separately to each input
        # (like what the paper did for GRUs and LSTMs).
        # After that, there is only one "input" at each layer.
        # There are num_layers + num_inputs - 1 weights.
        inputs_per_layer = [xrange(self.num_inputs)] + [1] * len(hidden_sizes)
        self.w = [
            [framework.variable('w_%d_layer_%d' % (i, l),
                                [input_dim, output_dim])
             for i in xrange(num_inputs)]
            for l, (num_inputs, input_dim, output_dim
                   ) in enumerate(zip(inputs_per_layer, dims, dims[1:]))
        ]

        self.alpha = [[
            framework.variable(
                'alpha_%d_layer_%d' % (i, l), [dim],
                initializer=tf.constant_initializer(1.0))
            for i in xrange(num_inputs)
        ] for l, (num_inputs, dim) in enumerate(zip(inputs_per_layer, dims[1:]))
                     ]
      else:
        dims = ((num_inputs * input_size,) + hidden_sizes +
                (num_outputs * output_size,))
        self.w = [
            framework.variable('w_layer_%d' % l, [input_dim, output_dim])
            for l, (input_dim, output_dim) in enumerate(zip(dims, dims[1:]))
        ]
      self.b = [framework.variable(
          'b_layer_%d' % l, [dim], initializer=tf.constant_initializer(0.0))
                for l, dim in enumerate(dims[1:])]

    super(LinearLoomOp, self).__init__([input_shape] * self.num_inputs,
                                       [output_shape] * self.num_outputs)

  def instantiate_batch(self, inputs):
    # tf.concat is a no-op when len(inputs) == 1.
    if self.layer_norm:
      prev_layer = inputs
      for act_fn, layer_ws, layer_alphas, layer_b in zip(self.activations,
                                                         self.w, self.alpha,
                                                         self.b):
        components = []
        assert len(prev_layer) == len(layer_ws) == len(layer_alphas)
        for inp, w, alpha in zip(prev_layer, layer_ws, layer_alphas):
          components.append(layer_normalization(inp, w, alpha))
        prev_layer = [act_fn(tf.add_n(components) + layer_b)]
      merged_output = prev_layer[0]
    else:
      prev_layer = tf.concat(inputs, 1)
      for act_fn, layer_w, layer_b in zip(self.activations, self.w, self.b):
        prev_layer = act_fn(tf.matmul(prev_layer, layer_w) + layer_b)
      merged_output = prev_layer

    return tf.split(
        value=merged_output, num_or_size_splits=self.num_outputs, axis=1)


class GatedLinearLoomOp(loom.LoomOp):
  """Gated version of LinearLoomOp.

  Gated inputs are summed together with a transformed version of the inputs
  in order to form each output. This means that if the gate values are
  set appropriately, each output can be a copy of some input, or a weighted
  average of the inputs.
  """

  def __init__(self,
               name,
               emb_size,
               num_inputs,
               num_outputs,
               hidden_outputs=(),
               activation=tf.nn.relu,
               hidden_activation=None,
               layer_norm=False,
               gate_type='sigmoid',
               tied_gates=False,
               g_c_bias=-1.0,
               g_x_bias=1.0):
    """Constructor.

    Args:
      name: Name of the op, used to create variables.
      emb_size: The length of each input and output vector.
      num_inputs: Number of input vectors accepted by the LoomOp.
      num_outputs: Number of output vectors produced by the LoomOp.
      hidden_outputs: Number of hidden output vectors in each layer.
      activation: Function to apply to `c` at the last layer.
      hidden_activation: Function to apply to `c` at all but the last layer.
        Same as `activation` if not specified.
      layer_norm: If True, apply layer normalization before activation.
      gate_type: 'sigmoid' or 'softmax'. With 'softmax', gates on `x_i` and `c`
        sum to 1. With 'sigmoid', each gate is between 0 and 1 but the values
        do not sum to 1.
      tied_gates: If True, the unactivated gate value for each x_i is identical
        across all outputs. If False, it is different for each output.
      g_c_bias: Added to `g_c` as a constant. Positive values imply greater
        weight on `c`.
      g_x_bias: Added to `g_x` as a constant. Positive values imply greater
        weight on `x` (the inputs).
    """
    self.emb_size = emb_size
    self.num_inputs = num_inputs
    self.num_outputs = num_outputs
    self.activation = activation
    self.layer_norm = layer_norm
    self.gate_type = gate_type
    self.tied_gates = tied_gates
    self.g_c_bias = g_c_bias
    self.g_x_bias = g_x_bias

    emb_shape = loom.TypeShape(tf.float32, (emb_size,))

    hidden_outputs = tuple(hidden_outputs)
    self.layer_counts = (num_inputs,) + hidden_outputs + (num_outputs,)
    self.intermediate_count = []
    self.w, self.alpha, self.b = [], [], []
    if hidden_activation is None:
      hidden_activation = activation
    self.activations = (
        [hidden_activation] * len(hidden_outputs) + [activation])
    with tf.variable_scope(name):
      for l, (num_inputs, num_outputs
             ) in enumerate(zip(self.layer_counts, self.layer_counts[1:])):
        # TODO(ricshin): If gate_type = softmax, num_inputs = 1, and num_outputs
        # = 1, reduce gate count by 1 and use sigm, 1 - sigm instead.
        if tied_gates:
          intermediate_count = num_inputs + 2 * num_outputs
        else:
          intermediate_count = (num_inputs + 2) * num_outputs
        self.intermediate_count.append(intermediate_count)

        if self.layer_norm:
          self.w.append([framework.variable('w_%d_layer_%d' % (i, l),
                                            [emb_size,
                                             intermediate_count * emb_size])
                         for i in xrange(num_inputs)])

          self.alpha.append([framework.variable(
              'alpha_%d_layer_%d' % (i, l), [intermediate_count * emb_size],
              initializer=tf.constant_initializer(1.0))
                             for i in xrange(num_inputs)])
        else:
          self.w.append(
              framework.variable('w_layer_%d' % l, [
                  num_inputs * emb_size, intermediate_count * emb_size
              ]))
          self.alpha.append(None)
        self.b.append(
            framework.variable(
                'b_layer_%d' % l, [intermediate_count * emb_size],
                initializer=tf.constant_initializer(0.0)))

    super(GatedLinearLoomOp, self).__init__([emb_shape] * self.num_inputs,
                                            [emb_shape] * self.num_outputs)

  def instantiate_batch(self, inputs):
    for (act_fn, layer_ws, layer_alphas, layer_b, int_count, num_inputs,
         num_outputs) in zip(self.activations, self.w, self.alpha, self.b,
                             self.intermediate_count, self.layer_counts,
                             self.layer_counts[1:]):
      if self.layer_norm:
        components = []
        for inp, w, alpha in zip(inputs, layer_ws, layer_alphas):
          components.append(layer_normalization(inp, w, alpha))
        concat_intermediates = tf.add_n(components) + layer_b
      else:
        # concat_inputs shape: bs x (num_inputs * emb_size)
        concat_inputs = tf.concat(inputs, 1)
        concat_intermediates = tf.matmul(concat_inputs, layer_ws) + layer_b
      intermediates = tf.split(
          value=concat_intermediates, num_or_size_splits=int_count, axis=1)

      # "x" refers to `inputs`.
      c = [act_fn(el) for el in intermediates[:num_outputs]]
      g_c = [g + self.g_c_bias
             for g in intermediates[num_outputs:2 * num_outputs]]
      g_x = [g + self.g_x_bias for g in intermediates[2 * num_outputs:]]

      outputs = []
      for i in xrange(num_outputs):
        x_gates = g_x[:num_inputs]
        if not self.tied_gates:
          # Need to use different values of g_x for the next output.
          g_x = g_x[num_inputs:]

        # bs x emb_size x (num_inputs + 1)
        gates = tf.stack(x_gates + [g_c[i]], axis=2)

        if self.gate_type == 'softmax':
          # (bs * emb_size) x (num_inputs + 1)
          gates = tf.reshape(gates, [-1, num_inputs + 1])
          gates = tf.nn.softmax(gates)
          # bs x emb_size x (num_inputs + 1)
          gates = tf.reshape(gates, [-1, self.emb_size, num_inputs + 1])
        elif self.gate_type == 'sigmoid':
          # TODO(ricshin): Avoid recomputing sigmoid(x_gates) * inputs
          # if tied_gates is True, since x_gates doesn't change.
          gates = tf.sigmoid(gates)
        else:
          raise ValueError('Invalid value for gate_type: %r' % self.gate_type)

        # bs x emb_size x (num_inputs + 1)
        all_inputs = tf.stack(inputs + [c[i]], axis=2)
        gated = gates * all_inputs

        # bs x emb_size
        outputs.append(tf.reduce_sum(gated, 2))
      inputs = outputs

    return outputs


def merge(name, arity, emb_size, **kwargs):
  return LinearLoomOp(
      name,
      num_inputs=arity,
      input_size=emb_size,
      num_outputs=1,
      output_size=emb_size,
      **kwargs)


def split(name, arity, emb_size, **kwargs):
  return LinearLoomOp(
      name,
      num_inputs=1,
      input_size=emb_size,
      num_outputs=arity,
      output_size=emb_size,
      **kwargs)


def merge_gated(name, arity, emb_size, **kwargs):
  return GatedLinearLoomOp(
      name, emb_size, num_inputs=arity, num_outputs=1, **kwargs)


def split_gated(name, arity, emb_size, **kwargs):
  return GatedLinearLoomOp(
      name, emb_size, num_inputs=1, num_outputs=arity, **kwargs)


def which(name, num_options, emb_size, **kwargs):
  return LinearLoomOp(
      name,
      num_inputs=1,
      input_size=emb_size,
      num_outputs=1,
      output_size=num_options,
      activation=lambda x: x,
      **kwargs)


class LSTMLoomOp(loom.LoomOp):
  """Implements an LSTM cell, equivalent to BasicLSTMCell."""

  def __init__(self, name, emb_size, activation=tf.tanh, layer_norm=False):

    emb_shape = loom.TypeShape(tf.float32, (emb_size,))

    with tf.variable_scope(name):
      if layer_norm:
        self.w_h = framework.variable('w_h', [emb_size, emb_size * 4])
        self.w_x = framework.variable('w_x', [emb_size, emb_size * 4])
        self.alpha_h = framework.variable(
            'alpha_h', [emb_size * 4], initializer=tf.constant_initializer(1.0))
        self.alpha_x = framework.variable(
            'alpha_x', [emb_size * 4], initializer=tf.constant_initializer(1.0))
      else:
        self.w = framework.variable('w', [emb_size * 2, emb_size * 4])
      self.b = framework.variable(
          'b', [emb_size * 4], initializer=tf.constant_initializer(0.0))

    self.layer_norm = layer_norm
    self.activation = activation

    super(LSTMLoomOp, self).__init__([emb_shape] * 3, [emb_shape] * 2)

  def instantiate_batch(self, args):
    inp, c, h = args
    if self.layer_norm:
      state = (layer_normalization(inp, self.w_x, self.alpha_x) +
               layer_normalization(inp, self.w_h, self.alpha_h) + self.b)
    else:
      state = tf.matmul(tf.concat([inp, h], 1), self.w) + self.b

    i, j, f, o = tf.split(value=state, num_or_size_splits=4, axis=1)
    new_c = c * tf.sigmoid(f + 1.0) + tf.sigmoid(i) * self.activation(j)
    new_h = self.activation(new_c) * tf.sigmoid(o)

    return [new_c, new_h]


class EmbeddingLookupLoomOp(loom.LoomOp):
  """Look up embeddings."""

  def __init__(self, name, num_options, emb_size, partitions=1, input_tag=''):
    input_shape = loom.TypeShape(tf.int64, [], input_tag)
    output_shape = loom.TypeShape(tf.float32, (emb_size,))

    options_per_partition = (num_options - 1) // partitions + 1
    self.embeddings = [tf.get_variable(
        name + '_emb', [options_per_partition, emb_size],
        initializer=tf.random_uniform_initializer(-0.05, 0.05))
                       for _ in xrange(partitions)]

    super(EmbeddingLookupLoomOp, self).__init__([input_shape], [output_shape])

  def instantiate_batch(self, args):
    indices, = args
    return [tf.nn.embedding_lookup(self.embeddings, indices)]


class MultinomialLoomOp(loom.LoomOp):
  """Sample from a multinomial parameterized with logits."""

  def __init__(self, num_options, tag=''):
    input_shape = loom.TypeShape(tf.float32, (num_options,))
    temperature_shape = loom.TypeShape(tf.float32, [])
    output_shape = loom.TypeShape(tf.int64, [], tag)

    super(MultinomialLoomOp, self).__init__([input_shape, temperature_shape],
                                            [output_shape])

  def instantiate_batch(self, args):
    logits, temperature = args
    temperature = tf.reshape(temperature, [-1, 1])
    return [tf.squeeze(tf.multinomial(logits / temperature, 1), [-1])]


class KLDivPosteriorPriorLoomOp(loom.LoomOp):
  """Compute posterior q(z | x) and the posterior-prior KL divergence.

  The KL divergence is KL(q(z | x) || p(z)) and the sample is from q(z | x).

  p(z) is the standard multivariate normal distribution.

  q(z | x) is computed within this LoomOp, by taking the input embedding and
  transforming it with a linear layer to get means and variances for a
  multivariate normal distribution with diagonal covariance.
  """

  def __init__(self, emb_size, reverse=False):
    """Constructor.

    Args:
      emb_size: Size of input.
      reverse: If True, compute KL(p(z) || q(z | x)) in addition.
    """
    emb_shape = loom.TypeShape(tf.float32, (emb_size,))
    cost_shape = loom.TypeShape(tf.float32, [], 'kl_div')

    self.w = framework.variable('q_z_x_w', [emb_size, emb_size * 2])
    self.b = framework.variable(
        'q_z_x_b', [emb_size * 2], initializer=tf.constant_initializer(0.0))
    self.emb_size = emb_size
    self.reverse = reverse

    if reverse:
      rev_cost_shape = loom.TypeShape(tf.float32, [], 'rev_kl_div')
      super(KLDivPosteriorPriorLoomOp, self).__init__(
          [emb_shape], [cost_shape, rev_cost_shape, emb_shape, emb_shape])
    else:
      super(KLDivPosteriorPriorLoomOp, self).__init__(
          [emb_shape], [cost_shape, emb_shape, emb_shape])

  def instantiate_batch(self, inputs):
    mean_and_log_stdev = tf.matmul(inputs[0], self.w) + self.b
    mean = mean_and_log_stdev[:, :self.emb_size]
    # Force minimum stdev to be 1e-2, so that the KL divergence never blows up.
    stdev = tf.nn.softplus(mean_and_log_stdev[:, self.emb_size:]) + 1e-2

    # TODO(ricshin): Switch to tfp.distributions.kl_divergence when it has constant
    # folding.
    kl_div = kl_div_gaussians(mean, stdev, 0, 1)

    if self.reverse:
      reverse_kl_div = kl_div_gaussians(0, 1, mean, stdev)
      return kl_div, reverse_kl_div, mean, stdev
    else:
      return kl_div, mean, stdev


class PrintLoomOp(loom.LoomOp):

  def __init__(self, type_shape, *args, **kwargs):
    self.args = args
    self.kwargs = kwargs
    super(PrintLoomOp, self).__init__([type_shape], [type_shape])

  def instantiate_batch(self, args):
    inp, = args
    return [tf.cond(
        tf.equal(tf.size(inp), 0), lambda: inp,
        lambda: tf.Print(inp, [inp], *self.args, **self.kwargs))]


class SampleFromGaussianLoomOp(loom.LoomOp):

  def __init__(self, emb_size):
    emb_shape = loom.TypeShape(tf.float32, (emb_size,))
    super(SampleFromGaussianLoomOp, self).__init__([emb_shape, emb_shape],
                                                   [emb_shape])

  def instantiate_batch(self, inputs):
    mean, stdev = inputs
    noise = tf.random_normal(tf.shape(mean))
    return [mean + noise * stdev]


class LogPdfWithStandardNormalLoomOp(loom.LoomOp):

  def __init__(self, emb_size):
    self._prior = distributions.Normal(0., 1.)
    super(LogPdfWithStandardNormalLoomOp, self).__init__(
        [loom.TypeShape(tf.float32, (emb_size,))],
        [loom.TypeShape(tf.float32, ())])

  def instantiate_batch(self, args):
    z, = args
    # log_pdf doesn't do constant folding for 0 and 1, which is unfortunate...
    return [tf.reduce_sum(self._prior.log_prob(z), reduction_indices=[1])]


class NegativeLogPdfWithMultivariateNormalLoomOp(loom.LoomOp):

  def __init__(self, emb_size):
    emb_shape = loom.TypeShape(tf.float32, (emb_size,))
    super(NegativeLogPdfWithMultivariateNormalLoomOp, self).__init__(
        [emb_shape, emb_shape, emb_shape], [loom.TypeShape(tf.float32, ())])

  def instantiate_batch(self, args):
    mu, sigma, z = args
    return [-tf.reduce_sum(distributions.Normal(mu, sigma).log_prob(z), [1])]


class SparseSoftmaxCrossEntropyLossLoomOp(loom.LoomOp):

  def __init__(self, num_options, tag_name='xent_cost'):
    logits_shape = loom.TypeShape(tf.float32, (num_options,))
    labels_shape = loom.TypeShape(tf.int64, [])
    output_shape = loom.TypeShape(tf.float32, [], tag_name)

    super(SparseSoftmaxCrossEntropyLossLoomOp, self).__init__(
        [logits_shape, labels_shape], [output_shape])

  def instantiate_batch(self, inputs):
    logits, labels = inputs
    result = (
        tf.contrib.nn.deprecated_flipped_sparse_softmax_cross_entropy_with_logits(  # pylint: disable=line-too-long
            logits, labels))
    return [result]


class UnaryLoomOp(loom.LoomOp):
  """Apply a unary operation to one input."""

  def __init__(self, op, left_shape, result_shape=None):
    # If these are None, set them equal to left_shape.
    result_shape = left_shape if result_shape is None else result_shape
    self.op = op

    super(UnaryLoomOp, self).__init__([left_shape], [result_shape])

  def instantiate_batch(self, inputs):
    return [self.op(*inputs)]


class BinaryLoomOp(loom.LoomOp):
  """Apply a binary operation to two inputs."""

  def __init__(self, op, left_shape, right_shape=None, result_shape=None):
    # If these are None, set them equal to left_shape.
    right_shape = left_shape if right_shape is None else right_shape
    result_shape = left_shape if result_shape is None else result_shape
    self.op = op

    super(BinaryLoomOp, self).__init__([left_shape, right_shape],
                                       [result_shape])

  def instantiate_batch(self, inputs):
    return [self.op(*inputs)]


def reduce_loom(op, operands):
  """Apply a binary LoomOp on `operands` to obtain one result."""
  if not operands:
    raise ValueError('operands is empty')

  # Binary tree reduction.
  while len(operands) > 1:
    evens = operands[::2]
    odds = operands[1::2]
    new_operands = [op(*pair) for pair in zip(evens, odds)]
    if len(evens) != len(odds):
      new_operands.append(operands[-1])
    operands = new_operands
  return operands[0]
