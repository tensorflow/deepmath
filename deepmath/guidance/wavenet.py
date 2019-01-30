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
"""Wavenet layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

layers = tf.contrib.layers
slim = tf.contrib.slim


def wavenet_layer(inp,
                  depth,
                  width=3,
                  rate=1,
                  context=None,
                  scope=None,
                  reuse=None):
  """Single wavenet layer.

  This assumes that the input is a rank 4 tensor of shape:
    [batch, reduced_text_dimension, auxilliary_text_dimension, feature_depth]
  If rate is more than one, this will be reshaped to
    [B, R//(2**(rate-1)), A*(2**(rate-1)), D]
  Then a conv2d will be applied with kernel size [width, 1].

  The rest of the wavenet activations will be applied and the result will be
  returned without reshaping, this allows a multilayer wavenet to be implemented
  by subsequent calls to wavenet_layer and rate=2.

  Arguments:
    inp: input tensor
    depth: depth of the intermediate nonlinear activations before reduced.
    width: the width of the conv filter, 2 by default.
    rate: the dilation, use 1 in the first layer and 2 in subsequent layers.
    context: Optional 2-D [batch, dim] tensor on which to condition each node.
    scope: name of scope if given.
    reuse: reuse for variable scope if given.

  Returns:
    output: output tensor.
  """
  tf.logging.info('Creating wavenet layer d=%d w=%d r=%d', depth, width, rate)
  with tf.variable_scope(scope, 'wavenet_layer', [inp], reuse=reuse):
    current_shape = inp.get_shape()
    true_shape = tf.shape(inp)
    in_depth = current_shape[3].value
    mul = 2**(rate - 1)
    reshaped = tf.reshape(
        inp,
        [true_shape[0], true_shape[1] // mul, mul * true_shape[2], in_depth])

    conved = slim.conv2d(
        reshaped,
        2 * depth, [width, 1],
        rate=1,
        padding='SAME',
        activation_fn=None)
    if context is not None:
      conved += layers.linear(context, 2 * depth)[:, None, None, :]

    act = tf.nn.tanh(conved[:, :, :, :depth])
    gate = tf.nn.sigmoid(conved[:, :, :, depth:])

    z = act * gate

    if in_depth != depth:
      z = slim.conv2d(z, in_depth, [1, 1], padding='SAME', activation_fn=None)

    return z + reshaped


def wavenet_block(net,
                  num_layers,
                  depth,
                  comb_weight=1.0,
                  context=None,
                  scope=None,
                  reuse=None,
                  width=3,
                  keep_prob=1.0):
  """Stack many increasingly dilated wavenet layers together.

  Arguments:
    net: input tensor, expected to be 4D to start [batch, text_length, 1, dim]
    num_layers: Number of wavenet layers to apply in the block, note that This
      requires the input text_length to be divisible by 2**num_layers.
    depth: The depth to use for each of the wavenet layers, internally.
    comb_weight: The weight for the residual update (multiplies the residual
      value).
    context: Optional 2-D tensor on which to condition each node.
    scope: Name of scope if given.
    reuse: Reuse for variable scope if given.
    width: Patch size of the convolution.
    keep_prob: Keep probability for the block input dropout.

  Returns:
    output: output tensor, reshaped back to be [batch, text_length, 1, dim]
  """
  inp = net
  tf.logging.info('Creating wavenet block with width %d', width)
  with tf.variable_scope(scope, 'wavenet_block', [net], reuse=reuse):
    # first wavenet layer is a rate=1 conv.
    input_shape = tf.shape(net)
    if keep_prob < 1.0:
      inp_shape = tf.shape(net)
      noise_shape=(inp_shape[0], 1, inp_shape[2], inp_shape[3])
      net = tf.nn.dropout(
          net,
          rate=(1.0 - keep_prob),
          noise_shape=noise_shape)
    net = wavenet_layer(net, depth, rate=1, width=width)
    for _ in range(num_layers):
      # repeated layers are rate=2 but operate on subsequently reshaped inputs.
      # so as to implement increasing dilations.
      net = wavenet_layer(net, depth, rate=2, width=width, context=context)
    # reshape back at top of block
    net = tf.reshape(net, input_shape)
  return comb_weight * net + inp
