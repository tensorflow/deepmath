"""Architecture functions for HOLparam models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import tensorflow as tf
from deepmath.deephol.train import losses
from deepmath.deephol.train import utils
from deepmath.deephol.train import wavenet

FLAGS = tf.flags.FLAGS

TRAIN = tf.estimator.ModeKeys.TRAIN


def get_vocab_embedding(embedding_str, params):
  return tf.get_variable(
      embedding_str,
      shape=(params.vocab_size, params.word_embedding_size),
      dtype=tf.float32)


def _pad_up_to(value, size, axis, name=None):
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


def _pad_to_multiple(value, size, axis, name=None):
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
    return _pad_up_to(value, size=new_length, axis=axis, name=name)


def wavenet_encoding(net, params, mode):
  """Embed a given input tensor using multiple wavenet blocks.

  Arguments:
    net: input tensor of shape [batch, text_length, word_embedding_size]
    params: Hyperparameters.
    mode: Estimator mode.

  Returns:
    output: output tensor of shape [batch, 1, text length, hidden_size]
  """
  if params.word_embedding_size != params.hidden_size:
    net = tf.layers.dense(net, params.hidden_size, activation=None)
  net = _pad_to_multiple(net, 2**params.wavenet_layers, axis=1)
  net = tf.expand_dims(net, 2)
  if params.input_keep_prob < 1.0 and mode == TRAIN:
    net = tf.nn.dropout(
        net,
        rate=(1.0 - params.input_keep_prob),
        noise_shape=(tf.shape(net)[0], tf.shape(net)[1], 1, 1))
  layer_keep_prob = params.layer_keep_prob
  if mode != TRAIN:
    layer_keep_prob = 1.0
  for _ in range(params.wavenet_blocks):
    net = wavenet.wavenet_block(
        net,
        num_layers=params.wavenet_layers,
        depth=params.wavenet_depth,
        comb_weight=params.layer_comb_weight,
        keep_prob=layer_keep_prob)
  return net


class EncodingSpec(
    collections.namedtuple(
        'EncodingSpec',
        ['enc', 'dist', 'pfstate_enc', 'thm_enc', 'att_key_sim'])):
  """Encoding specification.

  enc: Encoding of (pfstate, thm), possibly drawn from a learned distribution.
  dist: Conditional distribution, trained by a regularizer.
  pfstate_enc: Encoding of the proof state (goal only or goal with context).
  thm_enc: Encoding of the theorem.
  att_key_sim: Similarities of attention keys in the encoder.
  """
  __slots__ = ()

  def __new__(cls,
              enc=None,
              dist=None,
              pfstate_enc=None,
              thm_enc=None,
              att_key_sim=None):
    return super(EncodingSpec, cls).__new__(cls, enc, dist, pfstate_enc,
                                            thm_enc, att_key_sim)


def dilated_cnn_goal_encoder(features, labels, mode, params, config):
  """Dilated convolution network.

  Args:
    features: goal and theorem pair. goal_ids has shape [batch_size, length of
      longest goal]
    labels: labels are unused.
    mode: train or eval.
    params: hyperparameters
    config: configuration object

  Returns:
    Encoding for the goal. [batch_size * (1 + ratio_neg_examples), hidden_size]
  """
  del labels, config  # Unused by this encoder

  # goal_ids shape is [batch_size, length of goal]
  tf.add_to_collection('goal_ids', features['goal_ids'])
  goal_embedding = get_vocab_embedding('goal_embedding', params)
  # output shape is [batch_size, goal length, word_embedding_size]
  goal_net = tf.nn.embedding_lookup(goal_embedding, features['goal_ids'])
  tf.add_to_collection('goal_embedding', goal_net)
  with tf.variable_scope('goal', reuse=False):
    # output shape: [batch_size, 1, goal length, hidden_size]
    goal_net = wavenet_encoding(goal_net, params, mode)
  # output shape is [batch_size, hidden_size]
  goal_net = tf.reduce_max(goal_net, [1, 2])

  # The first goal_net in the collection matches the number of unique goals.
  # This will be used by the predictor to compute the embedding of the goals.
  tf.add_to_collection('goal_net', goal_net)

  # The second goal_net in the collection contains duplicates, aligning with the
  # number of positive and negative theorems. The predictor will feed this value
  # in to compute the score of goal/theorem pairs.
  # output shape: [goal_tiling_size * batch_size, hidden_size]
  goal_tiling_size = params.ratio_neg_examples + 1
  goal_net = tf.tile(goal_net, [goal_tiling_size, 1])
  tf.add_to_collection('goal_net', goal_net)

  return goal_net


def dilated_cnn_thm_encoder(features, labels, mode, params, config):
  """Dilated convolution network.

  Args:
    features: goal and theorem pair. thm_ids has shape [batch_size, length of
      longest theorem]
    labels: labels are unused.
    mode: train or eval.
    params: hyperparameters
    config: configuration object

  Returns:
    Encoding for the thm, shape  [batch_size, hidden_size]
  """
  del labels, config  # Unused by this encoder

  tf.add_to_collection('thm_ids', features['thm_ids'])
  # thm_ids shape is [batch_size, length of thm]
  if params.thm_vocab is not None:
    thm_embedding = get_vocab_embedding('thm_embedding', params)
    # output shape is [batch_size, thm length, word_embedding_size]
    thm_net = tf.nn.embedding_lookup(thm_embedding, features['thm_ids'])
  else:
    goal_embedding = get_vocab_embedding('goal_embedding', params)
    # output shape is [batch_size, thm length, word_embedding_size]
    thm_net = tf.nn.embedding_lookup(goal_embedding, features['thm_ids'])
  tf.add_to_collection('thm_embedding', thm_net)

  with tf.variable_scope('thm', reuse=False):
    # output shape: [batch_size, 1, thm length, hidden_size]
    thm_net = wavenet_encoding(thm_net, params, mode)

  # output shape is [batch_size, hidden_size]
  thm_net = tf.reduce_max(thm_net, [1, 2])
  tf.add_to_collection('thm_net', thm_net)

  return thm_net


def _concat_net_tac_id(net, labels, params):
  """Concatenate net with one-hot vectors of tac_id."""
  if labels is not None:
    tac_id = labels['tac_id']
  else:
    tac_id = tf.tile(tf.constant([-1]), [tf.shape(net)[0]])

  tf.add_to_collection('label_tac_id', tac_id)

  # shape: [batch_size, num_tactics]
  label_tac_one_hot = tf.one_hot(tac_id, params.num_tactics)
  tf.add_to_collection('label_tac_one_hot', label_tac_one_hot)

  # shape: [batch_size, hidden_size + num_tactics]
  net = tf.concat([net, tf.to_float(label_tac_one_hot)], axis=1)
  tf.add_to_collection('pfstate_and_tac', net)
  return net


def dilated_cnn_pairwise_encoder(features, labels, mode, params, config):
  """Dilated convolution network for goal_ids and thm_ids.

  Follows Estimator signature.

  Args:
    features: only the goal (represented as token ids) is used.
    labels: tactic id label is used for PARAMETERS_CONDITIONED_ON_TAC.
    mode: dropout only in mode train.
    params: hyperparameters
    config: configuration object

  Returns:
    encoding_distribution: A normal distribution that can be sampled from.
  """
  params = utils.Params(params)

  with tf.variable_scope('dilated_cnn_pairwise_encoder'):
    goal_net = dilated_cnn_goal_encoder(features, labels, mode, params, config)
    thm_net = dilated_cnn_thm_encoder(features, labels, mode, params, config)
    spec = EncodingSpec(pfstate_enc=goal_net, thm_enc=thm_net)

    # Concatenate theorem encoding, goal encoding, their dot product.
    # This attention-style concatenation performed well in language models.
    # Output shape: [batch_size, 3 * hidden_size]
    net = tf.concat([
        spec.pfstate_enc, spec.thm_enc,
        tf.multiply(spec.pfstate_enc, spec.thm_enc)
    ], -1)

    if params.parameters_conditioned_on_tac:
      # Concatenate one-hot encoding of tac_ids.
      net = _concat_net_tac_id(net, labels, params)

    if mode == TRAIN:
      net = tf.nn.dropout(net, rate=(1 - params.thm_keep_prob))
    net = tf.layers.dense(net, params.hidden_size, activation=tf.nn.relu)
    if mode == TRAIN:
      net = tf.nn.dropout(net, rate=(1 - params.thm_keep_prob))
    net = tf.layers.dense(net, params.hidden_size, activation=tf.nn.relu)
    if mode == TRAIN:
      net = tf.nn.dropout(net, rate=(1 - params.thm_keep_prob))
    net = tf.layers.dense(net, params.hidden_size, activation=tf.nn.relu)
    tf.add_to_collection('thm_goal_fc', net)

    return EncodingSpec(
        enc=net, dist=None, pfstate_enc=spec.pfstate_enc, thm_enc=spec.thm_enc)


def tactic_classifier(encoding_spec, labels, mode, params, config):
  """Given a proof state encoding, compute tactic logits."""
  del config  # Unused

  if encoding_spec.pfstate_enc is not None:
    # If negative examples were added, use only goal encodings.
    net = encoding_spec.pfstate_enc
  else:
    net = encoding_spec.enc

  # Shape: 2D [batch_size, hidden_size]
  tf.add_to_collection('tactic_net', net)

  if mode == TRAIN:
    net = tf.nn.dropout(net, rate=(1 - params.tac_keep_prob))
  net = tf.layers.dense(net, params.hidden_size, activation=tf.nn.relu)
  if mode == TRAIN:
    net = tf.nn.dropout(net, rate=(1 - params.tac_keep_prob))
  net = tf.layers.dense(net, params.hidden_size, activation=tf.nn.relu)
  if mode == TRAIN:
    net = tf.nn.dropout(net, rate=(1 - params.tac_keep_prob))
  tactic_logits = tf.layers.dense(net, params.num_tactics, activation=None)
  # Shape: 2D [batch_size, num_tactics]
  tf.add_to_collection('tactic_logits', tactic_logits)

  predictions = losses.tactic_predictions(tactic_logits, labels, mode, params)
  eval_metric_ops = losses.add_tactic_losses(predictions, labels, params)

  return predictions, eval_metric_ops


def pairwise_scorer(encoding_spec, labels, mode, params):
  """Given a (pfstate, thm) encoding, computes thm parameter scores."""
  del mode  # Unused in this scorer.

  net = encoding_spec.enc
  # Shape: 2D [batch_size, hidden_size]
  logits = tf.layers.dense(net, 1, activation=None)
  # Shape: 2D [batch_size, 1]
  tf.add_to_collection('pairwise_score', logits)

  predictions = losses.pairwise_predictions(logits, labels, params)
  eval_metric_ops = losses.add_pairwise_losses(predictions, params)

  return predictions, eval_metric_ops
