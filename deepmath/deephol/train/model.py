"""Instantiates the architectures according to params to obtain a model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from deepmath.deephol.train import extractor
from deepmath.deephol.train import utils

FLAGS = tf.flags.FLAGS


def model_fn(features, labels, mode, params, config):
  """Implements the ModelFn."""
  params = utils.Params(params)
  tf.logging.info('Creating Estimator in mode %s', mode)

  with tf.variable_scope('extractor'):
    extr = params.setdefault('extractor',
                             extractor.Extractor(params).get_extractor())
    features, labels = extr(features, labels)

  eval_metric_ops = {}
  with tf.variable_scope('encoder'):
    encoding_spec = params.encoder(features, labels, mode, params, config)
    tf.add_to_collection('encoding_net', encoding_spec.enc)
    if encoding_spec.att_key_sim is not None:
      eval_metric_ops.update(encoding_spec.att_key_sim)

  predictions = {}
  if params.classifier is not None:
    tf.logging.info('Using classifier')
    with tf.variable_scope('classifier'):
      p, e = params.classifier(encoding_spec, labels, mode, params, config)
      predictions.update(p)
      eval_metric_ops.update(e)

  if params.pairwise_scorer is not None:
    tf.logging.info('Using pairwise scorer.')
    with tf.variable_scope('pairwise_scorer'):
      p, e = params.pairwise_scorer(encoding_spec, labels, mode, params)
      predictions.update(p)
      eval_metric_ops.update(e)

  loss = tf.losses.get_total_loss()
  tf.summary.scalar('loss', loss)

  if mode == tf.estimator.ModeKeys.TRAIN:
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(
        learning_rate=params.learning_rate,
        global_step=global_step,
        decay_steps=100000,
        decay_rate=params.decay_rate)
    tf.summary.scalar('learning_rate', learning_rate)

    # Polyak averaging improves predictions at eval time.
    opt = tf.train.AdamOptimizer(learning_rate)
    if params.variable_av_decay > 0:
      opt = tf.contrib.opt.MovingAverageOptimizer(
          opt, average_decay=params.variable_av_decay)
    train_op = opt.minimize(loss, global_step=global_step)
    if params.variable_av_decay > 0:
      scaffold = tf.train.Scaffold(saver=opt.swapping_saver())
    else:
      scaffold = tf.train.Scaffold()
  else:
    train_op = None
    scaffold = None

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      eval_metric_ops=eval_metric_ops,
      train_op=train_op,
      scaffold=scaffold)
