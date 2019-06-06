"""Loss and prediction functions for HOLparam models."""

from __future__ import absolute_import
from __future__ import division
# Import Type Annotations
from __future__ import print_function

import tensorflow as tf

from typing import Any
from typing import Dict
from typing import Text

FLAGS = tf.flags.FLAGS


def tf_reduce_mean_weighted(values, weights):
  values = tf.reshape(values, [-1])
  weights = tf.reshape(weights, [-1])
  return tf.divide(tf.tensordot(values, weights, 1), tf.reduce_sum(weights))


def tactic_predictions(tactic_logits, labels, unused_mode, params):
  """Compute tactic predictions and losses from logits."""
  tac_id = labels['tac_id']
  weights = tf.cast(labels['tac_present'], tf.float32)

  # Whether the target tactic in the top k.
  tactic_topk_accuracy = tf_reduce_mean_weighted(
      tf.to_float(tf.nn.in_top_k(tactic_logits, tac_id, params.topk)), weights)

  # Whether the target tactic the selected tactic.
  # TODO(smloos): Update tactic accuracy to be weighted by tactic frequency.
  choices_tactic = tf.argmax(tactic_logits, -1)
  correct_tactics = tf.equal(tf.to_int64(tac_id), choices_tactic)
  tactic_accuracy = tf_reduce_mean_weighted(
      tf.to_float(correct_tactics), weights)

  # Compute the log loss for the tactic logits.
  log_prob_tactic = tf.losses.sparse_softmax_cross_entropy(
      logits=tactic_logits, labels=tac_id, weights=weights)

  predictions = {
      'log_prob_tactic': log_prob_tactic,
      'tactic_accuracy': tactic_accuracy,
      'target_tactics': tac_id,
      'predicted_tactics': choices_tactic,
      'tactic_topk_accuracy': tactic_topk_accuracy,
  }
  return predictions


def add_tactic_losses(predictions, labels, params):
  """Add tactic losses to total loss, and log summaries."""
  # Add log loss to total loss
  tf.losses.add_loss(params.tac_scale * predictions['log_prob_tactic'])

  # Report scalar values (e.g. accuracy, log loss)
  scalars = ['log_prob_tactic', 'tactic_accuracy', 'tactic_topk_accuracy']
  eval_metric_ops = {}
  for key in scalars:
    tf.summary.scalar(key, predictions[key])
    # Add eval metric ops for each scalar.
    scoped_key = '%s/%s' % (tf.get_variable_scope().name, key)
    eval_metric_ops[scoped_key] = tf.metrics.mean(predictions[key])

  # Get the value of the predicted and target tactics
  tf.summary.tensor_summary('target_tactics', predictions['target_tactics'])
  tf.summary.tensor_summary('predicted_tactics',
                            predictions['predicted_tactics'])

  # Accuracy calculated for each tactic and then averaged.
  eval_metric_ops['mean_tactic_accuracy'] = tf.metrics.mean_per_class_accuracy(
      labels=labels['tac_id'],
      predictions=predictions['predicted_tactics'],
      num_classes=params.num_tactics)
  return eval_metric_ops


def pairwise_predictions(logits: tf.Tensor, labels: Dict[Text, tf.Tensor],
                         params) -> Dict[Text, tf.Tensor]:
  """Given logits for (goal, thm) pair, make predictions."""
  # ignore examples where tactic is not present
  logits = tf.boolean_mask(logits, labels['tac_present'])
  thm_labels = tf.boolean_mask(labels['thm_label'], labels['tac_present'])
  del labels

  if params.pairwise_ce_scale:
    # TODO(smloos): tactic losses are averaged, pairwise summed, make it
    # consistent.
    ce_loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.expand_dims(thm_labels, 1),
        logits=logits,
        reduction=tf.losses.Reduction.SUM)
  else:
    ce_loss = tf.constant(0.)
  pos_logits = tf.boolean_mask(tf.squeeze(logits), thm_labels)
  neg_logits = tf.boolean_mask(tf.squeeze(logits), 1 - thm_labels)
  pos_pred = tf.sigmoid(pos_logits)
  neg_pred = tf.sigmoid(neg_logits)
  pos_acc = tf.reduce_mean(tf.to_float(tf.greater(pos_pred, 0.5)))
  neg_acc = tf.reduce_mean(tf.to_float(tf.less(neg_pred, 0.5)))

  pos_copies = tf.tile(pos_logits, [params.ratio_neg_examples])
  relative_pred = tf.reduce_mean(
      tf.to_float(tf.greater(pos_copies, neg_logits)))
  predictions = {
      'pos_logits': tf.reduce_mean(pos_logits),
      'neg_logits': tf.reduce_mean(neg_logits),
      'pos_pred': tf.reduce_mean(pos_pred),
      'neg_pred': tf.reduce_mean(neg_pred),
      'relative_pred': relative_pred,
      'pos_accuracy': pos_acc,
      'neg_accuracy': neg_acc,
      'accuracy_50_50': (pos_acc + neg_acc) / 2.,
      'log_prob_pairwise': ce_loss,
  }
  return predictions


def add_pairwise_losses(predictions: Dict[Text, tf.Tensor],
                        params) -> Dict[Text, Any]:
  """Add losses from the predictions and return eval_metric_ops."""
  tf.losses.add_loss(params.pairwise_ce_scale *
                     predictions['log_prob_pairwise'])
  eval_metric_ops = {}
  for key in predictions:
    tf.summary.scalar(key, predictions[key])
    # Add eval metric ops for each scalar.
    scoped_key = tf.get_variable_scope().name + '/' + key
    eval_metric_ops[scoped_key] = tf.metrics.mean(predictions[key])

  return eval_metric_ops
