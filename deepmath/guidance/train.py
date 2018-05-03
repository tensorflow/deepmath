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
"""Trainer for clause search models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import keras
import tensorflow as tf

from deepmath.util import model_utils

metrics = tf.contrib.metrics
slim = tf.contrib.slim
training = tf.contrib.training
flags = tf.flags
FLAGS = flags.FLAGS

# Logging and output flags
flags.DEFINE_string('logdir', None, 'Name of base log directory.')
flags.DEFINE_integer('save_summaries_secs', 0,
                     'Number of seconds between summary saves.')
flags.DEFINE_integer('save_checkpoint_secs', 60,
                     'Number of seconds between summary saves.')
flags.DEFINE_integer('save_summaries_steps', 1000,
                     'Number of seconds between summary saves.')
flags.DEFINE_integer('save_interval_secs', 600,
                     'Number of seconds between saves.')
flags.DEFINE_integer('eval_interval_secs', 600,
                     'Number of seconds between evaluations.')
flags.DEFINE_integer('log_every_n_iter', 100,
                     'Number of iterations between logging.')

# Distributed training flags
flags.DEFINE_string('super_master', '', 'Master argument for supervisor.')
flags.DEFINE_integer('task', 0,
                     'The Task ID. This flag is used when training with '
                     'multiple workers to identify each worker.')
flags.DEFINE_integer('ps_tasks', 0,
                     'The number of parameter servers. If the value is 0, then '
                     'the parameters are handled locally by the worker.')
flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')


def default_hparams():
  """Default values for training and evaluation hyperparameters."""
  return tf.contrib.training.HParams(
      # Training
      optimizer='adam',  # Optimizer to use (adam or rmsprop)
      learning_rate=0.0001,
      learning_rate_decay_factor=0.98,
      decay_steps=10000,  # Steps between learning rate decay
      gradient_clipping_norm=0.0,  # Norm to clip gradients to
      use_averages=False,  # Whether to do Polyak averaging
      moving_average_decay=0.9999,  # Decay for Polyak averaging
      label_smoothing=0.1,
      loss='xent',  # Loss type (xent or joint)
      seed=0,  # Graph-level random seed (0 for None)

      # Batch sizes and steps
      batch_size=32,  # Training batch size
      eval_batch_size=128,  # Evaluation batch size
      eval_examples=10000,  # Total examples per eval
      max_steps=10000000,  # Maximum number of training steps
      max_evals=10000000,  # Maximum number of eval steps

      # Synchronous training
      sync_replicas=False,  # Whether to use synchronous training
      backup_replicas=1,  # How many backup replicas to use for sync training
  )


def mode_dir(mode):
  """Returns the log directory for the given mode (train or eval)."""
  if mode not in ('train', 'eval'):
    raise ValueError('Invalid mode %r' % mode)
  if not FLAGS.logdir:
    raise ValueError('--logdir not specified')
  return os.path.join(FLAGS.logdir, mode)


def general_train(make_loss, hparams, make_hooks=None):
  """Trains a general model with a loss.

  Args:
    make_loss: Function which creates loss (and possibly registers accuracy
      summaries and other features).
    hparams: Hyperparameters (see default_hparams() for details).
    make_hooks: Optional, function which creates additional hooks for training.

  Returns:
    Final loss.

  Raises:
    ValueError: If flags are missing or invalid.
  """
  train_dir = mode_dir('train')
  if not tf.gfile.Exists(train_dir):
    tf.gfile.MakeDirs(train_dir)
  if hparams.seed:
    tf.set_random_seed(hparams.seed)

  # Configure keras
  keras.backend.set_learning_phase(1)
  keras.backend.manual_variable_initialization(True)

  with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks,
                                                merge_devices=True)):
    # Set the caching device to prevent hangs during distributed training
    vs = tf.get_variable_scope()
    if vs.caching_device is None:
      vs.set_caching_device(lambda op: op.device)

    # Grab loss and global step
    total_loss = make_loss()
    global_step = slim.get_or_create_global_step()

    # Set up Polyak averaging if desired
    if hparams.use_averages:
      moving_average_variables = tf.trainable_variables()
      moving_average_variables.extend(slim.losses.get_losses())
      moving_average_variables.append(total_loss)
      variable_averages = tf.train.ExponentialMovingAverage(
          hparams.moving_average_decay, global_step)
      # For sync_replicas, averaging happens in the chief queue runner
      if not hparams.sync_replicas:
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,
                             variable_averages.apply(moving_average_variables))
    else:
      variable_averages = None
      moving_average_variables = None

    # Decay learning rate exponentially
    learning_rate = tf.train.exponential_decay(
        hparams.learning_rate,
        global_step,
        hparams.decay_steps,
        hparams.learning_rate_decay_factor,
        staircase=True)
    tf.contrib.deprecated.scalar_summary('learning rate', learning_rate)

    # Create optimizer
    if hparams.optimizer == 'adam':
      optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-3)
    elif hparams.optimizer == 'rmsprop':
      optimizer = tf.train.RMSPropOptimizer(
          learning_rate=learning_rate, decay=0.9, momentum=0.9,
          epsilon=1e-5)
    else:
      raise ValueError('Unknown optimizer %s' % hparams.optimizer)

    is_chief = FLAGS.task == 0
    chief_only_hooks = []

    hooks = [tf.train.LoggingTensorHook({
        'global_step': global_step,
        'total_loss': total_loss
    }, every_n_iter=FLAGS.log_every_n_iter),
             tf.train.NanTensorHook(total_loss),
             tf.train.StopAtStepHook(hparams.max_steps),
            ]

    if make_hooks is not None:
      hooks.extend(make_hooks())

    # If desired, optimize synchronously
    if hparams.sync_replicas:
      optimizer = tf.SyncReplicasOptimizer(
          optimizer=optimizer,
          replicas_to_aggregate=FLAGS.worker_replicas - hparams.backup_replicas,
          variable_averages=variable_averages,
          variables_to_average=moving_average_variables,
          replica_id=FLAGS.task,
          total_num_replicas=FLAGS.worker_replicas)
      sync_replicas_hook = optimizer.make_session_run_hook(is_chief)
      hooks.append(sync_replicas_hook)

    # Train
    train_tensor = slim.learning.create_train_op(
        total_loss, optimizer,
        clip_gradient_norm=hparams.gradient_clipping_norm)
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=2)

    scaffold = tf.train.Scaffold(saver=saver)

    if FLAGS.save_summaries_secs > 0:
      save_summaries_secs = FLAGS.save_summaries_secs
      save_summaries_steps = None
    else:
      save_summaries_steps = FLAGS.save_summaries_steps
      save_summaries_secs = None
    with tf.train.MonitoredTrainingSession(
        master=FLAGS.super_master,
        is_chief=is_chief,
        hooks=hooks,
        chief_only_hooks=chief_only_hooks,
        checkpoint_dir=train_dir,
        scaffold=scaffold,
        save_checkpoint_secs=FLAGS.save_checkpoint_secs,
        save_summaries_secs=save_summaries_secs,
        save_summaries_steps=save_summaries_steps) as mon_sess:
      while not mon_sess.should_stop():
        mon_sess.run(train_tensor)


def sigmoid_train(make_model, hparams, joint_safe=False):
  """Trains a model with sigmoid loss.

  Args:
    make_model: Function returning `(logits, labels)`, where logits
      is a 1-D tensor of logits and labels is a 1-D tensor of bools.
    hparams: Hyperparameters (see default_hparams() for details).
    joint_safe: Whether the examples are pairs of true, false.

  Returns:
    Final loss.

  Raises:
    ValueError: If flags are missing or invalid.
  """
  def make_loss():
    """Builds a model and computes losses."""
    # Generate model
    logits, labels = make_model()
    if joint_safe:
      joint_logits, joint_labels = model_utils.paired_joint_logits_and_labels(
          logits, labels)

    # Compute loss
    if hparams.loss == 'xent':
      slim.losses.sigmoid_cross_entropy(
          logits[:, None], labels[:, None],
          label_smoothing=hparams.label_smoothing)
    elif hparams.loss == 'joint':
      if not joint_safe:
        raise ValueError('joint loss needs joint_safe=True')
      slim.losses.sigmoid_cross_entropy(
          joint_logits[:, None], joint_labels[:, None],
          label_smoothing=hparams.label_smoothing)
    else:
      raise ValueError('Unknown loss %r' % hparams.loss)
    total_loss = slim.losses.get_total_loss()
    tf.contrib.deprecated.scalar_summary('total loss', total_loss)

    # Measure discrete accuracy
    predictions = logits > 0
    accuracy = metrics.accuracy(predictions, labels)
    tf.contrib.deprecated.scalar_summary('train accuracy', accuracy)

    # Measure joint accuracy
    if joint_safe:
      joint_predictions = joint_logits > 0
      joint_accuracy = metrics.accuracy(joint_predictions, joint_labels)
      tf.contrib.deprecated.scalar_summary('train joint accuracy',
                                           joint_accuracy)

    # All done!
    return total_loss

  return general_train(make_loss, hparams)


def general_eval(make_metrics, hparams, make_hooks=None, eval_dir=None):
  """Evaluate a general model.

  Args:
    make_metrics: Function returning a list of `(accuracies, updates)` pairs
      as produced by the metrics module (see `sigmoid_eval` for examples).
    hparams: Hyperparameters.
    make_hooks: Optional, function to create additional hooks.
    eval_dir: Optional string for the directory where the evaluation results
              should be written to.

  Raises:
    ValueError: If flags are missing or invalid.
  """
  if hparams.use_averages:
    raise NotImplementedError('Figure out how to eval with Polyak averaging')
  train_dir = mode_dir('train')
  if eval_dir is None:
    eval_dir = mode_dir('eval')

  if not tf.gfile.Exists(eval_dir):
    tf.gfile.MakeDirs(eval_dir)
  if hparams.seed:
    tf.set_random_seed(hparams.seed)
  num_evals = (hparams.eval_examples - 1) // hparams.eval_batch_size + 1

  # Configure keras
  keras.backend.set_learning_phase(0)
  keras.backend.manual_variable_initialization(True)

  hooks = [
      training.StopAfterNEvalsHook(num_evals),
      training.SummaryAtEndHook(
          log_dir=eval_dir, summary_op=tf.summary.merge_all())
  ]

  with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks,
                                                merge_devices=True)):
    # Generate metrics
    with tf.variable_scope(tf.get_variable_scope()):
      metric_ops = make_metrics()

    # Evaluate
    accuracies, updates = zip(*metric_ops)

    if make_hooks is not None:
      with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        hooks.extend(make_hooks())

    training.evaluate_repeatedly(
        checkpoint_dir=train_dir,
        master=FLAGS.super_master,
        eval_ops=updates,
        final_ops=accuracies,
        eval_interval_secs=FLAGS.eval_interval_secs,
        hooks=hooks,
        max_number_of_evaluations=hparams.max_evals)


def sigmoid_eval(make_model, hparams, joint_safe=False):
  """Evaluate a model with sigmoid loss.

  Args:
    make_model: Function returning `(logits, labels)`, where logits
      is a 1-D tensor of logits and labels is a 1-D tensor of bools.
    hparams: Hyperparameters.
    joint_safe: Whether the examples are pairs of true, false.

  Raises:
    ValueError: If flags are missing or invalid.
  """
  def make_metrics():
    """Builds a model and computes metrics."""
    # Generate model
    logits, labels = make_model()
    if joint_safe:
      joint_logits, joint_labels = model_utils.paired_joint_logits_and_labels(
          logits, labels)

    # Measure accuracy
    metric_ops = [metrics.streaming_accuracy(logits > 0, labels)]
    tf.contrib.deprecated.scalar_summary('eval accuracy', metric_ops[-1][0])

    # Measure joint accuracy
    if joint_safe:
      metric_ops.append(metrics.streaming_accuracy(joint_logits > 0,
                                                   joint_labels))
      tf.contrib.deprecated.scalar_summary('eval joint accuracy',
                                           metric_ops[-1][0])
    return metric_ops

  general_eval(make_metrics, hparams)
