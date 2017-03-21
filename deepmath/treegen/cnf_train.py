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
"""Trains the model in cnf_train.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import json
import textwrap
import traceback
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from deepmath.treegen import cnf_model
from deepmath.treegen import cnf_utils
from deepmath.treegen import jsonl
from deepmath.util import model_utils

metrics = tf.contrib.metrics
slim = tf.contrib.slim
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_enum('mode', 'train', ['train', 'eval'],
                  'Whether we are training or evaluating.')

flags.DEFINE_string('hparams', '', 'All hyperparameter name=value pairs.')
flags.DEFINE_string('tf_log_dir', '/tmp/cnf_generate',
                    'Directory in which to write event logs.')
flags.DEFINE_integer('ps_tasks', 0,
                     'The number of parameter servers. If the value is 0, then '
                     'the parameters are handled locally by the worker.')

flags.DEFINE_bool('sync_replicas', False, 'Use SyncReplicasOptimizer.')
flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')
flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'Number of replicas to aggregate in SyncReplicasOptimizer.')

flags.DEFINE_string('master', 'local',
                    'BNS name of the Tensorflow master to use.')
flags.DEFINE_integer('task', 0, 'Task ID of the replica running the training.')
flags.DEFINE_string('clauses', None, 'Dataset of CNF expressions to use.')
flags.DEFINE_string('clause_metadata', None, 'Metadata about clauses.')
flags.DEFINE_integer('save_model_secs', 600, 'Number of seconds between saves.')
flags.DEFINE_integer('save_summaries_secs', 30,
                     'Number of seconds between summaries.')

# Model parameters
flags.DEFINE_enum('model_type', 'tree', ['tree', 'seq'],
                  'Type of model to use.')

# Model sampling flags
flags.DEFINE_integer('num_summary_samples', 10,
                     'Number of sampled formulas to put in summaries.')
flags.DEFINE_string('sampling_temps', '1e-3,1',
                    'Temperatures at which to sample.')

# Eval flags
flags.DEFINE_string('eval_dir', None,
                    'Directory to load checkpoints from for evaluation.')
flags.DEFINE_integer('eval_lines', None,
                     'Number of lines to evaluate each time.')
flags.DEFINE_integer('eval_interval_secs', None,
                     'How often to run the evaluation.')

flags.mark_flag_as_required('clauses')
flags.mark_flag_as_required('clause_metadata')


def load_data(random_start):
  data_iterator = jsonl.JSONLinesIterator(FLAGS.clauses, random_start)
  with tf.gfile.GFile(FLAGS.clause_metadata) as f:
    clause_metadata = json.load(f)

  return data_iterator, clause_metadata


def train(hparams):
  """Run training loop."""
  data_iterator, clause_metadata = load_data(random_start=True)

  with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
    # The following three lines prevent hangs during distributed training.
    vs = tf.get_variable_scope()
    if vs.caching_device is None:
      vs.set_caching_device(lambda op: op.device)

    # Build the graph.
    global_step = slim.variables.get_or_create_global_step()
    if FLAGS.model_type == 'tree':
      m = cnf_model.CNFTreeModel(data_iterator, hparams, clause_metadata)
    else:
      m = cnf_model.CNFSequenceModel(data_iterator, hparams, clause_metadata)

    variables = tf.trainable_variables()

    learning_rate = tf.train.exponential_decay(
        hparams.learning_rate,
        global_step,
        hparams.decay_steps,
        hparams.learning_rate_decay_factor,
        staircase=True)

    if hparams.optimizer == 'sgd':
      optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif hparams.optimizer == 'adam':
      optimizer = tf.train.AdamOptimizer(learning_rate)
    elif hparams.optimizer == 'rmsprop':
      optimizer = tf.train.RMSPropOptimizer(
          learning_rate=learning_rate, decay=0.9, momentum=0.9, epsilon=1e-5)
    else:
      raise RuntimeError('Unknown optimizer %s' % hparams.optimizer)

    if FLAGS.master not in ('', 'local') and FLAGS.sync_replicas:
      replica_id = tf.constant(FLAGS.task, tf.int32, shape=())
      optimizer = tf.LegacySyncReplicasOptimizer(
          opt=optimizer,
          replicas_to_aggregate=FLAGS.replicas_to_aggregate,
          replica_id=replica_id,
          total_num_replicas=FLAGS.worker_replicas)

    tf.contrib.deprecated.scalar_summary('lr', learning_rate)
    tf.contrib.deprecated.scalar_summary('loss', m.loss)
    for metric_name, metric_value in m.metrics.items():
      tf.contrib.deprecated.scalar_summary('metric/' + metric_name,
                                           metric_value)

    grads_and_vars = optimizer.compute_gradients(m.loss, variables)
    if hparams.grad_max_norm > 0:
      g, v = zip(*grads_and_vars)
      g, global_norm = tf.clip_by_global_norm(g, hparams.grad_max_norm)
      tf.contrib.deprecated.scalar_summary('global_norm', global_norm)
      grads_and_vars = zip(g, v)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step)
    summary_op = tf.get_summary_op()

    if FLAGS.master not in ('', 'local') and FLAGS.sync_replicas:
      init_token_op = optimizer.get_init_tokens_op()
      chief_queue_runner = optimizer.get_chief_queue_runner()

    saver = tf.Saver(keep_checkpoint_every_n_hours=1.0)

    supervisor = tf.Supervisor(
        is_chief=(FLAGS.task == 0),
        logdir=FLAGS.tf_log_dir,
        global_step=global_step,
        saver=saver,
        # We are going to compute summaries ourselves.
        summary_op=None,
        save_model_secs=FLAGS.save_model_secs,
        # But we set this so that this computes global_step/sec.
        save_summaries_secs=FLAGS.save_summaries_secs)
    sess = supervisor.prepare_or_wait_for_session(FLAGS.master)

    # TODO(ricshin):
    # Rewrite this to use supervisor.managed_session().
    # Look at how slim/learning.py handles SyncReplicas, in particular
    # init_token_op.  Use normal text summaries once they exist.
    # Use supervisor.should_stop().
    if FLAGS.task == 0:
      if FLAGS.master not in ('', 'local') and FLAGS.sync_replicas:
        supervisor.start_queue_runners(sess, [chief_queue_runner])
        sess.run(init_token_op)

      sampling_temps = [float(x) for x in FLAGS.sampling_temps.split(',')]

      def summarize():
        try:
          summary_strs, global_step_val = sess.run([summary_op, global_step])
          summaries = tf.Summary.FromString(summary_strs)

          for i, temp in itertools.product(
              xrange(FLAGS.num_summary_samples), sampling_temps):
            cnf = textwrap.wrap(cnf_utils.unparse_cnf(m.sample(sess)))
            summaries.value.add(tag='formula_temp%g_%d' % (temp, i),
                                tensor=make_tensor_proto('\n'.join(cnf)))

          supervisor.summary_writer.add_summary(summaries.SerializeToString(),
                                                global_step_val)
          status_str = ', '.join('%s=%f' % (value.tag, value.simple_value)
                                 for value in summaries.value
                                 if value.HasField('simple_value'))
          tf.logging.info('step=%d: %s', global_step_val, status_str)
        except:
          # The supervisor eats the backtrace, so print it here.
          traceback.print_exc()
          raise

      supervisor.loop(FLAGS.save_summaries_secs, summarize)

    # Run the trainer.
    for unused_i in xrange(hparams.max_steps):
      sess.run(train_op)


def evaluate(hparams):
  """Evaluate a model under training repeatedly."""
  data_iterator, clause_metadata = load_data(random_start=False)

  if FLAGS.model_type == 'tree':
    m = cnf_model.CNFTreeModel(data_iterator, hparams, clause_metadata)
  else:
    m = cnf_model.CNFSequenceModel(data_iterator, hparams, clause_metadata)

  all_metrics = [m.loss] + m.metrics.values()
  mean_values, mean_updates = zip(*(metrics.streaming_mean(v)
                                    for v in all_metrics))
  tf.contrib.deprecated.scalar_summary('loss', mean_values[0])
  for i, metric_name in enumerate(m.metrics.iterkeys()):
    tf.contrib.deprecated.scalar_summary('metric/' + metric_name,
                                         mean_values[i + 1])

  num_evals = (FLAGS.eval_lines - 1) // hparams.batch_size + 1

  slim.evaluation.evaluation_loop(
      FLAGS.master,
      FLAGS.eval_dir,
      FLAGS.tf_log_dir,
      num_evals,
      eval_op=tf.group(*mean_updates),
      eval_interval_secs=FLAGS.eval_interval_secs,
      # This resets the data iterator to the beginning of the file, so that
      # exactly the same lines are evaluated each loop iteration.
      # A py_func must return something convertible to a tensor; reset() returns
      # None, and reset() or "" returns "".
      final_op=tf.py_func(lambda: data_iterator.reset() or '', [], [tf.string]))


def main(unused_argv):
  hparams = tf.contrib.training.HParams(
      learning_rate=0.01,
      learning_rate_decay_factor=0.98,
      grad_max_norm=3,
      decay_steps=10000,
      optimizer='adam',
      max_steps=1000000)

  if FLAGS.model_type == 'tree':
    hparams = model_utils.merge_hparams(
        hparams, cnf_model.CNFTreeModel.default_hparams())
  else:
    hparams = model_utils.merge_hparams(
        hparams, cnf_model.CNFSequenceModel.default_hparams())

  hparams.parse(FLAGS.hparams)

  if FLAGS.mode == 'train':
    train(hparams)
  elif FLAGS.mode == 'eval':
    evaluate(hparams)


if __name__ == '__main__':
  tf.app.run()
