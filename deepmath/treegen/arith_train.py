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
"""Trains the model in arith_model.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from deepmath.treegen import arith_model
from deepmath.treegen import arith_utils
from deepmath.treegen import jsonl

flags = tf.flags
FLAGS = flags.FLAGS

# Hyperparameters
flags.DEFINE_integer('embedding_length', 5,
                     'How long to make the expression embedding vectors.')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate.')
flags.DEFINE_float('learning_rate_decay_factor', 0.98,
                   'Decay factor for learning rate.')
flags.DEFINE_integer('decay_steps', 10000, 'Number of steps between decays.')
flags.DEFINE_enum('optimizer', 'adam', ('sgd', 'adam', 'rmsprop'),
                  'Optimizer to use (either sgd, adam or rmsprop)')
flags.DEFINE_integer('max_steps', 1000000,
                     'The maximum number of batches to run the trainer for.')
flags.DEFINE_integer('batch_size', 100, 'How many samples to read per batch.')

# Train flags
flags.DEFINE_string('logdir', '/tmp/fol_generate',
                    'Directory in which to write event logs.')
flags.DEFINE_integer('ps_tasks', 0,
                     'The number of parameter servers. If the value is 0, then '
                     'the parameters are handled locally by the worker.')
flags.DEFINE_string('master', 'local',
                    'BNS name of the Tensorflow master to use.')
flags.DEFINE_integer('task', 0, 'Task ID of the replica running the training.')
flags.DEFINE_string('train_path', 'expressions.jsonl',
                    'JSON lines file containing the training dataset of '
                    'expressions.')


def main(unused_argv):
  train_iterator = jsonl.JSONLinesIterator(FLAGS.train_path)

  with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
    # Build the graph.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    m = arith_model.ArithmeticRecursiveGenerativeModel(FLAGS.embedding_length)

    variables = tf.trainable_variables()

    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,
                                               global_step,
                                               FLAGS.decay_steps,
                                               FLAGS.learning_rate_decay_factor,
                                               staircase=True)

    if FLAGS.optimizer == 'sgd':
      optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif FLAGS.optimizer == 'adam':
      optimizer = tf.train.AdamOptimizer(learning_rate)
    elif FLAGS.optimizer == 'rmsprop':
      optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                            decay=0.9,
                                            momentum=0.9,
                                            epsilon=1e-5)
    else:
      raise RuntimeError('Unknown optimizer %s' % FLAGS.optimizer)

    train_op = optimizer.minimize(m.loss,
                                  global_step=global_step,
                                  var_list=variables)

    supervisor = tf.Supervisor(is_chief=(FLAGS.task == 0),
                               logdir=FLAGS.logdir,
                               global_step=global_step,
                               save_model_secs=60,
                               summary_op=None)
    sess = supervisor.prepare_or_wait_for_session()

    # Run the trainer.
    for unused_i in xrange(FLAGS.max_steps):
      batch = [next(train_iterator) for _ in xrange(FLAGS.batch_size)]

      _, step, batch_loss, kl_loss = sess.run(
          [train_op, global_step, m.loss, m.kl_div_mean],
          feed_dict=m.build_feed_dict(batch))
      if step % 100 == 1:
        print('step=%d:  batch loss=%f, kl loss=%f' % (step, batch_loss,
                                                       kl_loss))

        exprs = m.sample_exprs(sess, 100)
        correct = np.sum(arith_utils.eval_expr(expr) == 0 for expr in exprs)
        print('Correct:', correct / 100.)
        for expr in exprs[:10]:
          print(arith_utils.stringify_expr(expr), arith_utils.eval_expr(expr))


if __name__ == '__main__':
  tf.app.run()
