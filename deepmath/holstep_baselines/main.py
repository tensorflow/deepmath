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
r"""Training / evaluation routine.

Example usage:

python main.py \
--model_name=cnn_2x_siamese \
--task_name=conditioned_classification \
--logdir=experiments/cnn_2x_siamese_experiment \
--source_dir=~/Downloads/holstep
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import keras
import tensorflow as tf

from deepmath.holstep_baselines import conditioned_classification_models
from deepmath.holstep_baselines import data_utils
from deepmath.holstep_baselines import unconditioned_classification_models

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('source_dir',
                           'hol-ml-dataset',
                           'Directory where the raw data is located.')
tf.app.flags.DEFINE_string('logdir',
                           '/tmp/hol',
                           'Base directory for saving models and metrics.')
tf.app.flags.DEFINE_string('model_name',
                           'cnn_2x',
                           'Name of model to train.')
tf.app.flags.DEFINE_string('task_name',
                           'unconditioned_classification',
                           'Name of task to run: "conditioned_classification" '
                           'or "unconditioned_classification".')
tf.app.flags.DEFINE_string('tokenization',
                           'char',
                           'Type of statement tokenization to use: "char" or '
                           '"token".')
tf.app.flags.DEFINE_integer('batch_size', 64,
                            'Size of a batch.')
tf.app.flags.DEFINE_integer('max_len', 512,
                            'Maximum length of input statements.')
tf.app.flags.DEFINE_integer('samples_per_epoch', 128000,
                            'Number of random step statements to draw for '
                            'training at each epoch.')
tf.app.flags.DEFINE_integer('val_samples', 246912,
                            'Number of (ordered) step statements to draw for '
                            'validation.')
tf.app.flags.DEFINE_integer('epochs', 40,
                            'Number of epochs to train.')
tf.app.flags.DEFINE_integer('verbose', 1,
                            'Verbosity mode (0, 1 or 2).')
tf.app.flags.DEFINE_string('checkpoint_path',
                           '',
                           'Path to checkpoint to (re)start from.')
tf.app.flags.DEFINE_integer('data_parsing_workers', 4,
                            'Number of threads to use to generate input data.')


def main(_):
  logging.basicConfig(level=logging.DEBUG)
  if not os.path.exists(FLAGS.logdir):
    os.makedirs(FLAGS.logdir)

  if FLAGS.tokenization == 'token':
    use_tokens = True
  elif FLAGS.tokenization == 'char':
    use_tokens = False
  else:
    raise ValueError('Unknown tokenization mode:', FLAGS.tokenization)

  # Parse the training and validation data.
  parser = data_utils.DataParser(FLAGS.source_dir, use_tokens=use_tokens,
                                 verbose=FLAGS.verbose)

  # Print useful stats about the parsed data.
  if FLAGS.verbose:
    logging.info('Training data stats:')
    parser.display_stats(parser.train_conjectures)
    logging.info('---')
    logging.info('Validation data stats:')
    parser.display_stats(parser.val_conjectures)

  voc_size = len(parser.vocabulary) + 1

  if FLAGS.task_name == 'conditioned_classification':
    # Get the function for building the model, and the encoding to use.
    make_model, encoding = conditioned_classification_models.MODELS.get(
        FLAGS.model_name, None)
    if not make_model:
      raise ValueError('Unknown model:', FLAGS.model_name)

    # Instantiate a generator that will yield batches of training data.
    train_generator = parser.training_steps_and_conjectures_generator(
        encoding=encoding, max_len=FLAGS.max_len,
        batch_size=FLAGS.batch_size)

    # Instantiate a generator that will yield batches of validation data.
    val_generator = parser.validation_steps_and_conjectures_generator(
        encoding=encoding, max_len=FLAGS.max_len,
        batch_size=FLAGS.batch_size)

  elif FLAGS.task_name == 'unconditioned_classification':
    make_model, encoding = unconditioned_classification_models.MODELS.get(
        FLAGS.model_name, None)
    if not make_model:
      raise ValueError('Unknown model:', FLAGS.model_name)

    train_generator = parser.training_steps_generator(
        encoding=encoding, max_len=FLAGS.max_len,
        batch_size=FLAGS.batch_size)

    val_generator = parser.validation_steps_generator(
        encoding=encoding, max_len=FLAGS.max_len,
        batch_size=FLAGS.batch_size)

  else:
    raise ValueError('Unknown task_name:', FLAGS.task_name)

  if FLAGS.checkpoint_path:
    # Optionally load an existing saved model.
    model = keras.models.load_model(FLAGS.checkpoint_path)
  else:
    # Instantiate a fresh model.
    model = make_model(voc_size, FLAGS.max_len)
    model.summary()
    model.compile(optimizer=keras.optimizers.RMSprop(lr=0.0001),
                  loss='binary_crossentropy',
                  metrics=['acc'])

  # Define a callback for saving the model to the log directory.
  checkpoint_path = os.path.join(FLAGS.logdir, FLAGS.model_name + '.h5')
  checkpointer = keras.callbacks.ModelCheckpoint(
      checkpoint_path, save_best_only=True)

  # Define a callback for writing TensorBoard logs to the log directory.
  tensorboard_vis = keras.callbacks.TensorBoard(log_dir=FLAGS.logdir)

  logging.info('Fit model...')
  history = model.fit_generator(train_generator,
                                samples_per_epoch=FLAGS.samples_per_epoch,
                                validation_data=val_generator,
                                nb_epoch=FLAGS.epochs,
                                nb_val_samples=FLAGS.val_samples,
                                pickle_safe=True,
                                nb_worker=FLAGS.data_parsing_workers,
                                verbose=FLAGS.verbose,
                                callbacks=[checkpointer, tensorboard_vis])

  # Save training history to a JSON file.
  f = open(os.path.join(FLAGS.logdir, 'history.json'), 'w')
  f.write(json.dumps(history.history))
  f.close()


if __name__ == '__main__':
  tf.app.run()
