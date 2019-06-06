"""Implementation of Holparam experiments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import pickle
import numpy as np
import tensorflow as tf

from deepmath.deephol.train import architectures
from deepmath.deephol.train import data
from deepmath.deephol.train import model
from deepmath.deephol.train import utils

tf.flags.DEFINE_string(
    'hparams', '',
    'Comma-separated list of `name=value` hyperparameter values.')

tf.flags.DEFINE_integer('save_checkpoints_steps', 10000,
                        'Steps between each checkpoint save.')

tf.flags.DEFINE_integer('keep_checkpoint_max', 10,
                        'Maximum number of checkpoints kept.')

tf.flags.DEFINE_integer('save_summary_steps', 200, 'Steps between summaries.')

tf.flags.DEFINE_string('dataset_dir', None,
                       'Directory containing train, valid, and test examples.')

tf.flags.DEFINE_string('eval_dataset_dir', None,
                       'Optional directory validation examples for eval.')

tf.flags.DEFINE_string('model_dir', None,
                       'The directory where the model will be stored.')

tf.flags.DEFINE_string(
    'loop_dir', None,
    'Directory containing examples generated during DeepHOL prover loop.')

tf.flags.DEFINE_integer(
    'topk', 5, 'Check if the correct tactic is in the topk predicted.')

tf.flags.DEFINE_enum(
    'model', 'PAIR_TAC', ['PAIR_TAC', 'PARAMETERS_CONDITIONED_ON_TAC'],
    'Select the training task and architecture for the network.\n'
    'PAIR_TAC: Pairwise score on thm,goal pairs, tactic classifier on goals.\n'
    'PARAMETERS_CONDITIONED_ON_TAC: Pairwise score depends on tactic label.\n')

tf.flags.DEFINE_integer(
    'max_steps', None,
    'Maximum number of steps to run the training for. default: 0 (no limit).')

tf.flags.DEFINE_integer(
    'max_threads', 0,
    'Maximum number of threads to run. default: 0 (system picks appropriate '
    'number).')

FLAGS = tf.flags.FLAGS

TRAIN = tf.estimator.ModeKeys.TRAIN
EVAL = tf.estimator.ModeKeys.EVAL
PREDICT = tf.estimator.ModeKeys.PREDICT


def get_empty_string_thm_emb(hparams):
  """Gets the thm_embedding of '' for histdependent experiments."""
  filepath = os.path.join(hparams.dataset_dir, hparams.thm_asmpt_list_file)
  for serialized_example in tf.compat.v1.io.tf_record_iterator(filepath):
    example = tf.train.Example()
    example.ParseFromString(serialized_example)
    thm_embedding = example.features.feature['thm_embedding'].float_list.value
    # First record belongs to the empty string, return
    return np.array(thm_embedding, dtype=np.float32)


def get_input_fns(params):
  """Gets the train, eval, and eval on train data input fns."""
  dataset_dir = params.dataset_dir
  tf.logging.info('Using dataset_dir %r', dataset_dir)

  with tf.name_scope('data'):
    dataset_train_fn = functools.partial(data.get_holparam_dataset, TRAIN)
    dataset_eval_fn = functools.partial(data.get_holparam_dataset, EVAL)

    train_input_fn = data.get_input_fn(
        dataset_train_fn,
        TRAIN,
        params,
        shuffle_queue=params.shuffle_queue,
        parser=params.train_parser,
        filt=params.train_filter)
    eval_input_fn = data.get_input_fn(
        dataset_eval_fn,
        EVAL,
        params,
        parser=params.eval_parser,
        filt=params.eval_filter)

    return train_input_fn, eval_input_fn


def train_and_eval(params):
  """Creates train_and_eval_fn for estimator framework."""
  train_input_fn, eval_input_fn = get_input_fns(params)

  session_config = tf.ConfigProto(
      intra_op_parallelism_threads=FLAGS.max_threads)
  run_config = tf.estimator.RunConfig(
      model_dir=FLAGS.model_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      keep_checkpoint_max=FLAGS.keep_checkpoint_max,
      save_summary_steps=FLAGS.save_summary_steps,
      session_config=session_config)

  estimator = tf.estimator.Estimator(
      model_fn=model.model_fn, params=params, config=run_config)

  train_spec = tf.estimator.TrainSpec(
      input_fn=train_input_fn, max_steps=FLAGS.max_steps)

  # Create feature_spec and label_spec for exporting eval graph.
  # Currently we ignore these placeholders for serving and use only feed_dict.
  feature_spec = {
      'goal': tf.placeholder(dtype=tf.string, shape=[None]),
      'thms': tf.placeholder(dtype=tf.string, shape=[None]),
      'thms_hard_negatives': tf.placeholder(dtype=tf.string, shape=[None])
  }
  label_spec = {'tac_id': tf.placeholder(dtype=tf.int64, shape=[None])}
  build_input = tf.contrib.estimator.build_raw_supervised_input_receiver_fn
  input_receiver_fn = build_input(feature_spec, label_spec)
  exporter = tf.estimator.BestExporter(
      name='best_exporter',
      event_file_pattern='best_eval/*.tfevents.*',
      exports_to_keep=5,
      serving_input_receiver_fn=input_receiver_fn)
  eval_spec = tf.estimator.EvalSpec(
      name='continuous_on_eval',
      input_fn=eval_input_fn,
      steps=None,
      exporters=exporter)
  # TODO(smloos): Find a place for this so that it is called only once.
  # Attempt to output the experiment params
  if not tf.gfile.Exists(FLAGS.model_dir):
    tf.gfile.MakeDirs(FLAGS.model_dir)
  save_file = os.path.join(FLAGS.model_dir, 'params.pkl')
  with tf.gfile.Open(save_file, 'w') as f:
    tf.logging.info('Saving params pickle: %s', save_file)
    # Pickle the params using the highest protocol available
    pickle.dump(params, f, -1)
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def main(argv):
  del argv  # Unused.

  tf.logging.set_verbosity(tf.logging.INFO)

  hparams = tf.contrib.training.HParams(
      text_summaries=False,
      dataset_dir=FLAGS.dataset_dir,
      # Eval set can optionally be different from the training set.
      eval_dataset_dir=FLAGS.eval_dataset_dir,
      variable_av_decay=0.0,  # NOTE: not compatible with model_head function.
      ### Parameters for using looped data (generated in proof search)
      shuffle_queue=50000,
      negative_example_shuffle_buffer=1000,
      # TODO(smloos): Parameter below is not used anywhere, delete it or use it.
      neg_thms_shuffle_queue=1000,
      ### Network size parameters
      word_embedding_size=128,  # Word embedding dimension
      vocab_size=2044 + 4,  # Use most common vocab words + 4 special tokens
      truncate_size=1000,  # Max number of tokens per term (goal/theorem)
      num_tactics=41,  # Number of tactics
      hidden_size=128,  # Encoding size of theorems and goals
      final_size=128,  # Size of the dense layers on top of the wavenet decoder.
      hidden_tactic_size=128,  # Size of the hidden layers for choosing tactics
      wavenet_depth=128,  # Depth of hidden wavenet layers
      wavenet_layers=4,  # Number of layers per wavenet block
      wavenet_blocks=2,  # Number of wavenet blocks
      ### Proof state parser+network parameters
      asl_length=8,  # Number of assumptions considered
      proofstate_enc_size=256,  # Size of the final dense layers in encoders
      conv_layers=2,  # Number of conv layers in conv or att encoder
      conv_filters=128,  # Number of convolutional filters to learn
      conv_kernelsize=4,  # Receptive field of convolutions
      dense_layers=3,  # Number of dense layers on top of asl in att encoder
      att_trainable_keys=1,  # Number of trainable attention keys
      ### Loss coefficients
      thm_scale=0.2,  # Scaling factor for the log loss encoding prediction.
      tac_scale=1.0,  # Scaling factor for the log loss tactic prediction.
      topk_scale=2.0,  # Scaling factor for the weighted topk tactic prediction.
      pairwise_ce_scale=0.5,
      pairwise_roc_batch_scale=4.0,
      pairwise_roc_goal_scale=4.0,
      att_key_sim_scale=0.0,  # Scaling factor for the attention key similarity.
      beta=0.001,  # Scaling factor for the information bottleneck, if used.
      ### Training parameters
      batch_size=4,
      # Integer multiple ratio neg_examples:positives.
      ratio_neg_examples=7,
      # Multiple of positives, <= ratio_neg_examples.
      ratio_max_hard_negative_examples=5,
      learning_rate=0.0001,
      enc_keep_prob=0.7,  # Parameter for dropout in proof state encoding.
      fc_keep_prob=0.7,  # Parameter for dropout in thm,goal concatentation.
      tac_keep_prob=0.7,  # Parameter for dropout in predicting tactics.
      thm_keep_prob=0.7,  # Parameter for dropout in predicting parameter score.
      input_keep_prob=1.0,
      layer_keep_prob=1.0,
      layer_comb_weight=1.0,
      decay_rate=0.98,
      topk=FLAGS.topk,
      goal_vocab='vocab_goal_ls.txt',
      thm_vocab='vocab_thms_ls.txt',
      replace_generic_variables_and_types=True,
      thm_asmpt_list_file='thm_asmpt_list_train.tfrecord',
      empty_string_thm_emb=None,
      ### Architecture components
      train_parser=None,
      eval_parser=None,
      train_filter=None,
      eval_filter=None,
      encoder=None,
      regularizer=None,
      bottleneck=None,
      classifier=None,
      pairwise_scorer=None,
      decoder=None,
      model_head=None,
      # Condition parameter selection on tactic (PARAMETERS_CONDITIONED_ON_TAC).
      parameters_conditioned_on_tac=False,
  )
  hparams.parse(FLAGS.hparams)
  if not (hparams.ratio_max_hard_negative_examples <=
          hparams.ratio_neg_examples):
    raise ValueError('failed: max_hard_negative_examples <= ratio_neg_examples')

  if hparams.dataset_dir is None:
    raise ValueError('failed: dataset_dir must be set.')

  with tf.gfile.Open(
      os.path.join(hparams.dataset_dir, hparams.goal_vocab), 'r') as f:
    vocab_len = len(f.readlines())
  if hparams.thm_vocab:
    with tf.gfile.Open(
        os.path.join(hparams.dataset_dir, hparams.thm_vocab), 'r') as f:
      thm_vocab_len = len(f.readlines())
    vocab_len = max(vocab_len, thm_vocab_len)
  hparams.vocab_size = min(hparams.vocab_size, vocab_len)
  tf.logging.info('Vocabulary size is: %d', hparams.vocab_size)

  if (FLAGS.model == 'PAIR_TAC' or
      FLAGS.model == 'PARAMETERS_CONDITIONED_ON_TAC'):
    hparams.train_parser = data.pairwise_thm_parser
    hparams.eval_parser = data.pairwise_thm_parser
    hparams.encoder = architectures.dilated_cnn_pairwise_encoder
    hparams.classifier = architectures.tactic_classifier
    hparams.pairwise_scorer = architectures.pairwise_scorer
    if FLAGS.model == 'PARAMETERS_CONDITIONED_ON_TAC':
      hparams.parameters_conditioned_on_tac = True

  params = utils.Params(**hparams.values())
  tf.logging.info('Using params %r', params)
  train_and_eval(params)


if __name__ == '__main__':
  tf.app.run(main)
