"""Various Apache Beam pipeline pieces for the prove-train loop.

This file contains all Apache Beam pipelines specific to the prove-train
loops.
"""
from __future__ import absolute_import
from __future__ import division
# Import Type Annotations
from __future__ import print_function
import io
import os
import random
import time
import apache_beam as beam
import numpy as np
from tensorflow import gfile
from tensorflow import logging
from typing import List, Tuple
from google.protobuf import text_format
from deepmath.deephol import deephol_pb2
from deepmath.deephol import embedding_store
from deepmath.deephol import holparam_predictor
from deepmath.deephol import io_util
from deepmath.deephol import prover_util
from deepmath.deephol.deephol_loop import checkpoint_monitor
from deepmath.deephol.deephol_loop import loop_meta
from deepmath.deephol.deephol_loop import loop_pb2
from deepmath.deephol.deephol_loop import options_pb2
from deepmath.deephol.deephol_loop import prooflog_to_tfexamples_lib
from deepmath.deephol.deephol_loop import prover_runner
from deepmath.deephol.deephol_loop import report
from deepmath.deephol.deephol_loop.missing import recordio
from deepmath.deephol.deephol_loop.missing import runner
from deepmath.deephol.deephol_loop.missing import sstableio
from deepmath.proof_assistant import proof_assistant_pb2

BATCH_SIZE = 128


class EmbeddingComputerDoFn(beam.DoFn):
  """Beam DoFn for computing embeddings."""

  def __init__(self, chkpt: str, batch_size: int,
               theorem_db: proof_assistant_pb2.TheoremDatabase):
    self.chkpt = chkpt
    self.theorem_db = theorem_db
    self.batch_size = batch_size

  def start_bundle(self):
    logging.info('Initializing the batching predictor...')
    self.predictor = holparam_predictor.HolparamPredictor(
        self.chkpt, max_embedding_batch_size=self.batch_size)
    logging.info('Initializing the embedding store...')
    self.emb_store = embedding_store.TheoremEmbeddingStore(self.predictor)

  def process(self, dummy_arg: str) -> List[bytes]:
    start_time = time.time()
    self.emb_store.compute_embeddings_for_thms_from_db(self.theorem_db)
    elapsed_time = time.time() - start_time
    logging.info('Embeddings computation time: %.1f', elapsed_time)
    result = io.BytesIO()
    np.save(result, self.emb_store.thm_embeddings)
    return [result.getvalue()]


def key_value_of_proto(proto):
  value = proto.SerializeToString()
  key = hash(value)
  return ('%x' % key, value)


class ProofLogToTFExamplesDoFn(beam.DoFn):
  """DoFn for converting proof logs to tf examples."""

  def __init__(self, tactics_filename: str,
               theorem_db: proof_assistant_pb2.TheoremDatabase,
               options: options_pb2.ConvertorOptions):
    options.tactics_path = tactics_filename
    self.converter = prooflog_to_tfexamples_lib.create_processor(
        options=options, theorem_database=theorem_db)

  def start_bundle(self):
    pass

  def process(self, proof_log: deephol_pb2.ProofLog) -> List[Tuple[int, str]]:
    return [
        key_value_of_proto(example)
        for example in self.converter.process_proof_log(proof_log)
    ]


def get_random_tactic_probability(config: loop_pb2.LoopConfig,
                                  current_round: int) -> float:
  if config.random_tactic_num_rounds == 0:
    return 0.0
  if current_round >= config.random_tactic_num_rounds:
    return config.random_tactic_probability_min
  decay_per_round = ((1.0 - config.random_tactic_probability_min) /
                     float(config.random_tactic_num_rounds))
  return 1.0 - float(current_round) * decay_per_round


class LoopPipeline(object):
  """High level class for controlling the main Loop."""

  def __init__(self, lm: loop_meta.LoopMeta, config: loop_pb2.LoopConfig):
    self.loop_meta = lm
    self.config = config
    self.checkpoint_monitor = checkpoint_monitor.CheckpointMonitor(
        config.path_model_directory, self.loop_meta.checkpoints_path())
    self.theorem_db = io_util.load_theorem_database_from_file(
        str(config.prover_options.path_theorem_database))
    if not self.theorem_db.HasField('name'):
      self.theorem_db.name = 'default'
    tasks_file = None
    if config.HasField('prover_tasks_file'):
      tasks_file = config.prover_tasks_file
    self.prover_tasks = prover_util.get_task_list(
        tasks_file, None, None, self.theorem_db,
        config.prover_options.splits_to_prove,
        set(config.prover_options.library_tags))
    logging.info('Number of tasks: %d', len(self.prover_tasks))

  def create_prover_tasks(self):
    num_selections = self.config.num_prover_tasks_per_round
    if num_selections > len(self.prover_tasks):
      return self.prover_tasks
    else:
      return random.sample(self.prover_tasks, num_selections)

  def prover_pipeline(self, tasks: List[proof_assistant_pb2.ProverTask], root):
    """Make a prover pipeline for the given task and this round."""
    current_round = self.loop_meta.status.current_round
    prover_options = deephol_pb2.ProverOptions()
    prover_options.CopyFrom(self.config.prover_options)
    prover_options.action_generator_options.random_tactic_probability = (
        get_random_tactic_probability(self.config, current_round))
    checkpoint = self.checkpoint_monitor.get_checkpoint()
    assert checkpoint, 'Model checkpoint is not present.'
    # Update prover options to utilize the latest checkpoint present. We also
    # make sure to utilize the embedding store as well.
    prover_options.path_model_prefix = checkpoint
    prover_options.theorem_embeddings = checkpoint + '.npy'
    assert gfile.Exists(
        prover_options.theorem_embeddings), ('Missing embeddings file "%s".' %
                                             prover_options.theorem_embeddings)
    output_dir = self.loop_meta.make_proof_logs_dir(current_round)
    output_prefix = os.path.join(output_dir, 'logs')
    logging.info('Prover options:\n%s',
                 text_format.MessageToString(prover_options))
    io_util.write_text_proto(
        str(os.path.join(output_dir, 'prover_options.pbtxt')), prover_options)

    return prover_runner.make_pipeline(tasks, prover_options, output_prefix)(
        root)

  def training_examples_pipeline(self, proof_logs):
    """Create the pipeline to convert ProofLogs to Examples.

    Args:
      proof_logs: beam node for the proof logs.
    """
    fresh_examples = proof_logs | ('ConvertToTFExamples' >> beam.ParDo(
        ProofLogToTFExamplesDoFn(
            str(self.config.prover_options.path_tactics), self.theorem_db,
            self.config.convertor_options)))
    current_round = self.loop_meta.status.current_round
    output_dir = self.loop_meta.make_training_examples_dir(current_round)
    fresh_dir = self.loop_meta.fresh_examples_path()
    _ = fresh_examples | 'WriteFreshTFExamples' >> (
        sstableio.WriteToSSTable(
            file_path_prefix=os.path.join(output_dir, 'train_examples'),
            key_coder=beam.coders.BytesCoder(),
            value_coder=beam.coders.BytesCoder()))
    _ = fresh_examples | 'WriteToFreshDirectory' >> (
        sstableio.WriteToSSTable(
            file_path_prefix=os.path.join(fresh_dir, 'train_examples'),
            num_shards=self.config.fresh_examples_shards,
            key_coder=beam.coders.BytesCoder(),
            value_coder=beam.coders.BytesCoder()))

  def historical_examples_pipeline(self, root, write_to_fresh=False):
    """Pipeline for generating historical examples.

    Args:
      root: The beam source to anchor the pipeline on.
      write_to_fresh: Boolean signifying whether the historical data should also
        be written to the fresh examples.
    """
    file_pattern = self.loop_meta.all_proof_logs_input_pattern()
    historical_dir = self.loop_meta.historical_examples_path()
    fresh_dir = self.loop_meta.fresh_examples_path()
    logging.info('Input proof logs file pattern:\n%s', file_pattern)
    collections = [
        root | ('ReadAllProofLogs%d' % i) >> recordio.ReadFromRecordIO(
            pattern, coder=beam.coders.ProtoCoder(deephol_pb2.ProofLog))
        for i, pattern in enumerate(file_pattern.split(','))
        if gfile.Glob(pattern)
    ]
    if collections:
      logging.info('Historical prooflog collections: %d.', len(collections))
      examples = (
          collections | 'FlattenInputProofLogs' >> beam.Flatten()
          | ('ConvertHistoricalToTFExamples' >> beam.ParDo(
              ProofLogToTFExamplesDoFn(
                  str(self.config.prover_options.path_tactics), self.theorem_db,
                  self.config.convertor_options))))
      _ = examples | 'WriteHistoricalTFExamples' >> sstableio.WriteToSSTable(
          file_path_prefix=os.path.join(historical_dir, 'train_examples'),
          key_coder=beam.coders.BytesCoder(),
          value_coder=beam.coders.BytesCoder(),
          num_shards=self.config.historical_examples_shards)
      if write_to_fresh:
        _ = examples | 'WriteFreshToHistoricalTFExamples' >> (
            sstableio.WriteToSSTable(
                file_path_prefix=os.path.join(fresh_dir, 'train_examples'),
                key_coder=beam.coders.BytesCoder(),
                value_coder=beam.coders.BytesCoder(),
                num_shards=self.config.fresh_examples_shards))
    else:
      logging.fatal('There are no historical files to process.')

  def create_initial_examples_sstables(self, initial_examples):
    """Create initial empty fresh and historical examples."""
    historical_dir = self.loop_meta.historical_examples_path()
    fresh_dir = self.loop_meta.fresh_examples_path()
    historical_files = gfile.Glob(
        os.path.join(self.loop_meta.historical_examples_path(),
                     'train_examples*'))
    fresh_files = gfile.Glob(
        os.path.join(self.loop_meta.fresh_examples_path(), 'train_examples*'))

    def pipeline(root):
      """Create the pipeline to write historical data."""
      if initial_examples:
        initial_ex = root | 'ReadInitialExamples' >> sstableio.ReadFromSSTable(
            initial_examples)
      else:
        initial_ex = root | 'CreateEmptyExamples' >> beam.Create([])
      if not fresh_files:
        _ = initial_ex | 'WriteToFreshDirectory' >> sstableio.WriteToSSTable(
            file_path_prefix=os.path.join(fresh_dir, 'train_examples'),
            num_shards=self.config.fresh_examples_shards,
            key_coder=beam.coders.BytesCoder(),
            value_coder=beam.coders.BytesCoder())
      if not historical_files:
        _ = (
            initial_ex
            | 'WriteToHistoricalDirectory' >> sstableio.WriteToSSTable(
                file_path_prefix=os.path.join(historical_dir, 'train_examples'),
                key_coder=beam.coders.BytesCoder(),
                value_coder=beam.coders.BytesCoder(),
                num_shards=self.config.historical_examples_shards))

    if not historical_files or not fresh_files:
      if self.config.inherited_proof_logs:
        # Glob doesn't recognize multiple patterns.
        for pattern in self.config.inherited_proof_logs.split(','):
          assert gfile.Glob(pattern), (
              'Can\'t find inherited proof logs at "%s"' % pattern)

        def use_historical_pipeline(root):
          self.historical_examples_pipeline(
              root, write_to_fresh=not fresh_files)

        return use_historical_pipeline
      else:
        return pipeline
    else:
      return None

  def embedding_store_pipeline(self, chkpt: str):
    """Create a pipeline to precompute embeddings for the theorem database.

    Args:
      chkpt: String with the checkpoint prefix.

    Returns:
      A function that builds the pipeline for computing the embeddings.
    """

    def make_pipeline(root):
      _ = (
          root | 'CreateDummyArg' >> beam.Create([''])
          | 'ComputeEmbeddings' >> beam.ParDo(
              EmbeddingComputerDoFn(chkpt, BATCH_SIZE, self.theorem_db))
          | 'WriteEmbeddingsToFile' >> beam.io.WriteToText(
              chkpt,
              '.npy',
              append_trailing_newlines=False,
              shard_name_template=''))

    return make_pipeline

  def setup_model_checkpoint_and_embeddings(self):
    """Copy embeddings over and precompute theorem database embeddings.

    This function makes sure that we have at least one model checkpoint
    file present. Also it copies over the latest new embeddings when they become
    available and precomputes the embedding store for them.
    """
    logging.info('Setting up model checkpoint and embeddings %s %s',
                 str(self.config.copy_model_checkpoints),
                 str(self.checkpoint_monitor.has_checkpoint()))
    # We can prohibit copying checkpoints by setting copy_model_checkpoints
    # to false, unless we don't have any checkpoint yet, in which case
    # we try to copy a new checkpoint over.
    while self.config.copy_model_checkpoints or not (
        self.checkpoint_monitor.has_checkpoint()):
      # Whether we have a pre-existing checkpoint.
      has_checkpoint = self.checkpoint_monitor.has_checkpoint()
      logging.info('has checkpoint: %s', has_checkpoint)
      # new_checkpoint is None if the training directory does not
      # have a more recent checkpoint than the one stored in the loop
      # directory. Otherwise it refers to the current newest checkpoint.
      new_checkpoint = self.checkpoint_monitor.new_checkpoint()
      logging.info('new checkpoint: %s', new_checkpoint)
      if new_checkpoint is not None:
        # We have a more recent checkpoint than in our local directory.
        logging.info('New checkpoint: "%s"', new_checkpoint)
        self.checkpoint_monitor.copy_latest_checkpoint()
        chkpt = os.path.join(self.loop_meta.checkpoints_path(), new_checkpoint)
        logging.info('Copied checkpoint: "%s"', chkpt)
        # We try to compute embeddings until we succeed.
        while not gfile.Exists(chkpt + '.npy'):
          runner.Runner().run(
              self.embedding_store_pipeline(chkpt)).wait_until_finish()
          if not gfile.Exists(chkpt + '.npy'):
            logging.error(
                'Could not generate embeddings for the latest '
                'checkpoint %s.', chkpt)
          else:
            self.checkpoint_monitor.update_latest_checkpoint(new_checkpoint)
            break
      # If we had a pre-existing checkpoint or we managed to copy over
      # a new one, then we are succeeded. Let's not check the checkpoint
      # unless we had none.
      if has_checkpoint or self.checkpoint_monitor.has_checkpoint():
        break
      else:
        # We don't have a checkpoint and never had one. Let's wait for
        # one appear in the training directory.
        logging.info('Waiting for the first model checkpoint to appear.')
        time.sleep(10)
    # TODO(szegedy): Cleanup old checkpoints if there are too many of them.

  def reporting_pipeline(self, proof_logs):
    current_round = self.loop_meta.status.current_round
    rp = report.ReportingPipeline(self.loop_meta.stats_path(current_round))
    rp.setup_pipeline(proof_logs)

  def aggregate_reporting(self):
    current_round = self.loop_meta.status.current_round
    rp = report.ReportingPipeline(self.loop_meta.stats_path(current_round))
    rp.write_final_stats()

  def perform_round(self, initial_examples):
    """Perform a single round of the loop and advance the loop counter."""
    current_round = self.loop_meta.status.current_round
    if current_round == 0:
      self.setup_examples(initial_examples)
    if not self.prover_tasks:
      logging.info('No tasks for proving...')
      return
    logging.info('******** ROUND %d', current_round)
    logging.info('Setting up latest checkpoints for ROUND %d', current_round)
    self.setup_model_checkpoint_and_embeddings()
    logging.info('Creating prover tasks for ROUND %d', current_round)
    tasks = self.create_prover_tasks()
    logging.info('Number of tasks: %d', len(tasks))
    logging.info(
        'Running prover and example generation pipeline '
        'for ROUND %d', current_round)

    def pipeline(root):
      proof_logs = self.prover_pipeline(tasks, root)
      self.reporting_pipeline(proof_logs)
      self.training_examples_pipeline(proof_logs)
      self.historical_examples_pipeline(root)

    runner.Runner().run(pipeline).wait_until_finish()
    self.aggregate_reporting()
    self.loop_meta.prepare_next_round()

  def setup_examples(self, initial_examples):
    logging.info('Creating initial examples pipeline')
    pipeline = self.create_initial_examples_sstables(initial_examples)
    if pipeline:
      logging.info('Generating initial examples sstables...')
      runner.Runner().run(pipeline).wait_until_finish()
    else:
      logging.info('Examples are already present')
