"""Apache Beam pipeline to run the prover.

This beam pipeline and DoFns runs the prover and creates proof-logs.
"""

from __future__ import absolute_import
from __future__ import division
# Import Type Annotations
from __future__ import print_function

import apache_beam as beam
from apache_beam.metrics import Metrics
from tensorflow import logging
from typing import List, Optional, Text, Tuple
from google.protobuf import text_format
from deepmath.public import build_data
from deepmath.deephol import deephol_pb2
from deepmath.deephol import io_util
from deepmath.deephol import prover
from deepmath.deephol.deephol_loop import options_pb2
from deepmath.deephol.deephol_loop import prooflog_to_tfexamples_lib
from deepmath.deephol.deephol_loop.missing import recordio
from deepmath.deephol.deephol_loop.missing import runner
from deepmath.deephol.deephol_loop.missing import sstableio
from deepmath.proof_assistant import proof_assistant_pb2


class ProverDoFn(beam.DoFn):
  """Beam DoFn for our prover."""

  def __init__(self, prover_options: deephol_pb2.ProverOptions):
    self.prover_options = prover_options
    self.proven_counter = Metrics.counter(self.__class__, 'proven')
    self.attempted_counter = Metrics.counter(self.__class__, 'attempted')
    self.failed_counter = Metrics.counter(self.__class__, 'failed')
    self.accepts_counter = Metrics.counter(self.__class__, 'accepts task')
    self.rejected_counter = Metrics.counter(self.__class__, 'rejected task')
    self.does_not_accept_counter = Metrics.counter(self.__class__,
                                                   'does not accept task')
    self.timeout_counter = Metrics.counter(self.__class__, 'timeout')

  def start_bundle(self):
    self.prover = prover.create_prover(self.prover_options)

  def process(self, task: proof_assistant_pb2.ProverTask
             ) -> List[deephol_pb2.ProofLog]:
    logging.info('Processing task:\n%s', text_format.MessageToString(task))
    self.attempted_counter.inc()
    if self.prover.accept_tasks:
      self.accepts_counter.inc()
    else:
      self.does_not_accept_counter.inc()
    proof_log = self.prover.prove(task)
    timed_out = self.prover.timed_out()
    if proof_log.rejected:
      self.rejected_counter.inc()
    if not proof_log.error_message:
      self.proven_counter.inc()
    else:
      logging.info('Failed proof with "%s"', proof_log.error_message)
      self.failed_counter.inc()
      if timed_out:
        self.timeout_counter.inc()
    proof_log.build_data = build_data.BuildData()
    return [proof_log]


def make_pipeline(prover_tasks: List[proof_assistant_pb2.ProverTask],
                  prover_options: deephol_pb2.ProverOptions, path_output: str):
  """A simple create-process-write Beam pipeline for proving theorems."""

  def pipeline(root):
    logs = (
        root
        | 'Create' >> beam.Create(prover_tasks)
        | 'Prove' >> beam.ParDo(ProverDoFn(prover_options)))
    _ = logs | 'Write' >> recordio.WriteToRecordIO(
        file_path_prefix=path_output,
        coder=beam.coders.ProtoCoder(deephol_pb2.ProofLog))
    return logs

  return pipeline


def key_value_of_proto(proto):
  value = proto.SerializeToString()
  key = hash(value)
  return ('%x' % key, value)


class ProofLogToTFExamplesDoFn(beam.DoFn):
  """DoFn for converting proof logs to tf examples."""

  def __init__(self, tactics_filename: str,
               theorem_db: proof_assistant_pb2.TheoremDatabase,
               scrub_parameters):
    options = options_pb2.ConvertorOptions(
        tactics_path=tactics_filename, scrub_parameters=scrub_parameters)
    self.converter = prooflog_to_tfexamples_lib.create_processor(
        options=options, theorem_database=theorem_db)

  def start_bundle(self):
    pass

  def process(self, proof_log: deephol_pb2.ProofLog) -> List[Tuple[int, str]]:
    return [
        key_value_of_proto(example)
        for example in self.converter.process_proof_log(proof_log)
    ]


def training_examples_pipeline(
    proof_logs,
    tactics_filename: Text,
    theorem_db: proof_assistant_pb2.TheoremDatabase,
    examples_sstables: List[Text],
    scrub_parameters: options_pb2.ConvertorOptions.ScrubParametersEnum,
):
  """Create the pipeline to convert ProofLogs to Examples.

  Args:
    proof_logs: beam node for the proof logs.
    tactics_filename: Name for the tactics file.
    theorem_db: Theorem database file.
    examples_sstables: List of strings with sstable pattern to write the
      examples to.
    scrub_parameters: Theorem parameters to scrub during examples generation.
  """
  examples = proof_logs | ('ConvertToTFExamples' >> beam.ParDo(
      ProofLogToTFExamplesDoFn(
          str(tactics_filename), theorem_db, scrub_parameters)))
  for i, examples_sstable in enumerate(examples_sstables):
    examples_prefix = examples_sstable
    num_shards = None,
    logging.info('sstable: %s', examples_sstable)
    if '@' in examples_sstable:
      examples_prefix, num_shards = examples_sstable.split('@')
      num_shards = int(num_shards)
    _ = examples | ('WriteExamples%d' % i) >> (
        sstableio.WriteToSSTable(
            file_path_prefix=examples_prefix,
            num_shards=num_shards,
            key_coder=beam.coders.BytesCoder(),
            value_coder=beam.coders.BytesCoder()))


def run_pipeline(examples_sstable: Optional[Text],
                 scrub_parameters: Optional[Text],
                 prover_tasks: List[proof_assistant_pb2.ProverTask],
                 prover_options: deephol_pb2.ProverOptions, path_output: str):
  """Create and run simple prover pipeline."""
  prover.cache_embeddings(prover_options)
  prover_pipeline = make_pipeline(prover_tasks, prover_options, path_output)
  pipeline = prover_pipeline
  if examples_sstable:
    theorem_db = io_util.load_theorem_database_from_file(
        str(prover_options.path_theorem_database))

    def examples_pipeline(root):
      """Examples pipeline."""
      scrub_str_enum_map = {
          'NOTHING':
              options_pb2.ConvertorOptions.NOTHING,
          'TESTING':
              options_pb2.ConvertorOptions.TESTING,
          'VALIDATION_AND_TESTING':
              options_pb2.ConvertorOptions.VALIDATION_AND_TESTING,
      }
      training_examples_pipeline(
          proof_logs=prover_pipeline(root),
          tactics_filename=prover_options.path_tactics,
          theorem_db=theorem_db,
          examples_sstables=[examples_sstable],
          scrub_parameters=scrub_str_enum_map[scrub_parameters])

    pipeline = examples_pipeline
  runner.Runner().run(pipeline).wait_until_finish()


def program_started():
  runner.program_started()
