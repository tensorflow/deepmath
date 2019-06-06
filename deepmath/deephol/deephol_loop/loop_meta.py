"""Support for controlling metadata and directories of the prove-train loop.

The Loop class should we the only class that has low level access to the
metadata and layout manipulation of the prove-train loop.
"""
from __future__ import absolute_import
from __future__ import division
# Import Type Annotations
from __future__ import print_function
import farmhash
import os
import time
from tensorflow import gfile
from tensorflow import logging
from typing import Optional
from deepmath.public import build_data
from deepmath.deephol import io_util
from deepmath.deephol.deephol_loop import loop_pb2
from deepmath.deephol.deephol_loop.missing import types_pb2

# Base name of the status file in the top level directory.
STATUS_BASENAME = 'status.pbtxt'
# Base name of the configuration file in the top level directory.
CONFIG_BASENAME = 'config.pbtxt'
# Top level directory of the proof log. Each round will have its subdirectory
# which is of format '%07d' % round. Each of that subdirectory will contain
# a shareded recordio file with the proofs from that round as well as a
# stats directory for proof statistics information about that proof round.
PROOF_LOGS_BASENAME = 'proof_logs'
# Subdirectory of each round with the statistics for that round.
PROOF_STATS_BASENAME = 'stats'
# Base name of the top level directory containing examples. It will contain a
# subdirectory of examples corresponding to each round with format
# '%07d' % round as well as two subdirectories "fresh" and "historical".
# These contain examples from the last round of training data mixing.
TRAINING_EXAMPLES_BASENAME = 'training_examples'
# Base name from older examples from this loop. This directory can be passed
# on to the TensorFlow training.
HISTORICAL_EXAMPLES_BASENAME = 'historical'
# Base name from the fresh examples from this loop. This directory can be passed
# on to the TensorFlow training.
FRESH_EXAMPLES_BASENAME = 'fresh'
# Base name of the directory containing the prover tasks to be run in
# each round. The task_files have the format 'tasks_%07d.pbtxts' % round.
PROVER_TASKS_BASENAME = 'prover_tasks'
# Base name of the directory containing the checkpoints and the
# precomputed embeddings for the theorem database
CHECKPOINTS_BASENAME = 'checkpoints'


def create_fingerprint():
  """Create a unique fingerprint for the running process.

  The main function should call this function once to create a unique
  fingerprint for that run.

  Returns:
    Unique fingerprint for this run.
  """
  build_run_str = (
      build_data.BuildData() + 'time: %d' % (int(round(time.time()))))
  return farmhash.fingerprint64(build_run_str)


def loop_round_string(loop_round: int):
  return '%07d' % loop_round


def _check_directory(filename: str) -> Optional[str]:
  if gfile.Exists(filename):
    if gfile.IsDirectory(filename):
      return None
    else:
      return '"%s" is expected to be a directory.' % filename
  else:
    return 'Expected directory %s does not exist.' % filename


def _check_file(filename: str) -> Optional[str]:
  if gfile.Exists(filename):
    stat_proto = gfile.Stat(filename, stat_proto=True)
    if stat_proto.file_type == types_pb2.FILE:
      return None
    else:
      return '"%s" is expected to be a regular file.' % filename
  else:
    return 'Expected file %s does not exist.' % filename


def make_dir(dir_name: str) -> str:
  if gfile.Exists(dir_name):
    if gfile.IsDirectory(dir_name):
      return dir_name
    else:
      logging.fatal(
          'Trying to create directory "%s", but there '
          'is a file with the same name', dir_name)
  gfile.MakeDirs(dir_name)
  return dir_name


class LoopMeta(object):
  """Meta data and directory manipulations for the prove-train loop.

  This class is responsible for maintaining a consistent set of
  proof logs, training data and storing meta-information, like
  current status of the loop execution. It also keeps track of the
  current proving round.
  """

  def __init__(self,
               root: str,
               config: loop_pb2.LoopConfig,
               controller_fingerprint: int,
               read_only=None):
    self.root = os.path.join(root, config.name)
    self.config = config
    self.read_only = read_only
    self.controller_fingerprint = controller_fingerprint
    self.status = None
    self.error = None
    if self.layout_exists():
      self.error = self.check_layout()
      self.read_status()
      assert self.status, 'Could not read status %s.' % self.status_filename()
    else:
      if read_only:
        self.error = 'Non-existent loop layout at %s' % self.root
      else:
        self.status = self.new_status()
        self.make_layout()
        self.error = self.check_layout()
    if self.error is not None:
      logging.error('%s', self.error)

  def status_filename(self):
    return os.path.join(self.root, STATUS_BASENAME)

  def config_filename(self):
    return os.path.join(self.root, CONFIG_BASENAME)

  def proof_logs_path(self, loop_round: Optional[int] = None):
    if loop_round is None:
      return os.path.join(self.root, PROOF_LOGS_BASENAME)
    else:
      return os.path.join(self.root, PROOF_LOGS_BASENAME,
                          loop_round_string(loop_round))

  def stats_path(self, loop_round: Optional[int] = None):
    return os.path.join(self.proof_logs_path(loop_round), PROOF_STATS_BASENAME)

  def make_proof_logs_dir(self, loop_round):
    return make_dir(self.proof_logs_path(loop_round))

  def training_examples_path(self, loop_round: Optional[int] = None):
    if loop_round is None:
      return os.path.join(self.root, TRAINING_EXAMPLES_BASENAME)
    else:
      return os.path.join(self.root, TRAINING_EXAMPLES_BASENAME,
                          loop_round_string(loop_round))

  def fresh_examples_path(self):
    return os.path.join(self.root, TRAINING_EXAMPLES_BASENAME,
                        FRESH_EXAMPLES_BASENAME)

  def historical_examples_path(self):
    return os.path.join(self.root, TRAINING_EXAMPLES_BASENAME,
                        HISTORICAL_EXAMPLES_BASENAME)

  def make_training_examples_dir(self, loop_round) -> str:
    return make_dir(self.training_examples_path(loop_round))

  def prover_tasks_path(self):
    return os.path.join(self.root, PROVER_TASKS_BASENAME)

  def checkpoints_path(self):
    return os.path.join(self.root, CHECKPOINTS_BASENAME)

  def read_status(self):
    self.status = io_util.load_text_proto(self.status_filename(),
                                          loop_pb2.LoopStatus, 'Status file')

  def override_status_fingerprint(self):
    self.status.running_controller = self.controller_fingerprint
    self.write_status()

  def check_status(self):
    status = io_util.load_text_proto(self.status_filename(),
                                     loop_pb2.LoopStatus, 'Status file')
    if (status.name != self.status.name or
        status.last_finished_round != self.status.last_finished_round or
        status.current_round != self.status.current_round or
        status.running_controller != self.running_controller):
      logging.fatal('Inconsistent status between stored status and disk')

  def write_status(self):
    io_util.write_text_proto(self.status_filename(), self.status)

  def new_status(self):
    return loop_pb2.LoopStatus(
        name=self.config.name,
        current_round=0,
        running_controller=self.controller_fingerprint)

  def layout_exists(self):
    return gfile.Exists(self.root)

  def check_layout(self) -> Optional[str]:
    """Check the current layout and return an error string or None.

    Returns:
      None if the current layout is correct, otherwise an error string.
      Currently it just checkes that each required files are present.
      Later it should check metadata and that the round-specific
      subdirectories are present.
      TODO(szegedy): more extensive checks.
    """
    return (_check_directory(self.root) or
            _check_file(self.status_filename()) or
            _check_file(self.config_filename()) or
            _check_directory(self.proof_logs_path()) or
            _check_directory(self.training_examples_path()) or
            _check_directory(self.prover_tasks_path()) or
            _check_directory(self.checkpoints_path()) or
            _check_directory(self.fresh_examples_path()) or
            _check_directory(self.historical_examples_path()))

  def make_layout(self):
    """Make a new layout for the prove-train loop."""
    if self.status is None:
      return 'make_layout: Internal status is not set.'
    if self.layout_exists():
      return 'make_layout: Layout %s exists at %s' % (self.config.name,
                                                      self.root)
    gfile.MakeDirs(self.root)
    if not gfile.Exists(self.root):
      return 'Could not create directory %s' % self.root
    io_util.write_text_proto(self.config_filename(), self.config)
    self.write_status()
    gfile.MakeDirs(self.proof_logs_path())
    gfile.MakeDirs(self.training_examples_path())
    gfile.MakeDirs(self.prover_tasks_path())
    gfile.MakeDirs(self.checkpoints_path())
    gfile.MakeDirs(self.fresh_examples_path())
    gfile.MakeDirs(self.historical_examples_path())

  def prepare_next_round(self):
    assert self.status
    self.status.current_round += 1
    self.write_status()

  def all_proof_logs_input_pattern(self):
    p = self.proof_logs_path()
    return (os.path.join(p, '[0-9]' * 7, 'logs-*-of-*') + ',' +
            self.config.inherited_proof_logs)
