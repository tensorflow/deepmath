"""Monitor the latest model checkpoint and compute embedding stores.

This library is a helper method for the loop to monitor checkpoints
when they get available. Once a new checkpoint appears, it gets copied over
to a temporary directory, then the embeddings are computed for the theorem
database. Finally, the checkpoint file is updated. Old checkpoints can be
removed in the meantime.
"""

from __future__ import absolute_import
from __future__ import division
# Import Type Annotations
from __future__ import print_function

import os
from tensorflow import gfile
from tensorflow import logging


def get_latest_checkpoint(dirname: str):
  """Get the latest checkpoint in the directory.

  Args:
    dirname: Name of the directory.

  Returns:
    Checkpoint prefix string.
  """
  chkpt_file = os.path.join(dirname, 'checkpoint')
  if not gfile.Exists(chkpt_file):
    logging.info('File %s does not exist', chkpt_file)
    return None
  chkpt_export_folder = os.path.join(dirname, 'export')
  if not gfile.Exists(chkpt_export_folder):
    logging.info('Eval export folder %s does not exist', chkpt_export_folder)
    return None
  num_lines = 0
  with gfile.Open(chkpt_file) as f:
    for l in f:
      num_lines += 1
      if l.startswith(b'model_checkpoint_path:'):
        return os.path.basename(l.strip().split()[1][1:-1])
  return None


def set_latest_checkpoint(dirname: str, chkpt: str):
  """Set the latest checkpoint in the checkpoint file.

  Args:
    dirname: Directory in which the checkpoint is located.
    chkpt: Checkpoint prefix.
  """
  chkpt_file = os.path.join(dirname, 'checkpoint')
  lines = []
  if gfile.Exists(chkpt_file):
    logging.info('Loading preexisting checkpoint file "%s"', chkpt_file)
    with gfile.Open(chkpt_file) as f:
      lines = [
          l.strip()
          for l in f.readlines()
          if l.startswith(b'all_model_checkpoint_paths:')
      ]
  else:
    logging.info('No preexisting checkpoint file "%s"', chkpt_file)
  with gfile.Open(chkpt_file, 'w') as f:
    lines = [
        '%s\n' % l.strip() for l in ([
            'model_checkpoint_path: "%s"' % chkpt,
            'all_model_checkpoint_paths: "%s"' % chkpt
        ] + lines)
    ]
    f.writelines(lines)


def verbose_copy(src, tgt, overwrite=True):
  logging.info('Copying "%s" -> "%s"', src, tgt)
  gfile.Copy(src, tgt, overwrite=overwrite)


class CheckpointMonitor(object):
  """Class for syncing checkpoints between two directories."""

  def __init__(self, model_directory, target_directory, checkpoints_to_keep=2):
    self.model_directory = model_directory
    self.target_directory = target_directory
    self.checkpoints_to_keep = checkpoints_to_keep

  def new_checkpoint(self):
    logging.info('Looking for checkpoint in "%s"', self.model_directory)
    chkpt = get_latest_checkpoint(self.model_directory)
    logging.info('Checkpoint: %s', chkpt)
    if chkpt != get_latest_checkpoint(self.target_directory):
      logging.info('latest checkpoint: %s',
                   get_latest_checkpoint(self.target_directory))
      return chkpt
    else:
      return None

  def copy_latest_checkpoint(self):
    """Copy over the latest checkpoints to the target directory."""
    chkpt = get_latest_checkpoint(self.model_directory)
    logging.info('Got latest checkpoint: %s', chkpt)
    if chkpt is None:
      return None
    # Check if the evaluation meta graph has been copied.
    if self.has_checkpoint() is None:
      # Don't copy temp export folders, e.g. 'temp-01234567/saved_model.pb'
      export_file = gfile.Glob(
          os.path.join(self.model_directory,
                       'export/best_exporter/[0-9]*/saved_model.pb'))[0]
      logging.info('Copying eval export file: %s', ', '.join(export_file))
      target_export_dir = os.path.join(
          self.target_directory, 'export/best_exporter',
          os.path.basename(os.path.dirname(export_file)))
      gfile.MakeDirs(target_export_dir)
      verbose_copy(
          export_file,
          os.path.join(target_export_dir, os.path.basename(export_file)))
    files = gfile.Glob(os.path.join(self.model_directory, chkpt) + b'.*')
    logging.info('Copying files: %s', ', '.join(files))
    for fname in files:
      verbose_copy(fname,
                   os.path.join(self.target_directory, os.path.basename(fname)))
    return chkpt

  def update_latest_checkpoint(self, chkpt):
    old_chkpt = get_latest_checkpoint(self.target_directory)
    if old_chkpt != chkpt:
      set_latest_checkpoint(self.target_directory, chkpt)

  def has_checkpoint(self):
    return get_latest_checkpoint(self.target_directory)

  def get_checkpoint(self):
    logging.info('Getting checkpoint for %s', self.target_directory)
    chkpt = get_latest_checkpoint(self.target_directory)
    if chkpt is None:
      return None
    else:
      return os.path.join(self.target_directory, chkpt)
