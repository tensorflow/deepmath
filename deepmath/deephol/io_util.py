"""Methods for reading/writing to disk used in main, and helpers.

Helper methods are to maintain invariants/assumptions about data, and do any
light-weight pre/post-processing.
"""

from __future__ import absolute_import
from __future__ import division
# Import Type Annotations
from __future__ import print_function
import re

import tensorflow as tf
from typing import List, Optional, Text
from google.protobuf import text_format
from deepmath.deephol import deephol_pb2
from deepmath.deephol.public import recordio_util
from deepmath.proof_assistant import proof_assistant_pb2


def _process_tactics_and_replacements(tactics_info: deephol_pb2.TacticsInfo,
                                      replacements: deephol_pb2.TacticsInfo
                                     ) -> List[deephol_pb2.Tactic]:
  """Check tactics are in order, have no gap in id, and apply replacements."""
  bad_ids = [(i, tactic.id)
             for i, tactic in enumerate(tactics_info.tactics)
             if tactic.id != i]
  if bad_ids:
    raise ValueError(' '.join('Tactic #%d should not have id %d.' %
                              (i, tactic_id) for i, tactic_id in bad_ids))
  bad_replacement_ids = [
      replacement.id
      for replacement in replacements.tactics
      if replacement.id < 0 or replacement.id >= len(tactics_info.tactics)
  ]
  if bad_replacement_ids:
    raise ValueError(' '.join('Replacement tactic id=%d out of bounds.' %
                              tactic_id for tactic_id in bad_replacement_ids))
  for replacement in replacements.tactics:
    tactics_info.tactics[replacement.id].CopyFrom(replacement)
  return list(tactics_info.tactics)


def load_proto(filename: Text,
               proto_constructor,
               description: Optional[Text] = None):
  """Load a binary protobuf from a .pb format file.

  Args:
    filename: Name of the file to be read.
    proto_constructor: The constructor method for the proto object.
    description: Optional string describing the content of the proto file.

  Returns:
    A protobuf parsed from the text file.
  """
  proto = proto_constructor()
  with tf.gfile.Open(filename, 'rb') as f:
    file_content = f.read()
    proto.ParseFromString(file_content)
  if description:
    tf.logging.info('Successfully read %s from "%s"', description, filename)
  return proto


def load_text_proto(filename: Text,
                    proto_constructor,
                    description: Optional[Text] = None):
  """Load a protobuf from a text format file.

  Args:
    filename: Name of the file to be read.
    proto_constructor: The constructor method for the proto object.
    description: Optional string describing the content of the proto file.

  Returns:
    A protobuf parsed from the text file.
  """
  proto = proto_constructor()
  with tf.gfile.Open(filename) as f:
    text_format.MergeLines(f, proto)
  if description:
    tf.logging.info('Successfully read %s from "%s"', description, filename)
  return proto


def write_text_proto(filename: Text, proto):
  with tf.gfile.Open(filename, 'w') as f:
    f.write(text_format.MessageToString(proto))


def write_text_protos(filename: Text, protos):
  first = True
  with tf.gfile.Open(filename, 'w') as f:
    for proto in protos:
      if not first:
        f.write('\n')
      else:
        first = False
      f.write(text_format.MessageToString(proto).replace('\n', ' '))


def write_protos(filename: Text, protos, text_output=True):
  if text_output:
    write_text_protos(filename, protos)
  else:
    recordio_util.write_protos_to_recordio(filename, protos)


def load_tactics_from_file(tactics_filename: Text,
                           tactics_replacement_filename: Optional[Text]
                          ) -> List[deephol_pb2.Tactic]:
  """Load tactics from file, and (optional) apply replacements."""
  tactics_info = load_text_proto(tactics_filename, deephol_pb2.TacticsInfo,
                                 'tactics')
  tf.logging.info('Found %d tactics.', len(tactics_info.tactics))
  replacements = deephol_pb2.TacticsInfo()
  if tactics_replacement_filename:
    replacements = load_text_proto(tactics_replacement_filename,
                                   deephol_pb2.TacticsInfo, 'replacements')
    tf.logging.info('Found %d tactic replacements.', len(replacements.tactics))
  return _process_tactics_and_replacements(tactics_info, replacements)


def load_theorem_database_from_file(filename: Text
                                   ) -> proof_assistant_pb2.TheoremDatabase:
  """Load a theorem database from a text protobuf file."""
  theorem_database = proof_assistant_pb2.TheoremDatabase()
  if filename.endswith('.recordio'):
    theorem_database = [
        x for x in recordio_util.read_protos_from_recordio(
            filename, proof_assistant_pb2.TheoremDatabase)
    ]
    theorem_database = theorem_database[0]
  else:
    with tf.gfile.Open(filename) as f:
      text_format.MergeLines(f, theorem_database)
  tf.logging.info('Successfully read theorem database from %s (%d theorems).',
                  filename, len(theorem_database.theorems))
  return theorem_database


def load_text_protos(filename, proto_class):
  """Load protos from a text file where each line is one proto."""
  cnt = 0
  with tf.gfile.Open(filename) as f:
    for line in f:
      if line.rstrip():
        proto = proto_class()
        text_format.Parse(line, proto)
        cnt += 1
        yield proto
  tf.logging.info('Read %d protos from %s of type %s', cnt, filename,
                  proto_class.DESCRIPTOR.full_name)


def read_protos(pattern: Text, proto_class):
  r"""Load protos or a single proto from various possible sources.

  The following sources are possible:
     - Any comma separated sources below:
     - glob of files with textpb or pbtxt extension: single text proto files.
     - glob of files with textpbs or pbtxts extension: proto text representation
                                                       in each line.

  Args:
    pattern: Either a sharding pattern or a comma separated sharding patterns.
    proto_class: The class name of the protobufs to be produced.

  Yields:
    Protobufs from the specified class parsed from the files with given pattern.
  """
  patterns = pattern.split(',')
  if len(patterns) > 1:
    for pattern in patterns:
      for proto in read_protos(pattern, proto_class):
        yield proto
    return
  match = re.search('@\\d+$', pattern)
  if match:
    for proto in recordio_util.read_protos_from_recordio(pattern, proto_class):
      yield proto
    return
  filenames = tf.io.gfile.glob(pattern)
  if not filenames:
    raise ValueError('read_protos: No files found matching %s' % pattern)
  for filename in filenames:
    if filename.endswith('.textpb') or filename.endswith('.pbtxt'):
      yield load_text_proto(filename, proto_class)
    elif filename.endswith('.textpbs') or filename.endswith('.pbtxts'):
      for proto in load_text_protos(filename, proto_class):
        yield proto
    else:
      for proto in recordio_util.read_protos_from_recordio(
          filename, proto_class):
        yield proto


def options_reader(options_proto, options_proto_path: Text,
                   overwrite: Optional[Text]):
  """Generic options reader, which can also be easily saved as protos.

  Arguments:
    options_proto: Type of the options proto object.
    options_proto_path: Path to file containing an options_proto.
    overwrite: A string containing options proto object.

  Returns:
    An options_proto proto containing parsed options.
  """
  ret = load_text_proto(options_proto_path, options_proto)
  if overwrite:
    text_format.Merge(overwrite, ret)
  tf.logging.info('Options:\n\n%s\n\n', ret)
  return ret
