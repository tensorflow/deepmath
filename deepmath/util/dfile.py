# Copyright 2016 Google Inc. All Rights Reserved.
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
"""File utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import numpy as np
import tensorflow as tf


_SHARD_SPEC_RE = re.compile(r'^(.*)@(\d+)((?:\.[^@]+)?)$')
_SSTABLE_MAGIC = b'\x8a\x4b\x9a\x00\x1d\x86\x2c\x97'


def _shard_width(shards):
  return max(5, int(np.log10(shards)) + 1)


def sharded_filenames(spec):
  """Expand a sharded filename specification into a list of paths.

  Args:
    spec: A sharded filename spec of the form `prefix@shards[.suffix]`.

  Returns:
    A list of paths to each shard, of the form `prefix-k-of-shards[.suffix]`.

  Raises:
    ValueError: If the spec is invalid.
  """
  m = _SHARD_SPEC_RE.match(spec)
  if not m:
    raise ValueError('Invalid shard spec %r' % spec)
  prefix = m.group(1)
  shards = int(m.group(2))
  suffix = m.group(3)
  width = _shard_width(shards)
  return ['%s-%0*d-of-%05d%s' % (prefix, width, i, shards, suffix)
          for i in range(shards)]


def is_sstable(path):
  """Check whether a file is in sstable format."""
  f = tf.gfile.Open(path, 'rb')
  size = f.size()
  if size < 8:
    return False
  f.seek(size - 8)
  return f.read(8) == _SSTABLE_MAGIC


def read_sstable_or_tfrecord(paths, shuffle=True):
  """Read examples from either TFRecord or SSTable.

  Creates a queue internally, so make sure queue runners are started.

  Args:
    paths: Paths to TFRecord or SSTable shards.
    shuffle: Whether to shuffle shards.

  Returns:
    key: Scalar string tensor key read from paths.
    value: Scalar string tensor value read from paths.
  """
  if is_sstable(paths[0]):
    # Import tensorflow.google only if SSTable support is required,
    # so that open source works without if in TFRecord mode.
    import tensorflow.google  # pylint: disable=g-import-not-at-top
    reader = tensorflow.google.SSTableReader()
  else:
    reader = tf.TFRecordReader()
  path_queue = tf.train.string_input_producer(paths, shuffle=shuffle)
  return reader.read(path_queue)
