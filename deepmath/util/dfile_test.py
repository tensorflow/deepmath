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
"""Tests for util.dfile."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
import tensorflow as tf
from deepmath.util import dfile


class DFileTest(tf.test.TestCase):

  def testShardWidth(self):
    self.assertEqual(dfile._shard_width(17), 5)
    self.assertEqual(dfile._shard_width(99999), 5)
    self.assertEqual(dfile._shard_width(100000), 6)
    self.assertEqual(dfile._shard_width(999999), 6)
    self.assertEqual(dfile._shard_width(1000000), 7)

  def testShardedFilenames(self):
    for suffix in '', '.png':
      shards = dfile.sharded_filenames('prefix@3' + suffix)
      self.assertEqual(shards, ['prefix-00000-of-00003' + suffix,
                                'prefix-00001-of-00003' + suffix,
                                'prefix-00002-of-00003' + suffix])

  def testSharedFilenamesInvalidSpec(self):
    with self.assertRaisesRegexp(ValueError, r'Invalid shard spec'):
      dfile.sharded_filenames('prefix@3.@w')

  def testIsSSTable(self):
    magic = dfile._SSTABLE_MAGIC
    for contents in b'', b'kjdslkajdaksjla', magic, b'x' + magic:
      f = tempfile.NamedTemporaryFile()
      f.write(contents)
      f.flush()
      self.assertEqual(dfile.is_sstable(f.name), contents.endswith(magic))

  def testReadTFRecord(self):
    data = b'yes', b'no'
    f = tempfile.NamedTemporaryFile()
    with tf.python_io.TFRecordWriter(f.name) as records:
      for s in data:
        records.write(s)
    _, value = dfile.read_sstable_or_tfrecord([f.name])
    with self.test_session():
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)
      for s in data:
        self.assertEqual(value.eval(), s)
      coord.request_stop()
      for thread in threads:
        thread.join()


if __name__ == '__main__':
  tf.test.main()
