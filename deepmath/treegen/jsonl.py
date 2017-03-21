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
"""Utilities for JSON-lines files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import random
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


class JSONLinesIterator(object):
  """Iterator over JSON-encoded lines of a file."""

  def __init__(self, filename, random_start=False):
    self.f = tf.gfile.GFile(filename)
    if random_start:
      # TODO(ricshin): Implement better shuffling options.
      self.seek_to_line_after(
          random.randint(0, self.f.Size() - 1))

  def __iter__(self):
    return self

  def next(self):
    line = self.f.readline()
    if not line:
      self.f.seek(0)
      line = self.f.readline()
    return json.loads(line)

  def seek_to_line_after(self, pos):
    self.f.seek(pos)

    # Find the next '\n'.
    for _ in xrange(self.f.Size()):
      char = self.f.read(1)
      if char == '\n':
        break
      if not char:
        # Reached EOF, go back to beginning of file.
        self.f.seek(0)
    else:
      # We never found a '\n' after going through the entire file,
      # so go back to beginning and treat the entire file as a line.
      self.f.seek(0)

  def reset(self):
    self.f.seek(0)
