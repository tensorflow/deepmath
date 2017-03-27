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
"""Write a graph for use in C++ clause search inference."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from deepmath.guidance import driver_lib

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('hparams', '', 'Comma separate list of name=value pairs.')
flags.DEFINE_string('output', None, 'Output graph def file')


def main(unused_argv):
  if not FLAGS.output:
    raise ValueError('--output is required')
  hparams = driver_lib.parse_hparams(FLAGS.hparams)
  driver_lib.export_inference_meta_graph(hparams, filename=FLAGS.output)


if __name__ == '__main__':
  tf.app.run()
