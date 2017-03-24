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
"""Check for boilerplate: licenses, futures, etc.

This makes it easier to find Python 2.7 / Python 3.x incompatibility bugs.
In particular, this test makes it illegal to write a Python file that
doesn't import division from __future__, which can catch subtle division
bugs in Python 3.

Note: We can't use tf.test in this file because it needs to run in an
environment that doesn't include license-free gen_blah_ops.py files.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import six

PY_LICENSE_RE = re.compile(r'''
(# coding=utf-8\n)?# Copyright \d+ Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 \(the "License"\);
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
'''[1:], flags=re.MULTILINE)

CC_LICENSE_RE = re.compile(r'''
/\* Copyright \d+ Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 \(the "License"\);
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================\*/
'''[1:], flags=re.MULTILINE)

BASE_DIR = os.path.normpath(os.path.join(__file__, '../..'))
FUTURE_RE = re.compile(r'^from __future__ import (\w+)\s*$', flags=re.MULTILINE)
REQUIRED_FUTURES = frozenset(['absolute_import', 'division', 'print_function'])
LICENSE_RE = {'.py': PY_LICENSE_RE, '.h': CC_LICENSE_RE, '.cc': CC_LICENSE_RE}


def main():
  # Make sure BASE_DIR ends with deepmath.  If it doesn't, we probably
  # computed the wrong directory.
  if os.path.split(BASE_DIR)[-1] != 'deepmath':
    raise AssertionError("BASE_DIR = '%s' doesn't end with deepmath" %
                         BASE_DIR)

  # Scan all files for errors
  errors = []
  for root, _, filenames in os.walk(BASE_DIR):
    for f in filenames:
      _, ext = os.path.splitext(f)
      if ext in ('.py', '.h', 'cc'):
        # Slurp in the whole file
        path = os.path.join(root, f)
        short = path[len(BASE_DIR) + 1:]
        if six.PY3:
          contents = open(path, encoding='utf-8').read()
        else:
          contents = open(path).read()

        # Ignore empty files
        if not contents:
          continue

        # Check for licenses
        if not LICENSE_RE[ext].match(contents):
          errors.append('%s: missing license' % short)

        if ext == '.py':
          # Check for futures
          futures = frozenset(FUTURE_RE.findall(contents))
          missing = REQUIRED_FUTURES - futures
          if missing:
            errors.append('%s: missing futures [%s]' %
                          (short, ', '.join(missing)))

          # Check for bad things
          if short != 'tools/boilerplate_test.py':
            bads = {'FLAGS.test_tmpdir': 'self.get_temp_dir()'}
            for bad, good in bads.items():
              if bad in contents:
                errors.append('%s: %s is bad, use %s instead' %
                              (short, bad, good))

  # Report errors
  print()
  for error in errors:
    print(error)
  if errors:
    print('\n%d total errors' % len(errors))
    sys.exit(1)


if __name__ == '__main__':
  main()
