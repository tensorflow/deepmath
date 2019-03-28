"""Open source mocks for reading/writing recordIO."""

from __future__ import absolute_import
from __future__ import division
# Import Type Annotations
from __future__ import print_function

from typing import Text


def read_protos_from_recordio(pattern: Text, proto_class):
  del pattern
  del proto_class
  assert False, 'Recordio input is not supported.'


def write_protos_to_recordio(filename: Text, protos):
  del filename
  del protos
  assert False, 'Recordio output is not supported.'
