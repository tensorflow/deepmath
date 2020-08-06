"""Open source mocks for reading/writing recordIO."""

from typing import Text


def read_protos_from_recordio(pattern: Text, proto_class):
  del pattern
  del proto_class
  assert False, 'Recordio input is not supported.'


def write_protos_to_recordio(filename: Text, protos):
  del filename
  del protos
  assert False, 'Recordio output is not supported.'
